"""Map a label such as "Vendor", "Billed by" or "Amount payable" to a canonical field.

Three steps, in order:
  1. exact match against the synonym list
  2. typo-tolerant fuzzy match (rapidfuzz)
  3. semantic match: cosine similarity of sentence embeddings
     (falls back to character n-gram similarity if the model cannot be loaded)

Speed for large volumes:
  * The embedding model is loaded lazily, only when a label is not covered by steps 1-2.
  * Every semantic decision is stored in a persistent label cache (data/label_cache.json), so
    labels that repeat across thousands of invoices never need the model again.
"""
import atexit
import hashlib
import json
import logging
import math
import os
import re
import threading
from collections import Counter, namedtuple
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz

from . import config
from .schema import ALONE_ONLY, FIELD_BLOCKLIST, FIELD_SYNONYMS

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

log = logging.getLogger(__name__)

Match = namedtuple("Match", "field score method phrase")
_MISSING = object()


def normalize_label(text: str) -> str:
    s = (text or "").lower()
    s = s.replace("&", " and ").replace("#", " no ").replace(".", "")
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = re.sub(r"^(the|your|our) ", "", s)
    return s


def _char_ngrams(s: str, n: int = 3) -> Counter:
    s = f" {s} "
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def _cosine(a: Counter, b: Counter) -> float:
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


class LabelMatcher:
    def __init__(self, use_embeddings=None, model_name=None, fuzzy_min=0.88, embed_min=0.74,
                 lexical_min=0.6, cache_path=None):
        self.use_embeddings = config.USE_EMBEDDINGS if use_embeddings is None else use_embeddings
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.fuzzy_min = fuzzy_min
        self.embed_min = embed_min
        self.lexical_min = lexical_min
        self.entries = [(normalize_label(p), field)
                        for field, phrases in FIELD_SYNONYMS.items() for p in phrases]
        self._ngrams = [_char_ngrams(p) for p, _ in self.entries]
        self._model = None
        self._syn_vecs = None
        self._model_failed = False
        self._model_lock = threading.Lock()
        self._encode_lock = threading.Lock()
        self._cache = {}  # in-memory: (label, semantic) -> Match | None
        # Cached decisions are only valid for the same synonyms and guard words.
        rules = json.dumps([FIELD_SYNONYMS, {k: sorted(v) for k, v in FIELD_BLOCKLIST.items()}], sort_keys=True)
        self._schema_key = hashlib.sha1(rules.encode("utf-8")).hexdigest()[:12]

        self._disk_path = Path(cache_path) if cache_path else (
            config.LABEL_CACHE_PATH if self.use_embeddings else None)
        self._disk_lock = threading.Lock()
        self._disk = self._load_disk_cache()
        self._unsaved = 0
        if self._disk_path:
            atexit.register(self.flush)

    # --- model and persistent cache -------------------------------------------------
    @property
    def backend(self) -> str:
        if self.use_embeddings and not self._model_failed:
            return f"embeddings ({self.model_name}) + rapidfuzz"
        return "lexical (rapidfuzz + character n-grams)"

    def warm_up(self):
        """Load the embedding model now, e.g. from a background thread when the app starts."""
        self._ensure_model()
        return self

    def _ensure_model(self):
        if not self.use_embeddings or self._model_failed:
            return None
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None and not self._model_failed:
                try:
                    from sentence_transformers import SentenceTransformer
                    try:  # use the local copy without any network round-trip
                        model = SentenceTransformer(self.model_name, local_files_only=True)
                    except Exception:  # first run: download the model once
                        model = SentenceTransformer(self.model_name)
                    self._syn_vecs = model.encode([p for p, _ in self.entries], normalize_embeddings=True)
                    self._model = model
                except Exception as exc:  # package missing or no internet on first run
                    log.warning("Embedding model unavailable, using lexical matching: %s", exc)
                    self._model_failed = True
        return self._model

    def _load_disk_cache(self):
        if not self._disk_path or not self._disk_path.exists():
            return {}
        try:
            data = json.loads(self._disk_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if data.get("model") != self.model_name or data.get("schema") != self._schema_key:
            return {}  # decisions from another model or older synonym rules are not reused
        return data.get("labels", {})

    def _remember(self, key, match):
        if not self._disk_path:
            return
        with self._disk_lock:
            self._disk[key] = list(match) if match else None
            self._unsaved += 1
        if self._unsaved >= 25:
            self.flush()

    def flush(self):
        """Write new label decisions to disk (atomic replace)."""
        if not self._disk_path or not self._unsaved:
            return
        with self._disk_lock:
            payload = json.dumps({"model": self.model_name, "schema": self._schema_key,
                                  "labels": self._disk}, indent=0)
            self._disk_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._disk_path.with_suffix(".tmp")
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, self._disk_path)
            self._unsaved = 0

    # --- matching -------------------------------------------------------------------
    @staticmethod
    def _blocked(field, tokens):
        return any(t in FIELD_BLOCKLIST.get(field, ()) for t in tokens)

    def match(self, label, semantic=True):
        """Return a Match or None. semantic=False skips embeddings (exact/fuzzy only)."""
        key = normalize_label(label)
        if not key:
            return None
        cache_key = (key, semantic)
        if cache_key not in self._cache:
            self._cache[cache_key] = self._match(key, semantic)
        return self._cache[cache_key]

    def _match(self, key, semantic):
        tokens = set(key.split())
        if len(tokens) > 6:  # real labels are short; long sentences are not labels
            return None
        for phrase, field in self.entries:
            # a listed phrase is meant as it is: "total amount after tax" is the total although it has
            # the word "tax" (the blocklist guards the looser fuzzy / semantic matches below)
            if phrase == key:
                return Match(field, 1.0, "exact", phrase)

        best = None
        for phrase, field in self.entries:
            if self._blocked(field, tokens) or phrase in ALONE_ONLY:  # "no", "name": exact matches only
                continue
            score = fuzz.ratio(key, phrase) / 100.0
            if score >= self.fuzzy_min and (best is None or score > best.score):
                best = Match(field, score, "fuzzy", phrase)
        if best or not semantic:
            return best

        if self.use_embeddings and not self._model_failed:
            stored = self._disk.get(key, _MISSING)
            if stored is not _MISSING:
                return Match(*stored) if stored else None
            if self._ensure_model() is not None:
                result = self._semantic(key, tokens)
                self._remember(key, result)
                return result

        vec = _char_ngrams(key)
        for (phrase, field), grams in zip(self.entries, self._ngrams):
            if self._blocked(field, tokens) or phrase in ALONE_ONLY:
                continue
            score = _cosine(vec, grams)
            if score >= self.lexical_min and (best is None or score > best.score):
                best = Match(field, score, "lexical", phrase)
        return best

    def _semantic(self, key, tokens):
        with self._encode_lock:  # one encode at a time; the model is shared by worker threads
            vec = self._model.encode([key], normalize_embeddings=True)[0]
        sims = self._syn_vecs @ vec
        for idx in np.argsort(-sims)[:12]:
            phrase, field = self.entries[idx]
            if self._blocked(field, tokens) or phrase in ALONE_ONLY:
                continue
            score = float(sims[idx])
            return Match(field, score, "embedding", phrase) if score >= self.embed_min else None
        return None
