"""Public entry point: parse one invoice file (PDF, scan or photo) into structured data."""
import re
from pathlib import Path

from . import fast_ocr
from .extract import extract_invoice
from .ingest import load_pages, overall_method
from .layout import build_lines
from .matcher import LabelMatcher


class InvoiceParser:
    def __init__(self, matcher=None):
        self._matcher = matcher

    @property
    def matcher(self) -> LabelMatcher:
        if self._matcher is None:  # loads the embedding model on first use
            self._matcher = LabelMatcher()
        return self._matcher

    def parse_bytes(self, filename: str, data: bytes) -> dict:
        pages = load_pages(filename, data)
        lines = build_lines(pages)
        result = extract_invoice(lines, self.matcher)
        if _second_look(pages, result):
            result = extract_invoice(build_lines(pages), self.matcher)
        for page in pages:
            page.ocr_lines = None  # the line images are not needed any more
        result["source_file"] = filename
        result["pages"] = len(pages)
        result["extraction"] = {"method": overall_method(pages), "label_matcher": self.matcher.backend}
        return result

    def parse_path(self, path) -> dict:
        p = Path(path)
        return self.parse_bytes(p.name, p.read_bytes())


def _letters_digits(text):
    return re.sub(r"[^0-9a-z]", "", (text or "").lower())


def _second_look(pages, result):
    """The lines that gave the item descriptions and the party names, read again by the handwriting
    model where the small model was not sure (fast_ocr.reread_lines): a printed form with a few
    words written in reads well as a whole, so its handwritten words are otherwise read only once.
    True when the words of a page changed (the fields are then found again)."""
    targets = [item.get("description") for item in result.get("line_items") or []]
    targets += [((result["fields"].get(party) or {}).get("value") or {}).get("name") for party in ("seller", "buyer")]
    targets = {t for t in map(_letters_digits, targets) if len(t) >= 3}
    changed = False
    for page in pages:
        if page.ocr_lines is None or not targets:
            continue
        picked = []
        for k, (text, _, _) in enumerate(page.ocr_lines.read):
            t = _letters_digits(text)
            if len(t) >= 3 and re.search(r"[a-z]", t) and any(t in x or x in t for x in targets):
                picked.append(k)
        words = fast_ocr.reread_lines(page.ocr_lines, picked)
        if words is not None:
            page.words = words
            changed = True
    return changed


_default_parser = None


def get_parser() -> InvoiceParser:
    global _default_parser
    if _default_parser is None:
        _default_parser = InvoiceParser()
    return _default_parser
