"""Public entry point: parse one invoice file (PDF, scan or photo) into structured data."""
import re
from pathlib import Path

from . import fast_ocr, spelling
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
        spelling.fix_parties(result, pages)
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
    """The lines that gave the item descriptions and the party names and addresses, read again by the
    handwriting model where the small model was not sure (fast_ocr.reread_lines): a printed form with a
    few words written in reads well as a whole, so its handwritten words are otherwise read only once.
    True when the words of a page changed (the fields are then found again). The address lines come
    last (a page re-reads at most fast_ocr.SECOND_LOOK_LINES lines), and only on a page with doubtful
    lines: on one that reads well as a whole their misread words are left to spelling.fix_parties, as
    reading them again cost a printed photo about a second."""
    parties = [(result["fields"].get(party) or {}).get("value") or {} for party in ("seller", "buyer")]
    targets = [item.get("description") for item in result.get("line_items") or []] + [p.get("name") for p in parties]
    groups = [{t for t in map(_letters_digits, texts) if len(t) >= 3}
              for texts in (targets, [p.get("address") for p in parties])]
    changed = False
    for page in pages:
        if page.ocr_lines is None or not any(groups):
            continue
        picked = []
        for targets in groups if page.ocr_lines.sure < fast_ocr.SECOND_LOOK_PAGE_BELOW else groups[:1]:
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
