"""Public entry point: parse one invoice file (PDF, scan or photo) into structured data."""
from pathlib import Path

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
        result["source_file"] = filename
        result["pages"] = len(pages)
        result["extraction"] = {"method": overall_method(pages), "label_matcher": self.matcher.backend}
        return result

    def parse_path(self, path) -> dict:
        p = Path(path)
        return self.parse_bytes(p.name, p.read_bytes())


_default_parser = None


def get_parser() -> InvoiceParser:
    global _default_parser
    if _default_parser is None:
        _default_parser = InvoiceParser()
    return _default_parser
