"""Shared data classes for words, lines and pages."""
from dataclasses import dataclass


@dataclass
class Word:
    """One token of text. Coordinates are normalised to 0..1 of the page size."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float = 1.0  # OCR confidence 0..1 (1.0 for text embedded in a PDF)
    page: int = 0

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class Page:
    index: int
    words: list
    method: str  # "digital" (PDF text layer) or "ocr" (scan / photo)
    ocr_conf: float  # mean OCR confidence of the page (1.0 for digital)
