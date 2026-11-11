"""Group words into text lines and cells.

A wide horizontal gap starts a new cell, so "Invoice No: 12   Date: 05/03/2024" becomes two cells,
and every cell keeps its x-position. Positions let values be matched to the column header above
them (line-item tables, summary tables) and keep two-column headers apart.
"""
import re
from dataclasses import dataclass, field
from statistics import median


@dataclass
class Cell:
    text: str
    x0: float  # left edge, 0..1 of page width
    x1: float  # right edge


@dataclass
class Line:
    index: int
    page: int
    text: str
    y0: float
    y1: float
    conf: float
    gap_before: bool = False  # True when there is a clear vertical gap above this line
    cell_boxes: list = field(default_factory=list)

    def cells(self):
        if self.cell_boxes:
            return [c.text for c in self.cell_boxes]
        return [c.strip() for c in re.split(r"\s{2,}", self.text) if c.strip()]


# OCR reads table borders as "|", "_", "—" ...: they are not text and would distort line grouping.
_BORDER_RE = re.compile(r"^[|_\-—–=~\[\]{}()!¦/\\.,'`\"]+$")


def _is_border(word) -> bool:
    return word.conf < 1.0 and bool(_BORDER_RE.match(word.text))  # only OCR words; PDF text is exact


def _group_into_lines(words):
    """Chain words left to right: each word joins the line whose last word is just left of it and at
    the same height. Following neighbour to neighbour tolerates the slight tilt and perspective of
    phone photos, where a printed line drifts up or down across the page."""
    lines = []
    for w in sorted(words, key=lambda w: w.x0):
        h = w.y1 - w.y0
        best, best_dy = None, None
        for line in lines:
            last = line[-1]
            if last.x1 > w.x0 + 0.3 * h:  # the line already extends past this word
                continue
            dy = abs(w.yc - last.yc)
            if dy <= 0.5 * min(h, last.y1 - last.y0) and (best is None or dy < best_dy):
                best, best_dy = line, dy
        if best is None:
            lines.append([w])
        else:
            best.append(w)
    return sorted(lines, key=lambda ws: sum(x.yc for x in ws) / len(ws))


def build_lines(pages):
    all_words = [w for p in pages for w in p.words if not _is_border(w)]
    if not all_words:
        return []
    heights = [w.y1 - w.y0 for w in all_words if w.y1 > w.y0]
    median_h = median(heights) if heights else 0.01
    char_w = median((w.x1 - w.x0) / max(1, len(w.text)) for w in all_words) or 0.01

    groups = []  # (page_index, [words])
    for page in pages:
        words = [w for w in page.words if not _is_border(w)]
        groups.extend((page.index, line) for line in _group_into_lines(words))

    lines = []
    prev = None
    for page_idx, words in groups:
        words = sorted(words, key=lambda w: w.x0)
        cell_words = [[words[0]]]
        for a, b in zip(words, words[1:]):
            if b.x0 - a.x1 > 2.2 * char_w:
                cell_words.append([b])
            else:
                cell_words[-1].append(b)
        cells = [Cell(" ".join(w.text for w in ws), ws[0].x0, ws[-1].x1) for ws in cell_words]
        y0 = min(w.y0 for w in words)
        y1 = max(w.y1 for w in words)
        conf = sum(w.conf for w in words) / len(words)
        gap_before = prev is not None and prev.page == page_idx and (y0 - prev.y1) > 0.8 * median_h
        line = Line(len(lines), page_idx, "  ".join(c.text for c in cells), y0, y1, conf, gap_before, cells)
        lines.append(line)
        prev = line
    return lines
