"""Group words into text lines and cells.

A wide horizontal gap starts a new cell, so "Invoice No: 12   Date: 05/03/2024" becomes two cells,
and every cell keeps its x-position. Positions let values be matched to the column header above
them (line-item tables, summary tables) and keep two-column headers apart.
"""
import re
from dataclasses import dataclass, field
from statistics import median

from .models import Word


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


_RUPEE_HEAD_RE = re.compile(r"(?i)^(rs\.?|rupees?|₹)$")
_PAISE_HEAD_RE = re.compile(r"(?i)^(p\.?|ps\.?|paise)$")
_RUPEES_RE = re.compile(r"^\d[\d,]*$")
_PAISE_RE = re.compile(r"^\d{1,2}$")
# "3805=93", "42288/14": rupees and paise written with a stroke between them
_STROKE_DECIMAL_RE = re.compile(r"^(\d[\d,]*)\s*([=/:|])\s*(\d{2})$")


def _same_row(a, b):
    return min(a.y1, b.y1) - max(a.y0, b.y0) >= 0.4 * min(a.y1 - a.y0, b.y1 - b.y0)


def join_paise(words):
    """Bill books print amounts in two columns, 'Rs.' and 'P.' (rupees | paise), and the paise are
    written in a box of their own: '42288 | 14' is 42288.14. On an OCR'd page with such a header,
    a whole number right of the 'Rs.' header followed closely by a 1-2 digit number that ends its
    row (the paise box is the last box of the row) become one amount. Rupees and paise written with
    a stroke between them become one amount too: '3805=93' always, '42288/14' only at the end of a
    row on such a page. Positions are not compared with the 'P.' header itself: on a tilted photo
    the columns lean. Words of a PDF text layer (confidence 1.0) are never changed."""
    ocr = [w for w in words if w.conf < 1.0]
    if not ocr:
        return words
    heads = [(r, p) for r in ocr if _RUPEE_HEAD_RE.match(r.text)
             for p in ocr if _PAISE_HEAD_RE.match(p.text) and p.x0 > r.x1 and _same_row(r, p)
             and p.x0 - r.x1 < 0.25]
    rupee_head = min(heads, key=lambda rp: rp[1].x0 - rp[0].x1)[0] if heads else None
    if rupee_head is None:  # the small 'P.' of the header was not read: the 'Rs.' heading alone will do
        rupee_head = next((w for w in ocr if _RUPEE_HEAD_RE.match(w.text) and w.text.lower() != "₹"), None)

    def ends_row(w):
        return not any(o is not w and o.x0 > w.x1 - 0.005 and _same_row(o, w) for o in words)

    merged, used = {}, set()  # index -> joined word; indexes of paise words joined into an amount
    for i, w in enumerate(words):
        if w.conf >= 1.0 or i in used:
            continue
        m = _STROKE_DECIMAL_RE.match(w.text)
        if m and (m.group(2) == "=" or (rupee_head and w.x0 > rupee_head.x0 - 0.1 and ends_row(w))):
            merged[i] = Word(f"{m.group(1)}.{m.group(3)}", w.x0, w.y0, w.x1, w.y1, w.conf, w.page)
        elif (rupee_head and w.y0 > rupee_head.y1 and _RUPEES_RE.match(w.text) and len(w.text) >= 2
              and w.x1 > rupee_head.x0):
            h = w.y1 - w.y0
            for j, p in enumerate(words):
                if (j == i or j in used or j in merged or p.conf >= 1.0 or not _PAISE_RE.match(p.text)
                        or not -0.2 * h <= p.x0 - w.x1 <= 0.15 or not _same_row(w, p) or not ends_row(p)):
                    continue
                merged[i] = Word(f"{w.text}.{int(p.text):02d}", w.x0, min(w.y0, p.y0), p.x1, max(w.y1, p.y1),
                                 min(w.conf, p.conf), w.page)
                used.add(j)
                break
    return [merged.get(i, w) for i, w in enumerate(words) if i not in used]


def build_lines(pages):
    for page in pages:  # amounts split over 'Rs.' and 'P.' columns become one amount
        page.words = join_paise(page.words)
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
