"""Turn label/value pairs, party blocks and tables into canonical invoice fields.

Strategy (no training needed):
  1. Each line is split into cells at wide gaps; every cell keeps its x-position.
     Cells are scanned for "Label: value" pairs, or for a label written without a colon.
     Labels are matched to canonical fields by the synonym + embedding matcher (matcher.py).
  2. Summary tables (headers such as "Net worth | VAT | Gross worth" above a "Total" row)
     are read by matching each amount to the header above it.
  3. Scalar fields (number, date, amount) take the best-scoring candidate. Dates use the
     day/month order detected for the whole document.
  4. Party fields (seller/buyer) take the label's column plus the address lines below it,
     and tax-ID / e-mail / phone lines that follow after a gap.
  5. Line items are read column by column under the item-table header (Qty, Price, Amount ...);
     wrapped description lines are joined to their item.
  6. Totals are cross-checked: subtotal + tax - discount = total, line items = subtotal.
"""
import re
from dataclasses import dataclass

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from . import normalize as nz
from .matcher import Match, normalize_label
from .schema import ALONE_ONLY, PARTY_FIELDS

REQUIRED_FIELDS = ("invoice_number", "invoice_date", "seller", "total")

# A label is up to ~40 characters of words, digits and punctuation, ending in a colon.
_LABEL_SEG_RE = re.compile(r"([A-Za-z][A-Za-z0-9 .#/()&'%@\-]{1,40}?)\s*:")
_TABLE_WORDS = {"description", "item", "items", "qty", "quantity", "rate", "price", "unit", "amount",
                "hsn", "sac", "particulars", "product", "uom", "total", "value", "sl", "sr", "discount", "tax"}
_TABLE_STRONG = {"qty", "quantity", "rate", "price", "unit", "amount", "hsn", "sac", "uom"}
_NUM_TOKEN_RE = re.compile(r"^-?\d[\d,.']*%?$")
_CELL_NUMBER_RE = re.compile(r"^\(?-?\d[\d.,' ]*\)?$")
_COMPANY_RE = re.compile(
    r"(?i)\b(pvt|private|ltd|limited|llp|inc|llc|corp|corporation|gmbh|plc|pte|company|"
    r"enterprises|traders|solutions|services|industries|technologies|group)\b"
    r"|&\s*co\b|\bco\.(?![\w-])")  # "& Co" / "Co." but not product codes such as "TEC-CO-4574"
# Lines that belong to a party block even after a blank gap (tax ID, bank, contact details).
_DETAIL_RE = re.compile(r"(?i)^\s*(tax|vat|gst|gstin|pan|tin|ein|abn|iban|swift|bic|bank|a/?c|account|"
                        r"email|e-mail|phone|tel|mobile|mob|fax|web|website|reg)\b")

PARTY_STOP = ("invoice no", "invoice number", "invoice date", "due date", "date", "description",
              "item", "qty", "quantity", "total", "subtotal", "sub total", "grand total", "amount",
              "terms", "notes", "payment", "thank", "signature", "for ")
ITEM_STOP = ("total", "sub total", "subtotal", "grand total", "amount in words", "net amount",
             "amount payable", "terms", "notes", "bank", "thank", "signature", "tax", "igst", "cgst",
             "sgst", "gst", "vat", "discount", "round off", "declaration", "payment", "summary",
             "shipping", "freight")
_HASH_NUMBER_RE = re.compile(r"(?:no\.?\s*)?#\s*:?\s*([A-Za-z0-9][A-Za-z0-9\-/.]{0,24})", re.I)
TOTAL_FIELDS = {"subtotal", "tax", "total", "discount", "shipping"}
# "To" / "From" without a colon are labels only when they stand alone in their cell
# ("TO" above an address block), not in sentences like "To be paid within 30 days".
WEAK_LABELS = {"to", "from"}

# Line-item column roles, checked in this order ("Net price" is a price, "Net worth" an amount).
_COLUMN_ROLES = (
    ("gross", re.compile(r"\bgross\b|\btotal incl")),
    ("unit_price", re.compile(r"\b(rate|price|unit cost|cost)\b")),
    ("quantity", re.compile(r"\b(qty|quantity|units|hours|hrs|pcs|nos)\b")),
    ("amount", re.compile(r"\b(amount|worth|value|total|net)\b")),
    ("description", re.compile(r"\b(description|item|items|particulars|product|products|service|services|"
                               r"details|goods|article)\b")),
)
_NUMERIC_ROLES = ("quantity", "unit_price", "amount", "gross")
# GST components, also when written as "Output CGST 9%" inside a table row
_TAX_COMPONENT_RE = re.compile(r"(?i)\b(cgst|sgst|igst|utgst)\b")
_IN_WORDS_RE = re.compile(r"(?i)\bin\s*words?")
_LABEL_CELL_RE = re.compile(r"^\s*[A-Za-z][A-Za-z .&/()'-]{1,30}:")  # "State : UP": a label with its value
# "M/s." / "M/S :" / "Messrs." starting a cell: the buyer's name follows (the letterhead's "M/S OM
# TRADERS" has no full stop or colon after M/S, so it is not taken)
_MESSRS_RE = re.compile(r"(?i)^\s*(?:m\s*/\s*s|messrs)\s*[.:][\s.:_…-]*(.*)$")
# very short labels that are only trusted with a colon (OCR noise often produces "po", "to" ...)
_NON_COLON_SKIP = {"po"}
# "for Kavya Infotech" printed above "Authorised Signatory"
_SIGNATURE_RE = re.compile(r"(?i)^\s*for\s+(?:and\s+on\s+behalf\s+of\s+)?(.{3,80}?)\s*[.:]?\s*$")


@dataclass
class Candidate:
    field: str
    score: float
    label: str
    value: str
    line_idx: int
    page: int
    ocr_conf: float
    line_text: str
    explicit: bool  # True for "Label: value" cells, False for label-only cells
    cell_idx: int = 0  # which column (cell) of the line the label sits in


def is_table_header(line) -> bool:
    tokens = set(re.findall(r"[a-z]+", line.text.lower()))
    if re.search(r"\d{3,}", line.text):
        return False  # "Total Amount Before Tax 42288": a labelled amount, column titles hold no figures
    return bool(tokens & _TABLE_STRONG) and len(tokens & _TABLE_WORDS) >= 3


def _starts_with_any(text, prefixes):
    t = text.strip().lower()
    return any(t.startswith(p) for p in prefixes)


def _starts_with_label(text, matcher):
    """True when a cell begins with an exact field label, e.g. 'Bill To Rahul Electronics'."""
    words = text.split()
    for n in range(min(3, len(words)), 0, -1):
        mt = matcher.match(" ".join(words[:n]), semantic=False)
        if mt and mt.method == "exact":
            return True
    return False


def _cell_number(text):
    """The number in a cell that holds only a number (and maybe a currency): '$ 1 234,56' -> 1234.56."""
    s = re.sub(r"(?i)[₹$€£]|\b(INR|USD|EUR|GBP|AUD|CAD|SGD|AED|Rs)\b\.?", "", text)
    s = s.strip(" |_.,;:!'\"`~")  # OCR leaves border bits on numbers: "2,50,000.00_", "22,500.00,"
    if not s or "%" in s or not _CELL_NUMBER_RE.match(s):
        return None
    return nz.parse_amount(s)


_QTY_WITH_UNIT_RE = re.compile(r"^\s*(-?\d[\d.,']*)\s*(?:[A-Za-z.]{1,8})?\s*$")


def _quantity(text):
    """A quantity cell may carry its unit: '2,00 each', '5 pcs', '10 Nos' -> 2.0, 5.0, 10.0."""
    value = _cell_number(text)
    if value is not None:
        return value
    m = _QTY_WITH_UNIT_RE.match(text)
    return nz.parse_amount(m.group(1)) if m else None


def _overlap(a, b):
    return min(a.x1, b.x1) - max(a.x0, b.x0)


def _nearest(cell, columns):
    """The column (header cell) that a value cell sits under: largest overlap, else nearest centre."""
    best = max(columns, key=lambda col: _overlap(col[1], cell))
    if _overlap(best[1], cell) > 0:
        return best
    centre = (cell.x0 + cell.x1) / 2
    return min(columns, key=lambda col: abs((col[1].x0 + col[1].x1) / 2 - centre))


# --- label candidates --------------------------------------------------------------------
def _choose_label(words, matcher):
    """Pick the field label at the end of a colon group, e.g. 'Acme Ltd Vendor' -> 'Vendor'."""
    if not words:
        return "", "", None
    trimmed = list(words)
    while trimmed and re.search(r"\d", trimmed[-1]):  # "VAT 19%" or "Tax (18%)" -> "VAT" / "Tax"
        trimmed.pop()
    if trimmed and len(trimmed) < len(words):
        mt = matcher.match(" ".join(trimmed), semantic=False)
        if mt:
            return " ".join(trimmed), "", mt
    for k in range(min(4, len(words)), 0, -1):
        cand = " ".join(words[-k:])
        mt = matcher.match(cand, semantic=False)
        if mt and k < len(words) and normalize_label(cand) in ALONE_ONLY:
            continue  # "A/c No.", "Bank Name": not the invoice number / the buyer
        if mt:
            return cand, " ".join(words[:-k]), mt
    full = " ".join(words)
    mt = matcher.match(full, semantic=True)
    if mt:
        return full, "", mt
    if len(words) <= 3:
        return full, "", None
    return words[-1], " ".join(words[:-1]), None


def split_segments(text, matcher):
    """'Invoice No: INV-1  Date: 05/03/2024' -> [(label, value, match), ...]"""
    ms = list(_LABEL_SEG_RE.finditer(text))
    segs = []
    for i, m in enumerate(ms):
        label, residue, mt = _choose_label(m.group(1).split(), matcher)
        if residue and segs:
            segs[-1][1] = (segs[-1][1] + " " + residue).strip()
        end = ms[i + 1].start(1) if i + 1 < len(ms) else len(text)
        segs.append([label, text[m.end():end].strip(), mt])
    return [(label, value, mt) for label, value, mt in segs]


def collect_candidates(lines, matcher, skip_lines=()):
    """Return (candidates, label line indexes, (line, column) label positions, total-line indexes)."""
    cands, label_lines, label_cells = [], set(), set()

    def add(mt, label, value, ln, k, explicit, factor=1.0):
        cands.append(Candidate(mt.field, mt.score * factor, label, value, ln.index, ln.page,
                               ln.conf, ln.text, explicit, k))
        label_lines.add(ln.index)
        label_cells.add((ln.index, k))

    for ln in lines:
        if ln.index in skip_lines or is_table_header(ln):
            continue
        cells = ln.cells()
        for k, cell in enumerate(cells):
            if _IN_WORDS_RE.search(cell):
                continue  # "Total Amount in words ...": read by _amount_in_words, never a figure's label
            nxt = cells[k + 1] if k + 1 < len(cells) else ""
            # the value may sit in the next column, unless that column is itself a label
            # ("Invoice No. | State : UP / State Code : 09": the 09 is the state code's)
            next_value = "" if (not nxt or _starts_with_label(nxt, matcher) or _LABEL_CELL_RE.match(nxt)) else nxt
            ms = _MESSRS_RE.match(cell)
            if ms:  # "M/s. ...... SURESH MITTAL": the party billed (a bill book's buyer line)
                value = ms.group(1).strip(" ._:-…") or next_value
                add(Match("buyer", 1.0, "exact", "m s"), "M/s.", value, ln, k, True)
                continue
            if ":" in cell:
                found = False
                for label, value, mt in split_segments(cell, matcher):
                    if mt is not None:
                        add(mt, label, value or next_value, ln, k, True)
                        found = True
                component = None if found else _TAX_COMPONENT_RE.search(cell)
                if component and "gstin" not in cell.lower():
                    # "Add : CGST @ 9 %  3,805.93" (bill books): the tax line, its amount ends the row
                    value = " ".join([cell[component.end():]] + cells[k + 1:]).strip()
                    add(Match("tax", 1.0, "exact", component.group(1).lower()), component.group(1).upper(),
                        value, ln, k, False, 0.9)
            else:
                words = cell.split()
                for n in range(min(4, len(words)), 0, -1):
                    label = " ".join(words[:n])
                    mt = matcher.match(label, semantic=False)
                    # "To" / "From" without a colon only as a heading that is the whole cell ("TO"),
                    # not inside sentences or OCR fragments such as "to)" from "(Ship to)"
                    weak = normalize_label(label) in WEAK_LABELS and not re.fullmatch(
                        r"(?i)\s*(to|from)\s*:?\s*", cell)
                    if mt and normalize_label(label) in _NON_COLON_SKIP:
                        mt = None
                    if mt and normalize_label(label) == "no" and not re.match(r"[\s.:#_-]*(\d|$)",
                                                                             " ".join(words[n:])):
                        mt = None  # "No." is a label when a number (or nothing) follows, not "No. of boxes"
                    # a misread label on a photo ('Involce No. | 15') counts when it is the whole cell and a
                    # number follows in the next one
                    misread = (mt is not None and mt.method == "fuzzy" and mt.score >= 0.9 and n == len(words)
                               and len(normalize_label(label)) >= 8 and re.search(r"\d", next_value))
                    if mt and (mt.method == "exact" or misread) and not weak:  # label-only text must match exactly
                        rest = " ".join(words[n:])
                        if re.fullmatch(r"\(.*\)", rest):  # "Buyer (Bill to)": the brackets belong to the label
                            label, rest = f"{label} {rest}", ""
                        if mt.field in TOTAL_FIELDS:  # "Total | 1 NOS | 2,95,000.00": the amount ends the row
                            value = " ".join([rest] + cells[k + 1:]).strip()
                        elif mt.field in PARTY_FIELDS and not re.search(r"[A-Za-z]{2,}", next_value):
                            # "Buyer (Bill to) | 4500012345": a number in the next box belongs to that box
                            # (e.g. the PO number); the party itself is written below its heading
                            value = rest
                        else:
                            value = " ".join(p for p in (rest, next_value) if p)
                        add(mt, label, value, ln, k, False, 0.9)
                        break
                else:
                    number = _HASH_NUMBER_RE.fullmatch(cell)
                    if number and ln.page == 0 and ln.y0 < 0.3:
                        # "INVOICE / # 5873": a bare '#' in the header is the invoice number
                        add(Match("invoice_number", 0.85, "exact", "#"), "#", number.group(1), ln, k, True)
                        continue
                    component = _TAX_COMPONENT_RE.search(cell)
                    if component and "gstin" not in cell.lower():
                        # "Output CGST 9%": a tax line whose amount sits at the end of the row
                        value = " ".join([cell[component.end():]] + cells[k + 1:]).strip()
                        add(Match("tax", 1.0, "exact", component.group(1).lower()), component.group(1).upper(),
                            value, ln, k, False, 0.9)
    totals_lines = {c.line_idx for c in cands if c.field in TOTAL_FIELDS}
    return cands, label_lines, label_cells, totals_lines


def _summary_table_candidates(lines, matcher):
    """Amounts in a summary table, matched to the column header above them.

        VAT [%]   Net worth   VAT      Gross worth
        Total     $ 194,76    $ 19,48  $ 214,24     -> subtotal 194.76, tax 19.48, total 214.24
    """
    cands, used = [], set()
    for i, ln in enumerate(lines):
        amounts = [(c, _cell_number(c.text)) for c in ln.cell_boxes]
        amounts = [(c, v) for c, v in amounts if v is not None]
        if len(amounts) < 2:
            continue
        header, header_idx = [], None
        for j in range(i - 1, max(-1, i - 4), -1):
            hd = lines[j]
            if hd.page != ln.page or is_table_header(hd):
                break  # an item-table header is not a summary header
            header = []
            for cell in hd.cell_boxes:
                if "%" in cell.text:
                    continue  # rate columns hold percentages, not amounts
                mt = matcher.match(cell.text, semantic=False)  # synonyms only: fast, keeps the label cache clean
                if mt and mt.field in TOTAL_FIELDS:
                    header.append((mt, cell))
            if len(header) >= 2:
                header_idx = j
                break
        if header_idx is None:
            continue
        first = ln.cell_boxes[0]
        row_label = first.text if _cell_number(first.text) is None else ""
        label_match = matcher.match(row_label, semantic=False) if row_label else None
        is_total_row = bool(re.search(r"(?i)\btotal\b", row_label) or (label_match and label_match.field == "total"))
        factor = 1.0 if is_total_row else 0.9  # the "Total" row beats per-rate breakdown rows
        for cell, value in amounts:
            mt, head_cell = _nearest(cell, header)
            cands.append(Candidate(mt.field, mt.score * factor, head_cell.text, cell.text, ln.index, ln.page,
                                   ln.conf, ln.text, True, ln.cell_boxes.index(cell)))
        used.update({ln.index, lines[header_idx].index})
    return cands, used


# --- scalar fields ---------------------------------------------------------------------
def _next_line(lines, label_lines, idx, page):
    if idx + 1 < len(lines):
        nxt = lines[idx + 1]
        if nxt.page == page and nxt.text.strip() and nxt.index not in label_lines:
            return nxt
    return None


def _first_identifier(text):
    for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-/_.]*", text):
        tok = tok.rstrip(".")
        if re.search(r"\d", tok) and not nz.find_date(tok) and not nz.find_gstin(tok):
            return tok  # an invoice / PO number is never a date or a GSTIN
    return None


_DIGIT_LOOKALIKES = str.maketrans("OoSslI|B", "00551118")


def _lookalike_amount(text):
    """A handwritten figure with digits read as look-alike letters: '380S' (5), '499l00t' (1, and the
    stroke after the figure). Only tokens with 3+ real digits, read last of all."""
    for token in reversed(text.split()):
        if len(re.findall(r"\d", token)) < 3:
            continue
        fixed = re.sub(r"[A-Za-z]{1,2}$", "", token.translate(_DIGIT_LOOKALIKES))
        if re.fullmatch(r"\d[\d,]*(\.\d{1,2})?", fixed):
            return nz.parse_amount(fixed)
    return None


def _last_amount(text):
    vals = [nz.parse_amount(m.group(0)) for m in nz.AMOUNT_RE.finditer(nz.strip_dates(text))]
    vals = [v for v in vals if v is not None]
    return vals[-1] if vals else None


# a number with a quantity unit: "1 NOS", "5 pcs", "2 Kgs" (Tally prints the total quantity in the Total row)
_QTY_UNIT_RE = re.compile(r"(?i)(?<![\w.,])\d[\d,.]*\s*(?:nos|no|pcs|pc|pieces?|units?|qty|kgs?|gms?|ltrs?|litres?"
                          r"|mtrs?|meters?|sets?|boxes|box|bags?|pairs?|dozens?|each)\b\.?")


# a number after a phone label: "Mob.:9000012345", "Ph: 0129-2234567", "Contact : 98711 82389"
_PHONE_NUMBER_RE = re.compile(r"(?i)\b(?:mob|mobile|ph|phone|tel|telephone|cell|contact|fax|whatsapp)\b\.?"
                              r"\s*(?:no\.?)?\s*[:\-]?\s*\+?\d[\d\s\-/,]{5,}")


def _amount_same_row(c, lines, label_lines=frozenset()):
    """The amount printed at the same height as its label, further right, when OCR put it on a line
    of its own. On a tilted or curved photo the right end of a row sits a little higher or lower, so
    'Total ... 1 NOS' and '2,95,000.00' can come out as two lines. Lines holding a label of their own
    are skipped: their numbers belong to that label."""
    row = lines[c.line_idx]
    if c.cell_idx >= len(row.cell_boxes):
        return None
    right_of = row.cell_boxes[c.cell_idx].x1
    height = row.y1 - row.y0
    best = None
    for ln in lines:
        if ln.page != row.page or ln.index == row.index or ln.index in label_lines:
            continue
        overlap = min(ln.y1, row.y1) - max(ln.y0, row.y0)
        if overlap < 0.5 * min(height, ln.y1 - ln.y0):
            continue
        for cell in ln.cell_boxes:
            if cell.x0 <= right_of:
                continue
            value = _cell_number(_QTY_UNIT_RE.sub(" ", cell.text))
            if value is not None and (best is None or cell.x0 > best[0]):
                best = (cell.x0, value)  # the rightmost one: the amount column
    return best[1] if best else None


def _values_below(lines, c, label_cells, max_lines=4):
    """Texts that may be the value of a label without one, best first.

    Boxed layouts (Tally and most Indian GST invoices) print the value under its label, in the same
    column: the first non-label cell below that overlaps the label. On a bill book the value is
    written by hand after the label and a little lower ("Invoice No." with "14" beside and below
    it, "Date" with "9-9-2026" right of it, under a neighbour's printed line): the cells below that
    start just right of the label come next."""
    row = lines[c.line_idx]
    if c.cell_idx >= len(row.cell_boxes):
        return []
    label_cell = row.cell_boxes[c.cell_idx]
    out = []
    for ln in lines[c.line_idx + 1: c.line_idx + 1 + max_lines]:
        if ln.page != c.page:
            break
        cells = [(k, cell) for k, cell in enumerate(ln.cell_boxes) if _overlap(cell, label_cell) > 0]
        if any((ln.index, k) in label_cells for k, _ in cells):
            break  # the next label in this column: the value is empty
        texts = [cell.text.strip(" |_[](){}:") for _, cell in cells]
        if any(texts):
            out.append(next(t for t in texts if t))
            break
    for ln in lines[c.line_idx + 1: c.line_idx + 3]:
        if ln.page != c.page:
            break
        for k, cell in enumerate(ln.cell_boxes):
            if (ln.index, k) not in label_cells and label_cell.x1 - 0.02 <= cell.x0 <= label_cell.x1 + 0.12:
                text = cell.text.strip(" |_[](){}:.")
                if text and text not in out:
                    out.append(text)
    return out


def _resolve(c, lines, label_lines, kind, day_first=None, label_cells=frozenset()):
    if c.value:
        sources = [(c.value, 1.0)]
    else:  # the value sits under the label (same column), just right of it lower down, or on the next line
        sources = [(text, 0.95) for text in _values_below(lines, c, label_cells)]
        nxt = _next_line(lines, label_lines, c.line_idx, c.page)
        if nxt is not None:
            sources.append((nxt.text, 0.9))
        if kind not in ("ident", "date", "amount"):
            sources = sources[:1]  # free text: the first place only
    for n, (text, penalty) in enumerate(sources):
        value, penalty = _parse_value(c, lines, label_lines, kind, day_first, text, penalty,
                                      last=n == len(sources) - 1)
        if value is not None:
            return value, penalty
    if kind == "amount" and not sources:
        value = _amount_same_row(c, lines, label_lines)
        return value, (0.9 if value is not None else 1.0)
    return None, 1.0


def _parse_value(c, lines, label_lines, kind, day_first, text, penalty, last=True):
    if kind == "ident":
        value = _first_identifier(text)
    elif kind == "date":
        value = nz.find_date(text, day_first)
        if value is None:  # written in boxes or with strokes: "11 | 09 | 2026", "10 9 26"
            joined = re.sub(r"\b(\d{1,2})\s*[|/ ]\s*(\d{1,2})\s*[|/ ]\s*((?:19|20)?\d{2})\b", r"\1/\2/\3", text)
            value = nz.find_date(joined, day_first) if joined != text else None
    elif kind == "amount":
        text = _QTY_UNIT_RE.sub(" ", text)  # "Total | 1 NOS | 2,95,000.00": the quantity is not money
        text = _PHONE_NUMBER_RE.sub(" ", text)  # "Tax Invoice  Mob.:9000012345": a phone number is not money
        value = _cell_number(text)
        if value is None:
            value = _last_amount(text)
        if value is None:
            # a handwritten stroke after the figure read as a letter ('49900t'), or '/-' ('49,900/-')
            trimmed = re.sub(r"(?<=\d)(?:[A-Za-z]{1,2}\b|/-)", " ", text)
            if trimmed != text:
                value = _cell_number(trimmed)
                value = _last_amount(trimmed) if value is None else value
        if value is None:
            value = _lookalike_amount(text)
        if value is None and last:
            value, penalty = _amount_same_row(c, lines, label_lines), 0.9
    else:
        value = text.strip() or None
    return value, (penalty if value is not None else 1.0)


def _pack(value, c, penalty=1.0):
    return {
        "value": value,
        "confidence": round(min(1.0, c.score * c.ocr_conf * penalty), 3),
        "source_label": c.label,
        "source_text": c.line_text,
        "page": c.page + 1,
    }


def _scalar(cands, field, lines, label_lines, kind, prefer_last=False, day_first=None, label_cells=frozenset()):
    pool = [c for c in cands if c.field == field]
    if prefer_last:
        pool.sort(key=lambda c: (-round(c.score, 2), -c.page, -c.line_idx))
    else:
        pool.sort(key=lambda c: (-round(c.score, 2), c.page, c.line_idx))
    for c in pool:
        value, penalty = _resolve(c, lines, label_lines, kind, day_first, label_cells)
        if value is not None:
            return _pack(value, c, penalty)
    return None


def _total_above_grand_total(cands, lines, label_lines, label_cells, total):
    """'TOTAL 13902.1 | CGST 1251.1 | SGST 1251.1 | GRAND TOTAL 16405' (bill books): with a grand total
    further down, a plain 'Total' above it is the taxable value, i.e. the subtotal."""
    grand = [c.line_idx for c in cands if c.field == "total" and normalize_label(c.label).startswith("grand total")]
    if not grand:
        return None
    plain = sorted((c for c in cands if c.field == "total" and normalize_label(c.label) == "total"
                    and c.line_idx < max(grand)), key=lambda c: c.line_idx)
    for c in plain:
        value, penalty = _resolve(c, lines, label_lines, "amount", label_cells=label_cells)
        if value is not None and 0 < value < total:
            return {**_pack(value, c, penalty), "source_label": f"{c.label} (above Grand Total)"}
    return None


def _tax(cands, lines, label_lines, label_cells=frozenset()):
    """Use CGST+SGST (or IGST) components when they are listed separately, otherwise the tax line."""
    comps = {}
    for c in cands:
        component = _TAX_COMPONENT_RE.search(c.label) if c.field == "tax" else None
        if component:
            comps.setdefault(component.group(1).lower(), c)
    if len(comps) >= 2:
        values, confs, labels = [], [], []
        for c in comps.values():
            value, penalty = _resolve(c, lines, label_lines, "amount", label_cells=label_cells)
            if value is not None:
                values.append(value)
                confs.append(min(1.0, c.score * c.ocr_conf * penalty))
                labels.append(c.label)
        if len(values) >= 2:
            components = {_TAX_COMPONENT_RE.search(lbl).group(1).upper(): v for lbl, v in zip(labels, values)}
            return {"value": round(sum(values), 2), "confidence": round(min(confs), 3),
                    "source_label": " + ".join(labels), "source_text": "", "page": None,
                    "components": components}
    return _scalar(cands, "tax", lines, label_lines, "amount", prefer_last=True, label_cells=label_cells)


# --- parties ---------------------------------------------------------------------------
_LABEL_TAILS = {"name", "ame", "me", "nam", "m/s", "ms", "/s", "s", "to", "buyer", "uyer", "yer", "party", "arty"}


def parse_party(block):
    text = "\n".join(block)
    name = re.split(r"(?i)\s*\b(?:GSTIN|GST|VAT|TIN|PAN|Tax\s*ID|Email|E-mail|Phone|Ph|Tel|Mobile|Mob)\b"
                    r"\s*[:#.-]?", block[0])[0]
    address = []
    for ln in block[1:]:
        ln = ln.lstrip(" _|~.,'\"`‘’“”")  # OCR leaves border bits at the start: "_GSTIN/UIN : ..."
        if re.match(r"(?i)\s*(gstin|gst\s*(no|in)?|uin|pan|vat\s*(no)?|tin|tax\s*id)\b", ln):
            continue  # the tax-ID line: its number is the party's tax_id, not part of the address
        s = nz.strip_contact(re.sub(r"[^\w\s,.\-/#&()':;@+]", "", ln))  # OCR junk characters ("�", ">")
        s = re.sub(r"\s{2,}", " ", s)
        if not re.search(r"[A-Za-z0-9]", s):
            continue
        s = re.sub(r"(?i)^(address|addr|add)\s*[:.-]\s*", "", s).strip(" ,;|")
        # the tax ID itself was taken out; drop its leftover label ("GSTIN/UIN :")
        s = re.sub(r"(?i)\b(GSTIN(\s*/\s*UIN)?|GST\s*(No|IN)?|UIN|PAN|VAT(\s*No)?|TIN|Tax\s*ID)\s*[:#.-]*\s*$",
                   "", s).strip(" ,;|:")
        if s:
            address.append(s)
    # the end of a label cut off by the page edge or a punch hole: 'ame: Kavya Infotech' ('Name:')
    cut = re.match(r"\s*([A-Za-z/.]{1,6})\s*[:;]\s*(\S.*)$", name)
    if cut:
        tail = cut.group(1).lower().strip("./")
        # misread as well ('amo:' for 'ame:')
        if tail in _LABEL_TAILS or (2 < len(tail) <= 4 and max(fuzz.ratio(tail, t) for t in ("ame", "name")) >= 66):
            name = cut.group(2)
    return {
        "name": re.sub(r"\s+", " ", name).strip(" ,;|:-_~/\\") or None,
        "address": ", ".join(address) or None,
        "tax_id": nz.find_tax_id(text),
        "email": nz.find_email(text),
        "phone": nz.find_phone(text),
    }


def _is_detail_line(text):
    return bool(_DETAIL_RE.match(text) or nz.find_tax_id(text) or nz.find_email(text) or nz.find_phone(text))


def _party(cands, field, lines, label_cells, label_lines):
    def rank(c):  # the billed party ("Bill to", "Buyer") beats the delivery address ("Ship to", "Consignee")
        delivery = re.search(r"(?i)\b(ship|shipped|consignee|deliver)", c.label)
        return -round(c.score * (0.95 if delivery else 1.0), 3), c.page, c.line_idx

    pool = sorted((c for c in cands if c.field == field), key=rank)
    for c in pool:
        block = [c.value] if c.value else []
        after_gap = False
        column = _party_column(lines[c.line_idx], c, label_cells)
        i = c.line_idx + 1
        while i < len(lines) and lines[i].page == c.page and len(block) < 9:
            ln = lines[i]
            if is_table_header(ln):
                break
            cs = ln.cells()
            if column and ln.cell_boxes:  # read only what sits under this party's heading (by x-position)
                picked = _cells_in_column(ln, *column)
                text = " ".join(ln.cell_boxes[j].text for j in picked)
                is_label = any((i, j) in label_cells for j in picked)
            elif len(cs) >= 2:  # two-column line: read only the column this party sits in
                if c.cell_idx >= len(cs):
                    break
                text = cs[c.cell_idx]
                is_label = (i, c.cell_idx) in label_cells
            else:
                text = cs[0] if cs else ""
                is_label = (i, c.cell_idx) in label_cells
            if ln.gap_before and block:
                after_gap = True  # after a blank gap only tax-ID / bank / contact lines belong here
            letters = sum(ch.isalpha() for ch in text)
            if (not re.search(r"[A-Za-z0-9]", text) or (letters < 3 and not re.search(r"\d{3}", text))
                    or (ln.conf < 0.5 and letters < 5)):
                i += 1  # OCR noise on scans and photos (specks, border bits: ":", "ole")
                continue
            if not block and fuzz.ratio(normalize_label(text), normalize_label(c.label)) >= 75:
                i += 1  # the heading read twice (blur / double image of 'Bill To:'), not the party
                continue
            if not _is_detail_line(text):
                if (after_gap or is_label or (len(cs) <= 1 and i in label_lines)
                        or _starts_with_any(text, PARTY_STOP)):
                    break
            if text.strip():
                block.append(text.strip())
            i += 1
        if not block:
            continue
        value = parse_party(block)
        if value["name"] and not value["tax_id"]:
            value["tax_id"] = _gstin_below(lines, c, column)
        if value["name"]:
            penalty = 1.0 if c.value else 0.95
            return {
                "value": value,
                "confidence": round(min(1.0, c.score * c.ocr_conf * penalty), 3),
                "source_label": c.label,
                "source_text": c.line_text,
                "page": c.page + 1,
            }
    return None


def _gstin_below(lines, c, column, max_lines=6):
    """A party's GSTIN a few rows under its name, in the party's column. Bill books write each detail
    on its own dotted row ('Name : ...', 'Address : ...', 'GSTIN : ...'), and the space between the
    rows ends the party block before the GSTIN row."""
    for ln in lines[c.line_idx + 1: c.line_idx + 1 + max_lines]:
        if ln.page != c.page or is_table_header(ln):
            break
        picked = _cells_in_column(ln, *column) if column and ln.cell_boxes else range(min(1, len(ln.cell_boxes)))
        for j in picked:
            gstin = nz.find_gstin(ln.cell_boxes[j].text)
            if gstin:
                return gstin
    return None


def _party_column(label_line, c, label_cells):
    """Where a party block sits: (left edges it may start at, right limit or None).

        Bill To:         Ship To:                         Date:  Jan 17 2012
        Alan Haines      San Salvador, San           Ship Mode:  Standard Class

    'Bill To' owns the text from its own left edge up to the next label on its line ('Ship To'),
    so 'Ship Mode' on the right and the Ship-To address never end up in the Bill-To block."""
    boxes = label_line.cell_boxes
    if not boxes or c.cell_idx >= len(boxes):
        return None
    anchors = [boxes[c.cell_idx].x0]
    right = None
    for j in range(c.cell_idx + 1, len(boxes)):
        if (label_line.index, j) in label_cells:
            right = boxes[j].x0
            break
        if j == c.cell_idx + 1 and c.value and boxes[j].text.strip() and boxes[j].text.strip() in c.value:
            anchors.append(boxes[j].x0)  # "Customer | Acme Corp": the block may align with the value
    return anchors, right


def _cells_in_column(ln, anchors, right, reach=0.2, tol=0.04):
    """Indexes of the cells of a line that belong to a party column."""
    picked, prev_x1 = [], None
    for j, box in enumerate(ln.cell_boxes):
        if right is not None and box.x0 >= right - 0.01:
            break
        near = any(a - tol <= box.x0 < a + reach for a in anchors)
        follows = prev_x1 is not None and box.x0 - prev_x1 < 0.06  # same text split by a wide space
        if near or follows or (right is not None and box.x0 >= min(anchors) - tol):
            picked.append(j)
            prev_x1 = box.x1
    return picked


def _column_block(lines, start, anchor, label_cells, max_lines=8):
    """Text under an anchor cell in the same column, until the next label (e.g. 'Consignee')."""
    out = []
    for ln in lines[start + 1: start + 1 + max_lines]:
        if ln.page != lines[start].page or is_table_header(ln):
            break
        right_edge = anchor.x0 + max(anchor.x1 - anchor.x0, 0.2)
        cells = [(k, c) for k, c in enumerate(ln.cell_boxes) if c.x1 > anchor.x0 - 0.02 and c.x0 < right_edge]
        if any((ln.index, k) in label_cells for k, _ in cells):
            break
        if any((ln.index, k) in label_cells and c.x1 <= anchor.x0 for k, c in enumerate(ln.cell_boxes)):
            break  # a label left of the name ('No.', 'Invoice No.'): the form below the letterhead begins
        text = " ".join(c.text for _, c in cells).strip(" |_")
        if text:
            out.append(text)
    return out


def _looks_like_name(name):
    """A party name has a word of 3+ letters and is not a tax ID or a bare number ('O6AABCS7788K1ZP')."""
    if not name or not re.search(r"[A-Za-z]{3,}", name):
        return False
    if not nz.find_gstin(name):
        return True
    # with a GSTIN in it, a word of the name must remain besides the ID and its label: a photo cut the
    # 'G' off 'GSTIN:09AGHPR3344C1ZG', and 'STIN' was taken for a seller's name
    words = [t for t in re.findall(r"[A-Za-z0-9]+", name) if not re.search(r"\d", t)]
    # 'GSTEN:09AGHPR3344C1ZG': a misread 'GSTIN' is still the label
    return any(len(t) >= 3 and t.upper() not in _ID_LABEL_WORDS and fuzz.ratio(t.upper(), "GSTIN") < 75
               for t in words)


_ID_LABEL_WORDS = {"GSTIN", "STIN", "GST", "GSTN", "UIN", "TIN", "PAN", "VAT", "NO", "NUMBER", "REG"}


def _seller_from_signature(lines, label_cells):
    """'for Kavya Infotech' above 'Authorised Signatory' names the seller on most Indian invoices.
    The same name near the top of page 1 gives the address and GSTIN."""
    for i in range(len(lines) - 1, -1, -1):
        if not any(re.search(r"(?i)authori[sz]ed|signator", lines[j].text) for j in range(i, min(len(lines), i + 4))):
            continue
        for cell in lines[i].cell_boxes:
            m = _SIGNATURE_RE.match(cell.text)
            name = m.group(1).strip(" .,:'\"`‘’“”") if m else ""
            if len(name) < 3 or re.search(r"(?i)authori|signator", name):
                continue
            for top in lines[:40]:
                if top.page != 0:
                    break
                for anchor in top.cell_boxes:
                    if fuzz.ratio(name.lower(), anchor.text.lower()) >= 85:
                        value = parse_party([anchor.text] + _column_block(lines, top.index, anchor, label_cells))
                        return {"value": value, "confidence": round(0.75 * top.conf, 3),
                                "source_label": "(name above 'Authorised Signatory')", "source_text": anchor.text,
                                "page": top.page + 1}
            value = {"name": name, "address": None, "tax_id": None, "email": None, "phone": None}
            return {"value": value, "confidence": round(0.6 * lines[i].conf, 3),
                    "source_label": "(name above 'Authorised Signatory')", "source_text": cell.text,
                    "page": lines[i].page + 1}
    return None


_WORDS_CELL_RE = re.compile(r"[A-Za-z][A-Za-z ,.\-]*")
_NOT_NUMBER_WORDS_RE = re.compile(r"(?i)\b(total|tax|add|round|grand|cgst|sgst|igst|utgst|bank|ifsc|amount|"
                                  r"before|after|net|less|discount|freight)\b")


def _words_after_label(lines, i):
    """The amount in words written after 'Total Amount in words' / 'Rupees (in words)' on a bill book:
    the rest of the label's cell and the word-only cells after it, then the leading word-only cells of
    the next two lines ('Seven thousand' / 'eighty only'), stopping at the neighbour column's label
    ('Add : CGST @ ...%'). Returns the texts joined line by line: ['Seven thousand', 'Seven thousand
    eighty only']."""
    pieces, label_right = [], None
    for n, ln in enumerate(lines[i: i + 5]):
        if ln.page != lines[i].page:
            break
        cells = [c.text for c in ln.cell_boxes]
        if n == 0:
            k = next((k for k, t in enumerate(cells) if _IN_WORDS_RE.search(t)), None)
            if k is None:
                return []
            label_right = ln.cell_boxes[k].x1
            rest = cells[k][_IN_WORDS_RE.search(cells[k]).end():].lstrip(" .:)-_…")
            cells = ([rest] if rest else []) + cells[k + 1:]
        elif ln.cell_boxes[0].x0 >= label_right:
            continue  # a row of the column on the right (the tax rows): the words go on below it
        elif not _WORDS_CELL_RE.fullmatch(cells[0].strip(" .:_…")) or _NOT_NUMBER_WORDS_RE.search(cells[0]):
            break  # the next printed item on the left ('Bank Name : ...')
        line_words = []
        for text in cells:
            text = text.strip(" .:_…")
            if not text or not _WORDS_CELL_RE.fullmatch(text) or _NOT_NUMBER_WORDS_RE.search(text):
                break  # the neighbour column's label ('Total Amount Before Tax', 'Add : CGST')
            line_words.append(text)
        pieces.append(" ".join(line_words))
    out, joined = [], ""
    for piece in pieces:
        if piece:
            joined = f"{joined} {piece}".strip()
            out.append(joined)
    return out


def _amount_in_words(lines):
    """'INR Two Lakh Ninety Five Thousand Only' -> (295000.0, line). Indian invoices always print
    the total in words, and words survive OCR better than long numbers."""
    for i, ln in enumerate(lines):
        if re.search(r"(?i)in\s*words", ln.text):
            # longest reading first: 'Seven thousand eighty only', not 'Seven thousand' alone
            for text in reversed(_words_after_label(lines, i)):
                value = nz.words_to_amount(text)
                if value:
                    return value, ln
            texts = [ln.text.split(":", 1)[1] if ":" in ln.text else ""] + [x.text for x in lines[i + 1: i + 3]]
        elif re.search(r"(?i)\bonly\b", ln.text) and re.search(r"(?i)\b(rupees?|inr|usd|dollars?|euros?|pounds?)\b",
                                                               ln.text):
            texts = [ln.text]
        else:
            continue
        for text in texts:
            value = nz.words_to_amount(text)
            if value:
                return value, ln
    return None, None


def _seller_from_tax_id(lines, skip=None):
    """No seller label (common on Indian GST invoices): the first GSTIN on page 1 belongs to the
    seller, so take the text block above it in the same column (name, address, GSTIN). The buyer's
    GSTIN (skip) is passed over: when the seller's own GSTIN is unreadable on a photo, the first
    readable one is often the buyer's or consignee's."""
    for i, ln in enumerate(lines[:60]):
        if ln.page != 0:
            break
        for cell in ln.cell_boxes:
            gstin = nz.find_gstin(cell.text)
            if not gstin or (skip and gstin == skip):
                continue
            block, column = [cell.text], (cell.x0 - 0.03, cell.x1)
            for up in range(i - 1, max(-1, i - 9), -1):
                if up + 1 != i and lines[up + 1].gap_before:
                    break  # a blank gap separates this line from the seller block
                above = lines[up]
                texts = [c.text for c in above.cell_boxes if c.x1 > column[0] and c.x0 < column[1]]
                text = " ".join(texts).strip(" |_")
                if not text or re.fullmatch(r"(?i)(tax |retail |proforma )?invoice|original.*|duplicate.*", text):
                    break
                block.insert(0, text)
            value = parse_party(block)
            if value["name"]:
                return {"value": value, "confidence": round(0.6 * ln.conf, 3),
                        "source_label": "(block with the first GSTIN)", "source_text": cell.text,
                        "page": ln.page + 1}
            return None
    return None


def _header_gstin(lines, seller_name, skip=None):
    """The GSTIN printed in the header around an inferred seller's name (bill books print
    'GSTIN : 09AGHPR3344C1ZG' above or beside the firm's name): on page 1, at most two lines below
    the name, in the top third of the page. The buyer's GSTIN (skip) is passed over."""
    name = re.sub(r"\s+", " ", seller_name or "").lower()
    seller_at = next((ln.index for ln in lines[:12] if ln.page == 0 and len(name) >= 3
                      and fuzz.partial_ratio(name, re.sub(r"\s+", " ", ln.text).lower()) >= 85), None)
    if seller_at is None:
        return None
    for ln in lines[: seller_at + 3]:
        if ln.page != 0 or ln.y0 > 0.35:
            break
        for cell in ln.cell_boxes:
            gstin = nz.find_gstin(cell.text)
            if gstin and gstin != skip:
                return gstin
    return None


_BLOCK_STOP_RE = re.compile(r"(?i)\b(no|date|dated|invoice|bill|state|code|gstin|mob|mobile|phone|tel|email)\b")


def _buyer_from_tax_id(lines, seller_tax_id, start=0):
    """No buyer label could be read (a bill book's 'M/s' printed small, or cut off by the photo): the
    first GSTIN on page 1 that is not the seller's belongs to the buyer, with the lines written above
    it in the same column (name, address) up to the first printed label ('No.', 'Date', ...)."""
    for i, ln in enumerate(lines[:40]):
        if ln.page != 0 or is_table_header(ln):
            break  # the parties are named above the item table
        for cell in ln.cell_boxes if i >= start else ():
            gstin = nz.find_gstin(cell.text)
            if not gstin or gstin == seller_tax_id:
                continue
            block, column = [cell.text], (cell.x0 - 0.05, cell.x1)
            for above in reversed(lines[max(0, i - 5): i]):
                texts = [c.text for c in above.cell_boxes if c.x1 > column[0] and c.x0 < column[1]]
                if not texts:
                    continue  # a row of the other column only ('Vehicle No.' a little higher than the address)
                text = " ".join(texts).strip(" |_:")
                if not text or _BLOCK_STOP_RE.search(text) or nz.find_gstin(text):
                    break
                block.insert(0, text)
            value = parse_party(block)
            if len(block) > 1 and _looks_like_name(value.get("name")):
                value["tax_id"] = gstin
                return {"value": value, "confidence": round(0.5 * ln.conf, 3),
                        "source_label": "(block with the second GSTIN)", "source_text": cell.text,
                        "page": ln.page + 1}
            return None
    return None


def _infer_seller(lines):
    """No 'Seller/Vendor' label found: use the first company-looking line at the top of page 1."""
    for ln in lines[:10]:
        if ln.page != 0 or ln.y0 > 0.3 or is_table_header(ln):
            break  # the issuer is named in the header, never below the item table
        if (_COMPANY_RE.search(ln.text) and not re.search(r"(?i)\binvoice\b", ln.text)
                and not re.search(r"(?i)\b(bill|ship|buyer|customer|client)\b", ln.text)):
            value = parse_party([ln.text.strip()])
            return {"value": value, "confidence": round(0.4 * ln.conf, 3),
                    "source_label": "(inferred from header)", "source_text": ln.text, "page": ln.page + 1}
    return _seller_from_letterhead(lines)


_DOC_TITLE_RE = re.compile(
    r"(?i)\b(invoice|bill|receipt|quotation|quote|estimate|statement|proforma|pro-forma|credit|debit|note|"
    r"original|duplicate|triplicate|copy|recipient|transporter|page|challan|order|purchase|cash|memo|gst|"
    r"tax|date|dated|no|number|paid|unpaid|due|draft)\b")


def _seller_from_letterhead(lines, label_lines=()):
    """The issuer's name printed alone at the top of page 1, before any label (a letterhead or logo text
    such as 'SuperStore'), when the invoice names the seller nowhere else."""
    for ln in lines[:6]:
        if ln.page != 0 or ln.y0 > 0.2:
            break
        if ":" in ln.text:
            break  # labels start ("Date:"): the letterhead is above them
        if ln.conf < 0.6:
            continue  # a misread logo or letterhead ("ae", "ree") is worse than no seller
        for cell in ln.cells():
            text = cell.strip(" |_-:")
            words = text.split()
            if ":" in cell or not 1 <= len(words) <= 6 or _DOC_TITLE_RE.search(text):
                continue
            letters = sum(ch.isalpha() for ch in text)
            if (letters < 4 or letters < 0.7 * len(text.replace(" ", "")) or nz.find_email(text)
                    or nz.find_phone(text) or re.search(r"(?i)www\.|https?://|\.(com|in|net|org)\b", text)):
                continue
            value = parse_party([text])
            if value["name"]:
                return {"value": value, "confidence": round(0.45 * ln.conf, 3),
                        "source_label": "(name at the top of the page)", "source_text": ln.text,
                        "page": ln.page + 1}
    return None


# --- line items --------------------------------------------------------------------------
def parse_item_row(text):
    """Fallback for tables without usable column headers: read qty / rate / amount from the
    trailing numbers of a row such as 'Widget A  2  150.00  300.00'."""
    toks = text.split()
    pos = []
    j = len(toks) - 1
    while j >= 0 and _NUM_TOKEN_RE.match(toks[j].strip("₹$€£()")):
        pos.insert(0, j)
        j -= 1
    if not pos:
        return None
    vals = [nz.parse_amount(toks[p].strip("₹$€£()")) for p in pos]
    n = len(vals)
    qty = rate = None
    start = None
    for k in range(n - 1, 1, -1):  # look for qty x rate = amount in the trailing numbers
        a, b, c = vals[k - 2], vals[k - 1], vals[k]
        if None in (a, b, c):
            continue
        if abs(a * b - c) <= max(0.02, 0.005 * abs(c)):
            qty, rate, start = a, b, k - 2
            amount = c
            break
    else:
        amount = vals[-1]
        if n >= 2:
            rate, start = vals[-2], n - 2
        else:
            start = n - 1
    desc = " ".join(toks[: pos[start]]).strip(" -|")
    if not re.search(r"[A-Za-z]", desc) or amount is None:
        return None
    return {"description": desc, "quantity": _whole(qty), "unit_price": rate, "amount": amount}


def _whole(qty):
    return int(qty) if qty is not None and float(qty).is_integer() else qty  # "2", not "2.0"


def _column_roles(header_line):
    columns = []
    for cell in header_line.cell_boxes:
        text = normalize_label(cell.text)
        role = "skip"
        if "%" not in cell.text:
            for name, pattern in _COLUMN_ROLES:
                if pattern.search(text):
                    role = name
                    break
        columns.append((role, cell))
    return columns


def _qty_rate_from_text(text, amount, qty, rate):
    """Quantity and rate when OCR merged their columns ('1 NOS 2,50,000.00'): the two numbers of the row
    whose product is exactly the amount. Values already read from their own column must agree."""
    nums = [v for v in (_cell_number(t) for t in text.split()) if v is not None and v > 0]
    pairs = [(nums[i], nums[i + 1]) for i in range(len(nums) - 1)]
    pairs += [(nums[i], nums[j]) for i in range(len(nums)) for j in range(i + 2, len(nums))]
    for q, r in pairs:
        if (abs(q * r - amount) <= max(0.05, 0.0005 * abs(amount)) and q <= 100000
                and (qty is None or abs(qty - q) < 1e-6) and (rate is None or abs(rate - r) <= 0.01)):
            return q, r
    return qty, rate


def _line_items(lines, stop_lines):
    header_idx = next((i for i, ln in enumerate(lines) if is_table_header(ln)), None)
    if header_idx is None:
        return [], False
    columns = _column_roles(lines[header_idx])
    roles = {r for r, _ in columns}
    if not roles & {"amount", "gross"}:  # headers we cannot read: fall back to the row heuristic
        items = []
        for ln in lines[header_idx + 1:]:
            if ln.index in stop_lines or _starts_with_any(ln.text, ITEM_STOP):
                break
            if _TAX_COMPONENT_RE.search(ln.text):
                continue  # "Output CGST 9%" rows inside the table are tax, not items
            row = parse_item_row(ln.text)
            if row:
                items.append(row)
        return items, True

    items = []
    for ln in lines[header_idx + 1:]:
        if ln.index in stop_lines or _starts_with_any(ln.text, ITEM_STOP):
            break
        if is_table_header(ln):  # header repeated on a new page
            columns = _column_roles(ln)
            continue
        if _TAX_COMPONENT_RE.search(ln.text):
            continue  # "Output CGST 9%" rows inside the table are tax, not items
        row = {"description": [], "quantity": None, "unit_price": None, "amount": None, "gross": None}
        for cell in ln.cell_boxes:
            role = _nearest(cell, [(r, c) for r, c in columns])[0]
            if role == "description":
                row["description"].append(cell.text)
            elif role in _NUMERIC_ROLES and row[role] is None:
                row[role] = _quantity(cell.text) if role == "quantity" else _cell_number(cell.text)
        amount = row["amount"] if row["amount"] is not None else row["gross"]
        desc = re.sub(r"^\d{1,3}\s*[|.)]\s*", "", " ".join(row["description"])).strip(" -|") or None  # drop "1|"
        if amount is not None and (row["quantity"] is None or row["unit_price"] is None):
            row["quantity"], row["unit_price"] = _qty_rate_from_text(ln.text, amount, row["quantity"],
                                                                     row["unit_price"])
        if amount is not None:
            items.append({"description": desc, "quantity": _whole(row["quantity"]),
                          "unit_price": row["unit_price"], "amount": amount})
        elif desc and items and row["quantity"] is None and row["unit_price"] is None:
            items[-1]["description"] = f"{items[-1]['description'] or ''} {desc}".strip()  # wrapped line
    return items, True


# --- checks and assembly -------------------------------------------------------------------
CHECK_MESSAGES = {
    "subtotal_plus_tax_minus_discount_equals_total": "Totals do not add up: subtotal + tax + shipping - discount "
                                                     "is not the total.",
    "line_items_sum_equals_subtotal": "Line items do not add up to the subtotal.",
    "total_matches_amount_in_words": "The total in figures differs from the amount in words.",
    "seller_gstin_valid": "Seller GSTIN is not valid (its check digit does not match).",
    "buyer_gstin_valid": "Buyer GSTIN is not valid (its check digit does not match).",
    "cgst_equals_sgst": "CGST and SGST should be equal.",
    "gst_type_matches_states": "GST type does not match the states: same state needs CGST + SGST, "
                               "different states need IGST.",
}


def _gst_checks(fields):
    """Indian GST rules that an accounts team checks before booking an invoice."""
    out = {}
    seller = ((fields.get("seller") or {}).get("value") or {}).get("tax_id")
    buyer = ((fields.get("buyer") or {}).get("value") or {}).get("tax_id")
    gstins = {}
    for check, gstin in (("seller_gstin_valid", seller), ("buyer_gstin_valid", buyer)):
        if gstin and nz.GSTIN_RE.fullmatch(gstin):
            out[check] = nz.gstin_checksum_ok(gstin)
            gstins[check] = gstin
    components = (fields.get("tax") or {}).get("components") or {}
    if "CGST" in components and "SGST" in components:
        out["cgst_equals_sgst"] = abs(components["CGST"] - components["SGST"]) <= 0.05
    if len(gstins) == 2 and components:
        same_state = seller[:2] == buyer[:2]  # the first two digits are the state code
        out["gst_type_matches_states"] = ("IGST" not in components) if same_state else (
            "IGST" in components and "CGST" not in components)
    return out


def _one_digit_apart(a, b):
    """True when two amounts differ in exactly one digit: the typical OCR misread (22,600 for 22,500)."""
    x, y = f"{abs(a):.2f}", f"{abs(b):.2f}"
    return len(x) == len(y) and sum(p != q for p, q in zip(x, y)) == 1


def _correct(field, new_value, why, warnings, name):
    old = field["value"]
    field["value"] = round(new_value, 2)
    field["confidence"] = round(0.9 * min(field.get("confidence") or 0.9, 0.9), 3)
    field["source_label"] = f"{field.get('source_label') or name} (corrected, read {old:,.2f})"
    warnings.append(f"{name} was read as {old:,.2f} and corrected to {new_value:,.2f}: {why}. Please check.")


def _reconcile_amounts(fields, items, words_total, warnings, figures=frozenset(), round_off=False):
    """Repair OCR misreads in the amounts using the invoice's own arithmetic.

    A value is changed only when (1) the other amounts on the invoice prove what it must be, and
    (2) the proven value differs from what was read in exactly one digit. A genuine mistake on the
    invoice therefore still shows up as a failed check; it is never silently 'fixed'. Every
    correction is listed in the warnings.

    The total may also be replaced when the amount in words and a figure printed on the invoice
    agree with each other (two independent readings) but not with the total that was read, e.g.
    the total quantity '1 NOS' taken from the Total row of a photo."""
    total = fields.get("total")
    if total and words_total is not None and abs(total["value"] - words_total) > 1:
        if total["value"] < 0 and abs(-total["value"] - words_total) <= 1:
            _correct(total, words_total, "the amount in words says so (a table border was read as a minus "
                     "sign or bracket)", warnings, "Total")
        elif _one_digit_apart(total["value"], words_total):
            _correct(total, words_total, "the amount in words says so", warnings, "Total")
        elif any(abs(f - words_total) <= 0.01 for f in figures):
            _correct(total, words_total, "the amount in words and the figure printed on the invoice both say so",
                     warnings, "Total")
    tot = _amount_of(fields, "total")
    total_confirmed = tot is not None and words_total is not None and abs(tot - words_total) <= 1
    item_sum = (round(sum(i["amount"] for i in items), 2)
                if items and all(i["amount"] is not None for i in items) else None)
    ship, disc = _amount_of(fields, "shipping") or 0, _amount_of(fields, "discount") or 0
    sub = _amount_of(fields, "subtotal")
    base = sub if sub is not None else item_sum  # taxable value
    subtotal_confirmed = sub is not None and item_sum is not None and abs(sub - item_sum) <= 1

    tax = fields.get("tax")
    comps = (tax or {}).get("components") or {}
    if "CGST" in comps and "SGST" in comps and int(comps["CGST"]) == int(comps["SGST"]) != 0:
        # CGST and SGST are always equal: when the paise of one were not readable (handwritten
        # '3805 = 93' read as '3805'), the one with paise gives both
        whole = [k for k in ("CGST", "SGST") if comps[k] == int(comps[k])]
        if len(whole) == 1:
            wrong, right = whole[0], "SGST" if whole[0] == "CGST" else "CGST"
            warnings.append(f"{wrong} was read as {comps[wrong]:,.2f}; the paise were not readable and "
                            f"{comps[right]:,.2f} was taken from {right} (they are always equal). Please check.")
            comps[wrong] = comps[right]
            tax["value"] = round(sum(comps.values()), 2)
            tax["confidence"] = round(0.9 * min(tax.get("confidence") or 0.9, 0.9), 3)
    if (tot is not None and base is not None and "CGST" in comps and "SGST" in comps
            and abs(comps["CGST"] - comps["SGST"]) > 0.05
            and (_one_digit_apart(comps["CGST"], comps["SGST"])
                 or int(comps["CGST"]) == int(comps["SGST"]))):  # handwritten paise not read: 3805 / 3805.93
        # CGST and SGST are always equal; the one that matches total - taxable value is the right one
        expected = round(tot - base - ship + disc, 2)
        right = next((k for k in ("CGST", "SGST") if abs(2 * comps[k] - expected) <= 0.05), None)
        if right:
            wrong = "SGST" if right == "CGST" else "CGST"
            old = comps[wrong]
            comps[wrong] = comps[right]
            tax["value"] = round(sum(comps.values()), 2)
            tax["confidence"] = round(0.9 * min(tax.get("confidence") or 0.9, 0.9), 3)
            warnings.append(f"{wrong} was read as {old:,.2f} and corrected to {comps[right]:,.2f}: CGST and SGST "
                            f"must be equal, and total - taxable value = {expected:,.2f}. Please check.")
    elif tax and tot is not None and base is not None and (total_confirmed or subtotal_confirmed):
        expected = round(tot - base - ship + disc, 2)
        # a grand total in whole rupees within a rupee of subtotal + tax was rounded off, not misread
        rounded = abs(tot - round(tot)) < 0.005 and abs(tax["value"] - expected) < 1
        if (expected > 0 and abs(tax["value"] - expected) > 0.05 and not rounded
                and _one_digit_apart(tax["value"], expected)):
            _correct(tax, expected, "total - subtotal gives this tax", warnings, "Tax")

    tax_value = _amount_of(fields, "tax")
    if (sub is not None and sub == int(sub) and total_confirmed and tax_value is not None
            and tax_value != int(tax_value)):
        # the subtotal's paise were not readable (handwritten in their own box): the total, confirmed by
        # the amount in words, minus the tax gives the same rupees with the paise
        expected = round(tot - tax_value - ship + disc, 2)
        if int(expected) == int(sub) and 0 < expected - sub < 1:
            _correct(fields["subtotal"], expected, "its paise were not readable; total (confirmed by the amount "
                     "in words) - tax gives this", warnings, "Subtotal")
            sub = expected
    if (sub is not None and item_sum is not None and tot is not None and tax_value is not None
            and abs(sub - item_sum) > 0.05):
        expected = round(tot - tax_value - ship + disc, 2)
        if abs(item_sum - expected) <= 0.05 and _one_digit_apart(sub, expected):
            _correct(fields["subtotal"], expected, "the line items and total - tax both give this",
                     warnings, "Subtotal")
    _gst_solve(fields, item_sum, words_total, warnings, round_off)


# GST rates (CGST + SGST together, or IGST), the most common first
GST_RATES = (0.18, 0.05, 0.12, 0.28, 0.40, 0.03, 0.015, 0.0025)


def _subsequence(short, long):
    rest = iter(long)
    return all(ch in rest for ch in short)


def _tax_support(read, want):
    """How well a tax figure read by OCR supports the amount `want`: 3 exactly, 2 the same rupees
    (handwritten paise not readable), 1 one digit misread, lost or added ('3808', '380' for 3805.93)."""
    if read is None:
        return 0
    if abs(read - want) < 0.005:
        return 3
    r, w = str(int(abs(read))), str(int(abs(want)))
    if r == w:
        return 2
    if len(w) >= 3 and (_one_digit_apart(int(abs(read)), int(abs(want)))
                        or (len(r) == len(w) - 1 >= 3 and _subsequence(r, w))
                        or (len(r) == len(w) + 1 and _subsequence(w, r))):
        return 1
    return 0


def _total_garbled(read, want, base):
    """True when a total read from handwriting cannot be right and looks like `want` with digits lost
    or added ('9900', '900', '499100' for 49900), or is below the taxable value."""
    if read < base:
        return True
    r, w = str(int(abs(read))), str(int(abs(want)))
    return len(r) != len(w) and Levenshtein.distance(r, w) <= 2


def _gst_solve(fields, item_sum, words_total, warnings, round_off=False):
    """Handwritten GST bills whose amounts do not add up: CGST = SGST = taxable value x rate / 2 (or
    IGST = taxable value x rate) at a standard GST rate, and taxable value + tax = total. The amounts
    are put back together only when the tax figures that were read support the rate (their rupees, or
    all but one digit, match) and the total read agrees, or cannot be right; every change is listed in
    the warnings. Amounts that already add up are never touched."""
    tax, sub_f, tot_f = fields.get("tax"), fields.get("subtotal"), fields.get("total")
    comps = (tax or {}).get("components") or {}
    if not comps or fields.get("discount") or fields.get("shipping"):
        return
    sub, tot = _amount_of(fields, "subtotal"), _amount_of(fields, "total")
    if sub is not None and tot is not None and abs(sub + sum(comps.values()) - tot) <= 0.01:
        return
    halves = "IGST" not in comps
    totals = [t for t in (tot, words_total) if t is not None]
    best = None
    for b in {v for v in (sub, item_sum) if v}:
        for rank, rate in enumerate(GST_RATES):
            options = {b}
            # with a total in whole rupees (read, or the one this rate gives), the taxable value's paise
            # follow from it: 42288 read, 49900 / 1.18 = 42288.14
            for t in totals + [float(round(b * (1 + rate)))] if b == int(b) else ():  # only paise not read
                if t == round(t) and t > b and int(round(t / (1 + rate), 2)) == int(b):
                    options.add(round(t / (1 + rate), 2))
            for base in options:
                h = round(base * rate / 2, 2) if halves else round(base * rate, 2)
                parts = {"CGST": h, "SGST": h} if halves else {"IGST": h}
                support = sum(_tax_support(comps.get(k), v) for k, v in parts.items())
                if any(comps.get(k) is not None and comps[k] >= v / 5 and not _tax_support(comps[k], v)
                       for k, v in parts.items()):
                    continue  # a clearly different figure was read (27,900 for 22,500): a real mismatch stays visible
                pred = round(base + sum(parts.values()), 2)
                agrees = any(abs(t - p) <= 1 for t in totals for p in (pred, round(pred)))
                exact = any(abs(t - pred) <= 0.01 for t in totals)
                if support < 2 or not (agrees and support + 3 >= 5 or support >= 4 and (
                        tot is None or any(_total_garbled(tot, p, base) for p in (pred, round(pred))))):
                    continue
                key = (support + 3 * agrees + exact, base == b, -rank)
                if best is None or key > best[0]:
                    best = (key, base, rate, parts)
    if best is None:
        return
    _, base, rate, parts = best
    h = next(iter(parts.values()))
    # a figure read with paise within 10 of the computed ones keeps its own (rounded down, or per line:
    # 1251.10 for 9% of 13902.10); CGST and SGST stay equal
    read = [comps[k] for k in parts if comps.get(k) is not None and abs(comps[k] - h) <= 0.1 and comps[k] != int(comps[k])]
    for k in parts:
        comps[k] = min(read, key=lambda v: abs(v - h)) if read else h
    old_tax = tax["value"]
    tax["value"] = round(sum(comps[k] for k in parts), 2)
    tax["confidence"] = round(0.8 * min(tax.get("confidence") or 0.9, 0.9), 3)
    why = (f"{'CGST and SGST' if len(parts) == 2 else 'IGST'} at {rate * 100:g}% of the taxable value "
           f"{base:,.2f}")
    if abs(old_tax - tax["value"]) > 0.01:
        warnings.append(f"Tax was read as {old_tax:,.2f} and set to {tax['value']:,.2f}: {why}. Please check.")
    if sub_f and abs(base - sub_f["value"]) > 0.01:
        _correct(sub_f, base, "the paise follow from the total at this GST rate", warnings, "Subtotal")
    pred = round(base + tax["value"], 2)
    total = float(round(pred)) if round_off else pred
    if tot_f is None:
        fields["total"] = {"value": total, "confidence": round(0.7 * tax["confidence"], 3),
                           "source_label": "Taxable value + tax", "source_text": "", "page": None}
        warnings.append(f"Total could not be read; taxable value + tax gives {total:,.2f} ({why}). Please check.")
    elif not any(abs(tot_f["value"] - p) <= 1 for p in (pred, round(pred))):
        _correct(tot_f, total, f"taxable value + tax ({why})", warnings, "Total")


def _amount_of(fields, key):
    f = fields.get(key)
    return f["value"] if f else None


def _checks(fields, items, words_total=None):
    out = {}
    if words_total is not None and _amount_of(fields, "total") is not None:
        out["total_matches_amount_in_words"] = abs(_amount_of(fields, "total") - words_total) <= 1
    sub, tax = _amount_of(fields, "subtotal"), _amount_of(fields, "tax")
    disc, tot = _amount_of(fields, "discount"), _amount_of(fields, "total")
    ship = _amount_of(fields, "shipping")
    if tot is not None and sub is not None and (tax is not None or disc is not None or ship is not None):
        expected = sub + (tax or 0) + (ship or 0) - (disc or 0)
        out["subtotal_plus_tax_minus_discount_equals_total"] = abs(expected - tot) <= max(0.05, 0.002 * abs(tot))
    if items and sub is not None:
        line_sum = sum(i["amount"] for i in items if i["amount"] is not None)
        out["line_items_sum_equals_subtotal"] = abs(line_sum - sub) <= max(0.05, 0.002 * abs(sub))
    return out


def extract_invoice(lines, matcher):
    """Return the field dictionary, line items, checks and warnings for one invoice."""
    full_text = "\n".join(ln.text for ln in lines)
    day_first = nz.infer_day_first(full_text)
    table_cands, table_lines = _summary_table_candidates(lines, matcher)
    cands, label_lines, label_cells, totals_lines = collect_candidates(lines, matcher, skip_lines=table_lines)
    cands += table_cands
    totals_lines |= table_lines

    warnings = []
    cells = label_cells
    fields = {
        "invoice_number": _scalar(cands, "invoice_number", lines, label_lines, "ident", label_cells=cells),
        "invoice_date": _scalar(cands, "invoice_date", lines, label_lines, "date", day_first=day_first,
                                label_cells=cells),
        "due_date": _scalar(cands, "due_date", lines, label_lines, "date", day_first=day_first, label_cells=cells),
        "po_number": _scalar(cands, "po_number", lines, label_lines, "ident", label_cells=cells),
        "seller": _party(cands, "seller", lines, label_cells, label_lines),
        "buyer": _party(cands, "buyer", lines, label_cells, label_lines),
        "subtotal": _scalar(cands, "subtotal", lines, label_lines, "amount", prefer_last=True, label_cells=cells),
        "tax": _tax(cands, lines, label_lines, cells),
        "discount": _scalar(cands, "discount", lines, label_lines, "amount", label_cells=cells),
        "shipping": _scalar(cands, "shipping", lines, label_lines, "amount", label_cells=cells),
        "total": _scalar(cands, "total", lines, label_lines, "amount", prefer_last=True, label_cells=cells),
    }

    if fields["subtotal"] is None and fields["total"] is not None:
        fields["subtotal"] = _total_above_grand_total(cands, lines, label_lines, cells, fields["total"]["value"])

    words_total, words_line = _amount_in_words(lines)
    sub = (fields["subtotal"] or {}).get("value")
    if words_total and sub and words_total < sub - 1 and not fields["discount"]:
        words_total = None  # misread handwriting ('Forty nine thousand' read as 'ine thousand'): below the subtotal
    if fields["total"] is None and words_total:
        fields["total"] = {"value": words_total, "confidence": round(0.85 * words_line.conf, 3),
                           "source_label": "Amount in words", "source_text": words_line.text,
                           "page": words_line.page + 1}

    if fields["seller"] is None:
        buyer_tax_id = ((fields["buyer"] or {}).get("value") or {}).get("tax_id")
        inferred = None
        for rule in (lambda: _seller_from_signature(lines, label_cells),
                     lambda: _seller_from_tax_id(lines, skip=buyer_tax_id), lambda: _infer_seller(lines)):
            found = rule()
            if found and _looks_like_name((found["value"] or {}).get("name")):
                inferred = found
                break
        if inferred:
            if not inferred["value"].get("tax_id"):
                inferred["value"]["tax_id"] = _header_gstin(lines, inferred["value"].get("name"), skip=buyer_tax_id)
            fields["seller"] = inferred
            warnings.append("Seller was inferred (no 'Seller/Vendor' label): please check it.")

    seller_tax_id = ((fields["seller"] or {}).get("value") or {}).get("tax_id")
    seller_name = ((fields["seller"] or {}).get("value") or {}).get("name")
    # the seller's own GSTIN unreadable ('09AGHPR3344C1Z', a letter short): the GSTINs below the seller's
    # name are still the buyer's
    below = next((k + 1 for k, ln in enumerate(lines[:15]) if seller_name and seller_name in ln.text), None)
    if fields["buyer"] is None and (seller_tax_id or below):
        found = _buyer_from_tax_id(lines, seller_tax_id, start=0 if seller_tax_id else below)
        if found:
            fields["buyer"] = found
            warnings.append("Buyer was inferred (no 'Buyer / Bill to' label): please check it.")

    currency = _scalar(cands, "currency", lines, label_lines, "text")
    if currency and currency["value"]:
        iso = nz.detect_currency(currency["value"]) or currency["value"].strip().upper()[:3]
        fields["currency"] = {**currency, "value": iso}
    else:
        source = fields["total"]["source_text"] if fields["total"] else ""
        iso = nz.detect_document_currency(source, full_text)
        fields["currency"] = ({"value": iso, "confidence": 0.6, "source_label": "detected from text",
                               "source_text": "", "page": None} if iso else None)

    items, has_table = _line_items(lines, totals_lines)
    for item in items:  # 'Carrler' -> 'Carrier' (look-alike letters in handwriting)
        item["description"] = nz.fix_item_words(item.get("description"))
    # every amount printed on the invoice (figures, not words)
    figures = {v for v in (nz.parse_amount(m.group(0)) for m in nz.AMOUNT_RE.finditer(nz.strip_dates(full_text)))
               if v is not None}
    _reconcile_amounts(fields, items, words_total, warnings, figures,
                       round_off=bool(re.search(r"(?i)\bround\s*-?\s*off\b|\br\s*/\s*off\b", full_text)))
    total, tax = _amount_of(fields, "total"), _amount_of(fields, "tax")
    if fields["subtotal"] is None and total is not None and tax is not None:
        derived = round(total - tax - (_amount_of(fields, "shipping") or 0) + (_amount_of(fields, "discount") or 0), 2)
        confidence = round(0.9 * min(fields["total"]["confidence"], fields["tax"]["confidence"]), 3)
        if items and abs(sum(i["amount"] or 0 for i in items) - derived) <= 1.0:  # only rounding may differ
            # not printed (common on GST invoices), but total - tax equals the line items exactly
            fields["subtotal"] = {"value": derived, "confidence": confidence,
                                  "source_label": "(total - tax, matches line items)", "source_text": "", "page": None}
        elif (derived > 0 and words_total is not None and abs(total - words_total) <= 1
              and any(abs(f - derived) <= 0.01 for f in figures)):
            # the line items were not readable (photo), but the total is confirmed by the amount in
            # words and total - tax is printed on the invoice (the taxable value / item amount)
            fields["subtotal"] = {"value": derived, "confidence": round(0.9 * confidence, 3),
                                  "source_label": "(total - tax, printed on the invoice)", "source_text": "",
                                  "page": None}
    checks = {**_checks(fields, items, words_total), **_gst_checks(fields)}
    if not has_table:
        warnings.append("No line-item table detected.")
    warnings += [CHECK_MESSAGES.get(name, name) for name, ok in checks.items() if ok is False]
    for name in REQUIRED_FIELDS:
        if fields.get(name) is None:
            warnings.append(f"Could not find {name.replace('_', ' ')}.")

    confs = [fields[n]["confidence"] for n in REQUIRED_FIELDS if fields.get(n)]
    overall = round(sum(confs) / len(REQUIRED_FIELDS), 3) if confs else 0.0
    return {
        "fields": fields,
        "line_items": items,
        "checks": checks,
        "warnings": warnings,
        "overall_confidence": overall,
        "date_order": "day-first" if day_first else "month-first",
        "raw_text": full_text,
    }
