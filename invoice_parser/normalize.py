"""Parse raw text into clean values: money amounts, dates, tax IDs, emails, phone numbers."""
import re

from dateutil import parser as date_parser

from . import config

CURRENCY_SYMBOLS = {"₹": "INR", "€": "EUR", "£": "GBP", "$": "USD"}
# not inside a longer word, but OCR junk such as "_" or "|" may touch it ("_INR Two Lakh ...")
ISO_CODE_RE = re.compile(r"(?<![A-Za-z0-9])(INR|USD|EUR|GBP|AUD|CAD|SGD|AED|JPY|CNY|CHF)(?![A-Za-z0-9])")
_CURRENCY_WORDS_RE = re.compile(r"(?i)\b(INR|USD|EUR|GBP|AUD|CAD|SGD|AED|JPY|CNY|CHF|Rs|Rupees?)\b\.?")

# Money-looking numbers: 1,234.50 | 1.234,50 | 12,34,567.00 | 1234 | (45.00) | 12.5
# Percentages such as "18%" are deliberately excluded.
AMOUNT_RE = re.compile(
    r"(?<![\w/-])\(?-?\d{1,3}(?:[,.']\d{2,3})+(?:[.,]\d{1,2})?\)?(?!\s*%)(?![\w/-])"
    r"|(?<![\w/-])\(?-?\d+(?:[.,]\d{1,2})?\)?(?!\s*%)(?![\w/-])"
)

DATE_PATTERNS = [
    re.compile(r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b"),                         # 2024-03-05
    re.compile(r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b"),                       # 05/03/2024, 5-3-24
    re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?[ -][A-Za-z]{3,9}\.?,?[ -]\d{2,4}\b"),  # 05 Mar 2024
    re.compile(r"\b[A-Za-z]{3,9}\.?[ -]\d{1,2}(?:st|nd|rd|th)?,?[ -]\d{2,4}\b"),  # Mar 05, 2024
]

GSTIN_RE = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]\b")
LABELLED_TAX_RE = re.compile(
    r"(?i)\b(?:GSTIN|GST\s*(?:No|Number|ID)|VAT\s*(?:No|Reg(?:istration)?|ID|Number)?|"
    r"Tax\s*(?:[Il1]d|No|Number)|TIN|EIN|PAN|ABN|UEN|Reg(?:istration)?\s*No)\.?\s*[:#-]?\s*"  # 'Tax ld': OCR's l for I
    r"([A-Z0-9][A-Z0-9/-]{4,24})"
)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
IBAN_RE = re.compile(r"(?i)\b(?:IBAN|SWIFT|BIC)\b\s*[:#]?\s*[A-Z0-9 ]{6,34}")
PHONE_RE = re.compile(r"\+?\d[\d\s()-]{7,}\d")


def parse_amount(text):
    """'1,234.50' -> 1234.5, '1.234,50 €' -> 1234.5, '12,34,567.00' -> 1234567.0, '(45)' -> -45.0"""
    if text is None:
        return None
    s = str(text).strip()
    negative = s.startswith("-") or (s.startswith("(") and s.endswith(")")) or bool(re.search(r"(?i)\bcr\.?$", s))
    s = _CURRENCY_WORDS_RE.sub(" ", s)
    s = re.sub(r"[₹$€£()%\s-]", "", s)
    m = re.search(r"\d[\d,.']*", s)
    if not m:
        return None
    num = m.group(0).replace("'", "").rstrip(",.")
    if not num:
        return None
    if "," in num and "." in num:
        dec = "." if num.rfind(".") > num.rfind(",") else ","
    elif "," in num:
        dec = "," if num.count(",") == 1 and len(num) - num.rfind(",") - 1 in (1, 2) else None
    elif "." in num:
        dec = "." if num.count(".") == 1 else None
    else:
        dec = None
    if dec:
        whole, frac = num.rsplit(dec, 1)
        whole = re.sub(r"[.,]", "", whole) or "0"
        value = float(f"{whole}.{frac}")
    else:
        value = float(re.sub(r"[.,]", "", num))
    return -value if negative else value


_NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})\b")
_US_STATE_ZIP_RE = re.compile(
    r"\b(?:AL|AK|AZ|AR|CA|CO|CT|DE|DC|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|"
    r"NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|AA|AE|AP) \d{5}(?:-\d{4})?\b")
DAY_FIRST_CURRENCIES = {"INR", "EUR", "GBP", "AUD", "SGD", "AED", "CHF"}


def infer_day_first(text):
    """Decide dd/mm vs mm/dd for one document.

    1. Evidence in the document itself: a date like 18/09/2015 (first part > 12) or 09/18/2015.
    2. Country hints: GSTIN / INR / Rs -> day first. EUR, GBP ... -> day first.
       USD or a US "STATE 12345" address -> month first.
    3. Otherwise the DAY_FIRST setting.
    """
    day = month = 0
    for m in _NUMERIC_DATE_RE.finditer(text or ""):
        a, b = int(m.group(1)), int(m.group(2))
        if a > 12 >= b:
            day += 1
        elif b > 12 >= a:
            month += 1
    if day != month:
        return day > month
    if GSTIN_RE.search(text) or re.search(r"₹|\bINR\b|\bRs\.?\s?\d", text):
        return True
    currency = detect_currency(text)
    if currency in DAY_FIRST_CURRENCIES:
        return True
    if currency == "USD" or _US_STATE_ZIP_RE.search(text):
        return False
    return config.DAY_FIRST


def parse_date(raw, dayfirst=None):
    """Return an ISO date (YYYY-MM-DD) or None. dayfirst=None uses the DAY_FIRST setting."""
    s = raw.strip()
    dayfirst = (config.DAY_FIRST if dayfirst is None else dayfirst) and not re.match(r"^\d{4}[-/.]", s)
    try:
        parsed = date_parser.parse(s, dayfirst=dayfirst)
    except (ValueError, OverflowError, TypeError):
        return None
    if not 1990 <= parsed.year <= 2100:
        return None
    return parsed.date().isoformat()


def find_date(text, dayfirst=None):
    """Find the first date inside a piece of text and return it as ISO, or None."""
    if not text:
        return None
    for pattern in DATE_PATTERNS:
        for m in pattern.finditer(text):
            iso = parse_date(m.group(0), dayfirst)
            if iso:
                return iso
    return None


def strip_dates(text):
    for pattern in DATE_PATTERNS:
        text = pattern.sub(" ", text)
    return text


_GSTIN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_GSTIN_CANDIDATE_RE = re.compile(r"(?<![A-Z0-9])[0-9A-Z]{15}(?![A-Z0-9])")
# GSTIN layout: 2 digits (state), 5 letters, 4 digits, 1 letter (PAN), 1 alphanumeric, 'Z', checksum
_GSTIN_KINDS = "DDLLLLLDDDDLAZA"
_TO_DIGIT = str.maketrans({"O": "0", "D": "0", "Q": "0", "U": "0", "I": "1", "L": "1", "Z": "2",
                           "S": "5", "B": "8", "G": "6", "T": "7"})
_TO_LETTER = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "7": "T"})
_LOOK_ALIKES = {"8": "B", "B": "8", "0": "OD", "O": "0D", "D": "0O", "1": "I", "I": "1", "5": "S", "S": "5",
                "2": "Z", "Z": "2", "6": "G", "G": "6", "7": "T", "T": "7"}


def gstin_check_char(first14: str) -> str:
    """The mod-36 check character of an Indian GSTIN, computed from its first 14 characters."""
    total = 0
    for i, ch in enumerate(first14):
        value = _GSTIN_CHARS.index(ch) * (2 if i % 2 else 1)
        total += value // 36 + value % 36
    return _GSTIN_CHARS[(36 - total % 36) % 36]


def gstin_checksum_ok(gstin: str) -> bool:
    """Validate the 15th character of an Indian GSTIN (mod-36 check digit)."""
    if len(gstin) != 15 or any(ch not in _GSTIN_CHARS for ch in gstin):
        return False
    return gstin_check_char(gstin[:14]) == gstin[14]


def find_gstin(text):
    """A GSTIN in text. OCR often swaps 0/O, 1/I, 5/S, 8/B: such swaps are repaired position by
    position, and a repaired GSTIN is only accepted when its checksum is valid."""
    if not text:
        return None
    upper = text.upper()
    m = GSTIN_RE.search(upper)
    if m:
        return m.group(0)
    # 15-character pieces, or 2-3 neighbouring pieces that OCR split apart ("06BMX PS9096Q3ZB")
    pieces = re.findall(r"[0-9A-Z]+", upper)
    tokens = [p for p in pieces if len(p) == 15]
    for n in (2, 3):
        tokens += ["".join(pieces[i:i + n]) for i in range(len(pieces) - n + 1)
                   if len("".join(pieces[i:i + n])) == 15]
    for token in tokens:
        fixed = "".join(ch.translate(_TO_DIGIT) if kind == "D" else ch.translate(_TO_LETTER) if kind == "L"
                        else "Z" if kind == "Z" else ch for ch, kind in zip(token, _GSTIN_KINDS))
        # a round letter written by hand reads as 0: it may be O, Q or D (the checksum decides)
        rounds = [i for i, (ch, kind) in enumerate(zip(token, _GSTIN_KINDS)) if kind == "L" and ch == "0"][:3]
        variants = [fixed]
        for i in rounds:
            variants = [v[:i] + letter + v[i + 1:] for v in variants for letter in "OQD"]
        for variant in variants:
            # the check character may itself be misread (8/B, 0/O ...): try its look-alikes
            for last in dict.fromkeys(variant[14] + _LOOK_ALIKES.get(variant[14], "")):
                candidate = variant[:14] + last
                if GSTIN_RE.fullmatch(candidate) and gstin_checksum_ok(candidate):
                    return candidate
    return None


_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALE_WORDS = {"thousand": 1_000, "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000,
                "crore": 10_000_000, "crores": 10_000_000, "million": 1_000_000, "billion": 1_000_000_000}
_FILLER_WORDS = {"and", "only", "rupees", "rupee", "inr", "rs", "usd", "us", "dollars", "dollar", "eur",
                 "euros", "euro", "gbp", "pounds", "pound", "amount", "total"}


def _words_value(tokens):
    total = current = 0
    seen = False
    for word in tokens:
        if word in _NUMBER_WORDS:
            current += _NUMBER_WORDS[word]
            seen = True
        elif word == "hundred":
            current = max(current, 1) * 100
            seen = True
        elif word in _SCALE_WORDS:
            total += max(current, 1) * _SCALE_WORDS[word]
            current = 0
            seen = True
        elif word in _FILLER_WORDS:
            continue
        elif seen:
            break  # the number is over
    return (total + current) if seen else None


def words_to_amount(text):
    """'INR Two Lakh Ninety Five Thousand Only' -> 295000.0 (Indian lakh / crore and Western scales);
    '... and Fifty Paise Only' adds 0.50."""
    if not text:
        return None
    t = re.sub(r"[^a-z ]", " ", text.lower().replace("-", " "))
    fraction = 0.0
    m = re.search(r"\band\s+([a-z ]+?)\s+(paise|paisa|cents?)\b", t)
    if m:
        cents = _words_value(m.group(1).split())
        fraction = (cents or 0) / 100
        t = t[: m.start()]
    whole = _words_value(t.split())
    if whole is None:
        return None
    return round(whole + fraction, 2)


def find_tax_id(text):
    if not text:
        return None
    gstin = find_gstin(text)
    if gstin:
        return gstin
    m = LABELLED_TAX_RE.search(text)
    return m.group(1).upper() if m else None


def find_email(text):
    m = EMAIL_RE.search(text or "")
    return m.group(0) if m else None


def find_phone(text):
    for m in PHONE_RE.finditer(text or ""):
        digits = re.sub(r"\D", "", m.group(0))
        if 10 <= len(digits) <= 13:
            return m.group(0).strip()
    return None


def strip_contact(text):
    """Remove e-mail, phone, tax-ID and contact labels from an address line."""
    s = EMAIL_RE.sub(" ", text)
    s = IBAN_RE.sub(" ", s)
    s = LABELLED_TAX_RE.sub(" ", s)
    s = GSTIN_RE.sub(" ", s)
    s = PHONE_RE.sub(lambda m: " " if 10 <= len(re.sub(r"\D", "", m.group(0))) <= 13 else m.group(0), s)
    s = re.sub(r"(?i)\b(email|e-mail|phone|ph|tel|mobile|mob|contact)\b\s*[:.]?", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip(" ,;|:-")


def detect_document_currency(total_text, full_text):
    """Currency of a whole invoice. Explicit codes win over symbols, and Indian markers (GSTIN, ₹, Rs)
    win over a lone symbol, because OCR noise easily produces a stray '£' or '$'."""
    m = ISO_CODE_RE.search(total_text or "")
    if m:
        return m.group(1)
    m = re.search(r"([₹$€£])\s?\d", total_text or "")  # a symbol written right before the total amount
    if m:
        return CURRENCY_SYMBOLS[m.group(1)]
    m = ISO_CODE_RE.search(full_text or "")
    if m:
        return m.group(1)
    if GSTIN_RE.search(full_text or "") or re.search(r"₹|\bRs\.?\s?\d", full_text or ""):
        return "INR"
    return detect_currency(total_text) or detect_currency(full_text)


def detect_currency(text):
    """ISO code from text such as 'INR 1,200' or '₹1,200'. Returns None when nothing is found."""
    if not text:
        return None
    m = ISO_CODE_RE.search(text)
    if m:
        return m.group(1)
    for symbol, iso in CURRENCY_SYMBOLS.items():
        if symbol in text:
            return iso
    return None
