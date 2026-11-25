"""Common spellings for the party names and addresses that OCR read with doubt.

Handwriting comes back with look-alike letters: 'Lovistics' for Logistics, 'L+d' for Ltd, 'maombai' for
Mumbai, 'Naggr' for Nagar, 'Bulding' for Building. A word the reader was not sure of (confidence below
SURE) that is not a common word itself, but is close to exactly one common word of Indian business names
and addresses, is set to that word. Close means: letters that look alike in handwriting swapped (a/o/u,
g/a, i/l/t, '+' for t ...), at most one letter missed or added, and the first letter the same or a
look-alike.

* Names get only the business words (Pvt, Ltd, Traders, Logistics ...). Place words (Mumbai, Nagar,
  Andheri ...) are used in addresses only: a person's name can be close to a place.
* Words of four characters or fewer are changed only when they hold something that is not a letter
  ('L+d', 'Pv+', 'N0'): many short names are real words.
* A few fixed patterns: 'Plot Ho.12' -> 'Plot No. 12', 'Put Ltd' -> 'Pvt Ltd', '+d' -> 'Ltd',
  'Andheri(e)' -> 'Andheri (E)'; a place written in small letters ('mumbai') gets its capital, and
  'cHS' is written 'CHS'.
Text read from a PDF's own text layer is never changed. Every changed word is listed in the warnings.
"""
import re
from functools import lru_cache

from rapidfuzz.distance import Levenshtein

LETTER_SURE = 0.9  # a word whose every letter was read with at least this confidence is kept as read
SURE = 0.95  # ... or the word itself, where letter confidences are not known (other readers)

_BUSINESS = """Pvt Private Ltd Limited Company Corporation Enterprises Enterprise Traders Trading Industries
Industry Logistics Solutions Services Technologies Technology Infotech Softech Software Systems Agencies Agency
Associates Brothers International Exports Imports Overseas Engineering Engineers Electricals Electrical
Electronics Hardware Sanitary Sanitaryware Plumbing Construction Constructions Builders Developers
Infrastructure Projects Marketing Distributors Suppliers Supplies Manufacturing Manufacturers Pharmaceuticals
Pharma Chemicals Plastics Polymers Packaging Packers Textiles Fabrics Garments Steels Metals Motors Automobiles
Transport Transporters Travels Foods Products Stores Emporium Furniture Furnishers Interiors Decorators Paints
Marbles Granites Timbers Plywood Medical Medicos Hospital Pharmacy Chemists Opticals Jewellers Communications
Computers Networks Digital Printers Stationers Stationery Consultants Consultancy Management Ventures Holdings
Global Impex Udyog Bhandar Fabricators Workshop Machinery Equipments Equipment Instruments Controls Automation
Batteries Cables Lighting Security Facilities Caterers Restaurant Sweets Bakers Mills Fashions Collections
Creations Telecom Hydraulics Pneumatics Fasteners Bearings Spares Tyres Aluminium Courier Couriers Cargo Carriers
Movers Freight Shipping Express Warehousing Sales Merchants Wholesale Retail Agro Dairy Studio Graphics
Advertising Events Academy Institute Clinic Diagnostics Laboratories Healthcare Polishing Cleaning Repairing
Works Solar Infra Estates Realty Properties Housing Finance Insurance Securities Capital Investments"""
_ADDRESS = """Plot House Flat Shop Floor Ground Building Bldg Tower Towers Complex Plaza Arcade Mall Market Bazar
Bazaar Chowk Road Marg Street Lane Gali Galli Nagar Vihar Colony Enclave Sector Phase Block Pocket Society
Apartment Apartments Residency Heights Park Garden Gardens Estate Industrial Area Village Tehsil District Near
Opposite Behind Beside Main Cross Highway Station Railway Stand Gate East West North South Extension Layout
Godown Warehouse Office Premises Centre Center Bhawan Bhavan Niwas Sadan Mandir Circle Junction Bypass Chawl
Wadi Pada Peth Mohalla Puram Ganj Kunj Kendra Khasra Survey Basement Mezzanine Wing Unit Gala Compound Kothi
Sadak Avenue Square Cottage Villa Bungalow Hall Thana"""
_PLACES = """Mumbai Delhi Gurugram Gurgaon Noida Faridabad Ghaziabad Pune Thane Navi Nashik Nagpur Aurangabad
Bengaluru Bangalore Chennai Kolkata Hyderabad Secunderabad Ahmedabad Surat Vadodara Rajkot Jaipur Jodhpur
Udaipur Lucknow Kanpur Varanasi Prayagraj Allahabad Meerut Bareilly Aligarh Moradabad Dehradun Haridwar
Chandigarh Mohali Panchkula Ludhiana Amritsar Jalandhar Patiala Ambala Karnal Panipat Sonipat Sonepat Rohtak
Hisar Rewari Bhiwadi Manesar Palwal Bahadurgarh Indore Bhopal Gwalior Jabalpur Raipur Bhilai Ranchi Jamshedpur
Patna Bhubaneswar Cuttack Guwahati Kochi Ernakulam Thiruvananthapuram Kozhikode Coimbatore Madurai Tiruppur
Mysuru Mysore Mangaluru Mangalore Hubli Belgaum Visakhapatnam Vijayawada Guntur Warangal Panaji Vapi Silvassa
Bhiwandi Dombivli Vasai Virar Panvel Andheri Borivali Malad Goregaon Kandivali Bandra Dadar Kurla Ghatkopar
Mulund Vikhroli Powai Chembur Worli Byculla Colaba Santacruz Jogeshwari Dahisar Bhayandar Okhla Saket Dwarka
Rohini Pitampura Janakpuri Shahdara Naraina Mayapuri Maharashtra Haryana Punjab Rajasthan Gujarat Karnataka
Kerala Telangana Jharkhand Odisha Bengal Assam Uttarakhand Himachal Chhattisgarh Pradesh Uttar Madhya Andhra
Tamil India"""
_ABBREVIATIONS = {"LLP", "HUF", "CHS", "MIDC", "GIDC", "HSIIDC", "HSIDC", "RIICO", "UPSIDC", "IMT"}


def _lexicon(*texts):
    return {w.lower(): w for text in texts for w in text.split()}


_NAME_WORDS = _lexicon(_BUSINESS, "LLP HUF")
_PLACE_WORDS = _lexicon(_PLACES)
_ADDRESS_WORDS = {**_lexicon(_BUSINESS, _ADDRESS, " ".join(_ABBREVIATIONS)), **_PLACE_WORDS}

# letters (and figures / signs) that handwriting and OCR mix up, as pairs
_LOOK_ALIKE = {frozenset(p) for p in """ao au ou ae ce co eo uv vw nu nm nh nr hb hk il ij it lt tf ag gq ad gy
o0 l1 i1 t+ f+ s5 b8 b6 g9 q9 z2 i| l|""".split()}
_LIGHT = set("aeiouilrt+|")  # missed or added most easily: vowels and thin letters
_LIMITS = ((4, 0.5), (6, 0.8), (9, 1.2))  # (up to this many characters, the largest distance)
_LIMIT_LONG = 1.6
_MARGIN = 0.3  # the closest common word must be this much closer than the next one

_TOKEN = re.compile(r"[A-Za-z0-9+|]+")
_LTD = re.compile(r"(?:[Ll1I|]?\+[dD]|[1|][tT+][dD]|[Ll][fF][dD])")  # 'L+d', '+d', '1td', 'Lfd'
_PVT_BEFORE_LTD = re.compile(r"(?i)\b(p[uvy][tf+])(?=\.?\s*(?:ltd|limited)\b)")  # 'Put Ltd', 'Pv+ Ltd'
# 'Plot Ho.12', 'Flat N0 4': the 'No.' of a plot, house or shop number
_NUMBER_LABEL = re.compile(r"(?i)\b(plot|house|h|shop|flat|room|office|unit|gala|khasra|kh|survey|gate|door|block|"
                           r"wing|pocket|gali|bldg)(\s*[.\-]?\s*)([hnm][o0a])(\s*[.\-:]?\s*)(?=\d)")
_SIDE_OF_TOWN = re.compile(r"(?<=[A-Za-z])\s*\(\s*([ew])\s*\)")  # 'Andheri(e)' -> 'Andheri (E)'


def _sub(a, b):
    return 0.0 if a == b else 0.4 if frozenset((a, b)) in _LOOK_ALIKE else 1.0


def _indel(ch):
    return 0.6 if ch in _LIGHT else 1.0


@lru_cache(maxsize=65536)
def _distance(a, b):
    """Edit distance of two lower-case words weighted for handwriting: look-alike letters 0.4, other
    letters 1, a vowel or thin letter missed or added 0.6, another letter 1."""
    prev = [0.0]
    for ch in b:
        prev.append(prev[-1] + _indel(ch))
    for ca in a:
        cur = [prev[0] + _indel(ca)]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + _indel(ca), cur[j - 1] + _indel(cb), prev[j - 1] + _sub(ca, cb)))
        prev = cur
    return prev[-1]


def _limit(n):
    return next((limit for most, limit in _LIMITS if n <= most), _LIMIT_LONG)


def closest_word(token, words):
    """The common word a misread token stands for, or None (see the module's rules)."""
    low = token.lower()
    if low in words or sum(ch.isdigit() for ch in low) > 1 or sum(ch.isalpha() for ch in low) < 2:
        return None
    if len(low) <= 4 and low.isalpha():
        return None  # a short word of letters only: often a real word or name
    limit = _limit(len(low))
    found = []
    for key, word in words.items():
        if abs(len(key) - len(low)) > 1 or (key[0] != low[0] and frozenset((key[0], low[0])) not in _LOOK_ALIKE):
            continue
        if Levenshtein.distance(low, key, score_cutoff=4) > 4:
            continue
        cost = _distance(low, key)
        if cost <= limit:
            found.append((cost, word))
    if not found:
        return None
    found.sort()
    if len(found) > 1 and found[1][0] - found[0][0] < _MARGIN:
        return None  # two common words fit about as well: leave it as read
    return found[0][1]


def _cased(word, like):
    letters = [ch for ch in like if ch.isalpha()]
    return word.upper() if len(letters) >= 2 and all(ch.isupper() for ch in letters) else word


def _join_cut_words(text, words, note):
    """A word cut in two at a wide gap of the handwriting ('m umbai'): the two parts, neither a common
    word itself, joined when together they are one."""
    spans = [m.span() for m in _TOKEN.finditer(text)]
    out, last, i = [], 0, 0
    while i < len(spans) - 1:
        (s1, e1), (s2, e2) = spans[i], spans[i + 1]
        a, b = text[s1:e1], text[s2:e2]
        word = words.get((a + b).lower()) if text[e1:s2] == " " and a.isalpha() and b.isalpha() else None
        if word and len(a + b) >= 5 and a.lower() not in words and b.lower() not in words:
            out += [text[last:s1], note(f"{a} {b}", _cased(word, a + b))]
            last, i = e2, i + 2
        else:
            i += 1
    return "".join(out) + text[last:]


def fix_text(text, words, unsure):
    """A party name or address with common spellings: (new text, [(word as read, word now)]).
    words: {lower case: common word}; unsure(token): True when the reader doubted that word."""
    changes = []

    def note(old, new):
        if old.lower() != new.lower():
            changes.append((old, new))
        return new

    def number_label(m):
        if m.group(3).lower() == "no":
            return m.group(0)
        note(m.group(3), "No")
        return f"{m.group(1)} No. "

    text = _NUMBER_LABEL.sub(number_label, text)
    text = _SIDE_OF_TOWN.sub(lambda m: f" ({m.group(1).upper()})", text)

    text = _join_cut_words(text, words, note)

    def token(m):
        tok = m.group(0)
        if _LTD.fullmatch(tok):
            return note(tok, "Ltd")
        if tok.upper() in _ABBREVIATIONS and len(tok) >= 3:
            return tok.upper()
        if tok.islower() and tok in _PLACE_WORDS and words is not _NAME_WORDS:
            return _PLACE_WORDS[tok]  # 'mumbai' -> 'Mumbai'
        if not unsure(tok):
            return tok
        word = closest_word(tok, words)
        return note(tok, _cased(word, tok)) if word else tok

    text = _TOKEN.sub(token, text)
    text = _PVT_BEFORE_LTD.sub(lambda m: m.group(1) if m.group(1).lower() == "pvt" else note(m.group(1), "Pvt"),
                               text)
    return text, changes


def _doubted(pages):
    """unsure(token): did the reader doubt this word anywhere on the read (OCR) pages? A word's own
    confidence is the mean of its letters', so a sure-looking 'Lovistics' (0.97) hides a doubtful 'v'
    (0.77): the fast reader's letter confidences are used where its lines are kept (the lowest letter of
    the word below LETTER_SURE), else the word's confidence (below SURE)."""
    letters, words = {}, {}
    for page in pages:
        if page.method != "ocr":
            continue
        for text, _, info in getattr(getattr(page, "ocr_lines", None), "read", None) or ():
            sure = getattr(info, "confs", None)
            if sure and len(sure) == len(text):
                for m in _TOKEN.finditer(text):
                    key = m.group(0).lower()
                    letters[key] = min(letters.get(key, 1.0), min(sure[m.start():m.end()]))
        for w in page.words:
            for tok in _TOKEN.findall(w.text):
                words[tok.lower()] = min(words.get(tok.lower(), 1.0), w.conf)
    if not words:
        return None

    def unsure(tok):
        key = tok.lower()
        return letters[key] < LETTER_SURE if key in letters else words.get(key, 0.0) < SURE

    return unsure


def fix_parties(result, pages):
    """Common spellings in the seller's and buyer's names and addresses read from a photo or scan
    (see the module's notes). The changed words are listed in one warning."""
    fields = result.get("fields") or {}
    unsure = _doubted(pages)
    if unsure is None:
        return
    notes = []
    for key in ("seller", "buyer"):
        field = fields.get(key) or {}
        value = field.get("value")
        page = field.get("page")
        if not value or not page or page > len(pages) or pages[page - 1].method != "ocr":
            continue
        for part, words in (("name", _NAME_WORDS), ("address", _ADDRESS_WORDS)):
            text = value.get(part)
            if not text:
                continue
            new, changes = fix_text(text, words, unsure)
            value[part] = new
            if changes:
                notes.append(f"{key.capitalize()} {part}: " + ", ".join(f"{a} → {b}" for a, b in changes))
    if notes:
        result.setdefault("warnings", []).append(
            "Words read unclearly were set to their common spelling (" + "; ".join(notes) + "): please check them.")
