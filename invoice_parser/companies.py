"""Companies (one folder each) and the side of every invoice: purchase or sale.

Every invoice can belong to one company: the business whose books it is part of. It is picked when
uploading; an invoice uploaded without one goes to the known company that it names (as seller or
buyer, by GSTIN or name), if exactly one does. "Your companies" (the businesses that use the app,
typed once on the Companies page with their GSTINs if wanted) are known companies even before any
invoice is in their folders.

For its company an invoice is
  * a purchase (the company is the buyer: money it pays), or
  * a sale     (the company is the seller: money it receives).
The company's own GSTIN is the one typed for it, else learned from its invoices (the party whose name
matches the company, else the GSTIN on most of them), so the side is found even when the company name
is short, like "JP". A reviewer can set the side by hand (entry_type). Invoices without a company count
as purchases, like the bills a business receives.
(Inside the code and the stored figures purchases are still called 'debit' and sales 'credit'.)

A bill can also be in a project (set by a person: a project is not printed on the bill). One project can
run in several of your companies. Its margin is its sales minus its purchases, both without GST (GST is
passed on or claimed back, it is not the company's money). In the figures of all companies together, a
bill from one of your companies to another is left out: that money stays inside your group (in each
company's own books it is still the seller's sale and the buyer's purchase).
"""
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from functools import lru_cache

PURCHASE, SALE = "purchase", "sale"
ENTRY_TYPES = (PURCHASE, SALE)
NO_COMPANY = "-"  # filter value: the invoices without a company
NO_PROJECT = "-"  # filter value: the invoices in no project
# the fields of the counted invoices that the dashboard and the folders need
MONEY_FIELDS = ["id", "company", "entry_type", "currency", "invoice_date", "total", "tax", "seller_name",
                "seller_tax_id", "buyer_name", "buyer_tax_id", "project"]
_LEGAL_WORDS = {"M", "S", "MS", "MESSRS", "THE", "PVT", "PRIVATE", "LTD", "LIMITED", "LLP", "CO", "COMPANY",
                "AND", "INC", "CORP", "CORPORATION", "LLC", "PLC"}


def clean_company(name, known=()):
    """A company name as typed, tidied; the existing spelling when it is a known company in other case."""
    name = re.sub(r"\s+", " ", (name or "").strip())[:80]
    if not name or name == NO_COMPANY:
        return None
    for other in known:
        if other.lower() == name.lower():
            return other
    return name


def clean_project(name, known=()):
    """A project name as typed, tidied; the existing spelling when it differs only in case."""
    return clean_company(name, known)


OWN_COMPANIES_MAX = 20
_GSTIN_RE = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][0-9A-Z]Z[0-9A-Z]\b")


def parse_own_companies(text):
    """'Your companies' as typed, one per line: a name, with the GSTIN anywhere on the line if wanted
    ('Kavya Infotech Pvt Ltd, GSTIN 06BKXPT4822Q3ZY') -> [{"name", "gstin"}]. A line without a name is
    left out."""
    firms, seen = [], set()
    for line in (text or "").splitlines():
        found = _GSTIN_RE.search(line.upper())
        gstin = found.group(0) if found else None
        if found:  # the GSTIN and its label are not part of the name
            line = line[:found.start()] + line[found.end():]
            line = re.sub(r"(?i)\bGST(?:IN)?(?:\s*/\s*UIN)?\b\s*(?:no\b\.?|number\b)?", " ", line)
        name = clean_company(line.strip(" \t,;:|-"))
        if name and name.lower() not in seen:
            seen.add(name.lower())
            firms.append({"name": name, "gstin": gstin})
    return firms[:OWN_COMPANIES_MAX]


def own_companies_text(firms):
    """The text of the 'Your companies' box: one company per line, its GSTIN after a comma."""
    return "\n".join(f["name"] + (f", {f['gstin']}" if f.get("gstin") else "") for f in firms or [])


def known_companies(folder_names, firms):
    """Every known company: the folders that hold invoices and your own companies (one spelling each)."""
    names = {n.lower(): n for n in folder_names if n}
    for f in firms or []:
        names.setdefault(f["name"].lower(), f["name"])
    return sorted(names.values(), key=str.lower)


@lru_cache(maxsize=8192)
def _words(name):
    return tuple(w for w in re.sub(r"[^A-Z0-9]+", " ", (name or "").upper()).split() if w not in _LEGAL_WORDS)


@lru_cache(maxsize=65536)
def _gstin(value):
    value = re.sub(r"[^A-Z0-9]", "", (value or "").upper())
    return value if len(value) >= 8 else None


@lru_cache(maxsize=65536)
def names_match(company, party):
    """Same business? 'JP' ~ 'J.P. Enterprises', 'Kavya Infotech' ~ 'M/s Kavya Infotech Pvt Ltd',
    and small misreadings ('Ganesba Soflech' ~ 'Ganesha Softech'). Remembered: the same few party
    names come back on thousands of invoices (the dashboard compares every one)."""
    a, b = _words(company), _words(party)
    if not a or not b:
        return False
    ja, jb = "".join(a), "".join(b)
    if ja == jb:
        return True
    for short, words in ((ja, b), (jb, a)):  # the short name is the first word(s) of the long one
        joined = ""
        for w in words:
            joined += w
            if joined == short:
                return True
            if len(joined) >= len(short):
                break
    if min(len(ja), len(jb)) >= 4 and (set(a) <= set(b) or set(b) <= set(a)):
        return True
    return min(len(ja), len(jb)) >= 6 and SequenceMatcher(None, ja, jb).ratio() >= 0.85


def _identity(company, rows):
    """The company's own GSTIN, from its invoices."""
    named, seen = Counter(), Counter()
    for r in rows:
        for side, kind in (("seller", SALE), ("buyer", PURCHASE)):
            g = _gstin(r.get(f"{side}_tax_id"))
            if not g:
                continue
            seen[g] += 1
            if r.get("entry_type") == kind:  # set by hand: this side is the company
                named[g] += 2
            elif names_match(company, r.get(f"{side}_name")):
                named[g] += 1
    if named:
        return named.most_common(1)[0][0]
    top = seen.most_common(2)
    if top and top[0][1] >= 2 and (len(top) == 1 or top[0][1] > top[1][1]):
        return top[0][0]
    return None


def company_gstins(rows, own=()):
    """{company: its GSTIN or None} for the companies of these invoices and your own companies (own:
    [{"name", "gstin"}]); a GSTIN typed for one of your companies wins over the one learned."""
    groups = defaultdict(list)
    for r in rows:
        if r.get("company"):
            groups[r["company"]].append(r)
    gstins = {c: _identity(c, rs) for c, rs in groups.items()}
    for f in own or ():
        gstin = _gstin(f.get("gstin"))
        if gstin or f["name"] not in gstins:
            gstins[f["name"]] = gstin or gstins.get(f["name"])
    return gstins


def side_of(row, gstin=None):
    """'purchase' (debit) or 'sale' (credit) for the invoice's company."""
    if row.get("entry_type") in ENTRY_TYPES:
        return row["entry_type"]
    company = row.get("company")
    if company:
        if gstin:
            if _gstin(row.get("seller_tax_id")) == gstin:
                return SALE
            if _gstin(row.get("buyer_tax_id")) == gstin:
                return PURCHASE
        if names_match(company, row.get("seller_name")) and not names_match(company, row.get("buyer_name")):
            return SALE
    return PURCHASE


def names_company(row, company, gstin=None):
    """Is the company the seller or the buyer on the invoice? By its GSTIN when known, else by name."""
    if gstin:
        return gstin in {_gstin(row.get("seller_tax_id")), _gstin(row.get("buyer_tax_id"))}
    return names_match(company, row.get("seller_name")) or names_match(company, row.get("buyer_name"))


def _own_party(row, side, own, gstins):
    """The one of your companies that is the seller / buyer of the bill: by GSTIN when both are known,
    else by name."""
    gstin = _gstin(row.get(f"{side}_tax_id"))
    for f in own:
        known = gstins.get(f["name"])
        if (known == gstin) if known and gstin else names_match(f["name"], row.get(f"{side}_name")):
            return f["name"]
    return None


def between_your_companies(row, own, gstins):
    """A bill from one of your companies to another one: money that stays inside your group."""
    if len(own or ()) < 2:
        return False
    seller = _own_party(row, "seller", own, gstins)
    buyer = seller and _own_party(row, "buyer", own, gstins)
    return bool(buyer) and buyer != seller


def side_seen_by(row, company, gstins):
    """The invoice's side in the books of `company` (None: in its own folder's books). A bill from one of
    your companies to another is the seller's sale and, seen by the buyer, its purchase."""
    if not company or row.get("company") == company:
        return side_of(row, gstins.get(row.get("company")))
    return side_of({**row, "company": company, "entry_type": None}, gstins.get(company))


def with_sides(rows, money_rows, own=()):
    """Rows (exports, data sheet) with 'entry_type' set to the side the invoice counts on."""
    gstins = company_gstins(money_rows, own)
    for r in rows:
        r["entry_type"] = side_of(r, gstins.get(r.get("company")))
    return rows


def guess_company(row, companies, gstins):
    """The one known company this invoice names (seller or buyer), else None. Two of them on one invoice
    (one of your companies selling to the other): the seller's, where it is a sale."""
    parties = {_gstin(row.get("seller_tax_id")), _gstin(row.get("buyer_tax_id"))} - {None}
    hits = {c for c in companies
            if (gstins.get(c) and gstins[c] in parties)
            or names_match(c, row.get("seller_name")) or names_match(c, row.get("buyer_name"))}
    if len(hits) > 1:
        seller = _gstin(row.get("seller_tax_id"))

        def sells(c):  # by GSTIN when both are known, else by name
            return gstins[c] == seller if gstins.get(c) and seller else names_match(c, row.get("seller_name"))

        hits = {c for c in hits if sells(c)}
    return hits.pop() if len(hits) == 1 else None  # no company, or the seller not clear: ask a person


GSTINS_KEPT_SECONDS = 60


def _gstins_for_guessing(store):
    """The companies' GSTINs, worked out again at most once a minute (kept on the store): a folder of
    thousands of invoices uploaded without a company otherwise read every saved invoice again for each
    one of them."""
    now = time.monotonic()
    kept = getattr(store, "_gstins_for_guessing", None)
    if kept is None or now - kept[0] > GSTINS_KEPT_SECONDS:
        kept = (now, company_gstins(store.money_rows(), store.own_companies()))
        store._gstins_for_guessing = kept
    return kept[1]


def auto_company(store, invoice_id):
    """Put an invoice uploaded without a company into the known company it names."""
    row = store.get(invoice_id)
    if not row or row.get("company") or row.get("status") != "done":
        return None
    companies = store.company_names()
    if not companies:
        return None
    hit = guess_company(row, companies, _gstins_for_guessing(store))
    if hit:
        store.set_company([invoice_id], hit)
    return hit


def file_without_company(store, passes=3):
    """Put every processed invoice without a company into the one known company it names (seller or
    buyer), keeping a side set by hand. Run when your companies are saved (new invoices go there as
    they are processed). A company's GSTIN learned from the invoices just filed by its name finds the
    ones where its name was misread, in the next pass. Returns how many were moved."""
    moved = 0
    for _ in range(passes):
        store._gstins_for_guessing = None  # your companies, or what is in their folders, changed
        companies = store.company_names()
        if not companies:
            break
        gstins = _gstins_for_guessing(store)
        now = 0
        for row in store.export_rows(status="done", company=NO_COMPANY):
            hit = guess_company(row, companies, gstins)
            if hit:
                now += store.set_company([row["id"]], hit, row.get("entry_type"))
        moved += now
        if not now:
            break
    store._gstins_for_guessing = None
    return moved


def _period(period, today):
    """(period, first date, last date, month keys or None for one bucket per year)."""
    fy_start = today.year if today.month >= 4 else today.year - 1  # Indian financial year
    if period == "fy":
        return period, f"{fy_start}-04-01", f"{fy_start + 1}-03-31", \
            [f"{fy_start + (m > 12)}-{(m - 1) % 12 + 1:02d}" for m in range(4, 16)]
    if period == "lfy":
        return period, f"{fy_start - 1}-04-01", f"{fy_start}-03-31", \
            [f"{fy_start - 1 + (m > 12)}-{(m - 1) % 12 + 1:02d}" for m in range(4, 16)]
    if period == "all":
        return period, "0000-01-01", "9999-12-31", None
    y, m = today.year, today.month
    keys = []
    for _ in range(12):
        keys.insert(0, f"{y}-{m:02d}")
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    return "12m", f"{keys[0]}-01", f"{today.year}-{today.month:02d}-31", keys


@lru_cache(maxsize=65536)
def _party_key(name, gstin):
    return _gstin(gstin) or re.sub(r"\s+", " ", (name or "").strip()).upper()


def _top_parties(rows, side, limit=10):
    groups = {}
    for r in rows:
        name = (r.get(f"{side}_name") or "").strip()
        if not name:
            continue
        g = groups.setdefault(_party_key(name, r.get(f"{side}_tax_id")),
                              {"names": Counter(), "gstin": None, "invoices": 0, "total": 0.0, "tax": 0.0})
        g["names"][name] += 1
        g["gstin"] = g["gstin"] or r.get(f"{side}_tax_id")
        g["invoices"] += 1
        g["total"] += r.get("total") or 0.0
        g["tax"] += r.get("tax") or 0.0
    out = [{"vendor_key": k, "name": g["names"].most_common(1)[0][0], "gstin": g["gstin"],
            "invoices": g["invoices"], "total": round(g["total"], 2), "tax": round(g["tax"], 2)}
           for k, g in groups.items()]
    return sorted(out, key=lambda v: (-v["total"], v["name"]))[:limit]


def _project_rows(rows):
    """One row per project (and one for the bills in no project): sales and purchases without GST, and
    the margin. A bill whose GST was not found counts with its whole total (no_tax says how many)."""
    groups = {}
    for r in rows:
        g = groups.setdefault(r.get("project") or None, {
            "name": r.get("project") or None, "invoices": 0, "sales": 0.0, "purchases": 0.0, "sales_n": 0,
            "purchases_n": 0, "no_tax": 0})
        side = "sales" if r["side"] == SALE else "purchases"
        if r.get("tax") is None and r.get("total"):
            g["no_tax"] += 1
        g[side] += (r.get("total") or 0.0) - (r.get("tax") or 0.0)
        g[f"{side}_n"] += 1
        g["invoices"] += 1
    for g in groups.values():
        g["sales"], g["purchases"] = round(g["sales"], 2), round(g["purchases"], 2)
        g["margin"] = round(g["sales"] - g["purchases"], 2)
        g["margin_pct"] = round(100 * g["margin"] / g["sales"], 1) if g["sales"] > 0 else None
    return sorted(groups.values(),
                  key=lambda g: (g["name"] is None, -(g["sales"] + g["purchases"]), (g["name"] or "").lower()))


def build_dashboard(rows, currency=None, period="12m", today=None, company=None, known=(), own=(), project=None):
    """Dashboard figures from the counted invoices (MONEY_FIELDS), for ONE currency (amounts in different
    currencies are never added together), one period of invoice dates, all or one company and all or one
    project.
    period: '12m' last 12 months, 'fy' this financial year (Apr-Mar), 'lfy' last financial year,
    'all' everything (shown per year). known: every company name (also those with nothing counted yet).
    own: your companies [{"name", "gstin"}]. project: a project's name, NO_PROJECT or None (all)."""
    today = today or datetime.now().date()
    gstins = company_gstins(rows, own)
    companies = sorted({r["company"] for r in rows if r.get("company")} | set(known), key=str.lower)
    has_unassigned = any(not r.get("company") for r in rows)
    projects = sorted({r["project"] for r in rows if r.get("project")}, key=str.lower)
    if company == NO_COMPANY:
        rows = [r for r in rows if not r.get("company")]
    elif company in companies:
        # the company's books: its folder, and the bills in another folder (or none) that name its GSTIN,
        # like a bill from your other company: a purchase here, a sale in the seller's books
        rows = [r for r in rows if r.get("company") == company or names_company(r, company, gstins.get(company))]
    else:
        company = None
    if project == NO_PROJECT:
        rows = [r for r in rows if not r.get("project")]
    elif project in projects:
        rows = [r for r in rows if r.get("project") == project]
    else:
        project = None
    base = {"companies": companies, "company": company, "has_unassigned": has_unassigned,
            "company_gstin": gstins.get(company), "own_companies": bool(own), "projects": projects,
            "project": project}

    counts, last = Counter(), {}
    for r in rows:
        if r.get("currency"):
            counts[r["currency"]] += 1
            last[r["currency"]] = max(last.get(r["currency"], 0), r["id"])
    # invoices whose currency was not found cannot be added to any total: the page says how many
    base["no_currency"] = sum(1 for r in rows if not r.get("currency"))
    currencies = [{"currency": c, "n": n} for c, n in sorted(counts.items(), key=lambda cn: (-cn[1], -last[cn[0]]))]
    if not currencies:
        return {**base, "currencies": [], "currency": None, "period": period, "buckets": [], "vendors": [],
                "customers": [], "company_rows": [], "project_rows": [], "summary": None, "undated": 0,
                "internal": 0, "by_year": False}
    if currency not in counts:
        currency = currencies[0]["currency"]
    period, start, end, month_keys = _period(period, today)

    mine = [r for r in rows if r.get("currency") == currency]
    undated = sum(1 for r in mine if not r.get("invoice_date"))
    dated = [r for r in mine if r.get("invoice_date") and start <= r["invoice_date"] <= end]
    viewed = company if company and company != NO_COMPANY else None
    for r in dated:
        r["side"] = side_seen_by(r, viewed, gstins)
        if viewed:
            r["company"] = viewed  # counted in the books shown
    # all companies together: the bills between your companies are not the group's sales or purchases
    group = company is None
    counted = [r for r in dated if not (group and between_your_companies(r, own, gstins))]

    width = 7 if month_keys is not None else 4
    buckets = {k: None for k in (month_keys or sorted({r["invoice_date"][:4] for r in dated}))}
    for k in buckets:
        buckets[k] = {"key": k, "invoices": 0, "total": 0.0, "tax": 0.0, "debit": 0.0, "credit": 0.0,
                      "debit_tax": 0.0, "credit_tax": 0.0}
    summary = {"invoices": len(counted), "total": 0.0, "tax": 0.0, "debit": 0.0, "credit": 0.0, "debit_n": 0,
               "credit_n": 0, "tax_in": 0.0, "tax_out": 0.0}
    for r in counted:
        total, tax = r.get("total") or 0.0, r.get("tax") or 0.0
        side, money = ("debit", "tax_in") if r["side"] == PURCHASE else ("credit", "tax_out")
        b = buckets.get(r["invoice_date"][:width])
        if b is not None:
            b["invoices"] += 1
            b["total"] += total
            b["tax"] += tax
            b[side] += total
            b[f"{side}_tax"] += tax
        summary["total"] += total
        summary["tax"] += tax
        summary[side] += total
        summary[f"{side}_n"] += 1
        summary[money] += tax
    per_company = {}  # each folder with all its bills, as before (a bill to your other company is a sale here)
    for r in dated:
        total = r.get("total") or 0.0
        side = "debit" if r["side"] == PURCHASE else "credit"
        c = per_company.setdefault(r.get("company") or NO_COMPANY, {
            "name": r.get("company"), "invoices": 0, "debit": 0.0, "credit": 0.0, "debit_n": 0, "credit_n": 0})
        c["invoices"] += 1
        c[side] += total
        c[f"{side}_n"] += 1
    for b in buckets.values():
        for key in ("total", "tax", "debit", "credit", "debit_tax", "credit_tax"):
            b[key] = round(b[key], 2)
        b["taxable"] = round(b["total"] - b["tax"], 2)
    for key in ("total", "tax", "debit", "credit", "tax_in", "tax_out"):
        summary[key] = round(summary[key], 2)
    summary["net"] = round(summary["credit"] - summary["debit"], 2)
    summary["gst_net"] = round(summary["tax_out"] - summary["tax_in"], 2)  # > 0: GST to pay
    purchases = [r for r in counted if r["side"] == PURCHASE]
    sales = [r for r in counted if r["side"] == SALE]
    summary["vendors"] = len({_party_key(r.get("seller_name"), r.get("seller_tax_id"))
                              for r in purchases if (r.get("seller_name") or "").strip()})
    summary["companies"] = len({r["company"] for r in dated if r.get("company")})
    company_rows = []
    for c in per_company.values():
        c["debit"], c["credit"] = round(c["debit"], 2), round(c["credit"], 2)
        c["net"] = round(c["credit"] - c["debit"], 2)
        c["gstin"] = gstins.get(c["name"])
        company_rows.append(c)
    company_rows.sort(key=lambda c: (c["name"] is None, -(c["debit"] + c["credit"]), (c["name"] or "").lower()))
    return {**base, "currencies": currencies, "currency": currency, "period": period,
            "buckets": list(buckets.values()), "vendors": _top_parties(purchases, "seller"),
            "customers": _top_parties(sales, "buyer"), "company_rows": company_rows,
            "project_rows": _project_rows(counted), "summary": summary, "undated": undated,
            "internal": len(dated) - len(counted), "by_year": month_keys is None}


def folders(counts, money_rows, own=()):
    """The Companies page: one folder per company (and one for invoices without a company); your own
    companies have one even while it is empty.
    counts: {company or None: {"invoices", "needs_review", "last_added"}} from the store."""
    gstins = company_gstins(money_rows, own)
    counts = dict(counts)
    for name in known_companies((), own):
        if not any(c and c.lower() == name.lower() for c in counts):
            counts[name] = {"invoices": 0, "needs_review": 0, "last_added": None}
    money = defaultdict(lambda: defaultdict(lambda: {"debit": 0.0, "credit": 0.0}))
    for r in money_rows:
        if not r.get("currency"):
            continue
        side = "debit" if side_of(r, gstins.get(r.get("company"))) == PURCHASE else "credit"
        money[r.get("company")][r["currency"]][side] += r.get("total") or 0.0
    out = []
    for name, c in counts.items():
        by_currency = money.get(name) or {}
        currency = max(by_currency, key=lambda cur: sum(by_currency[cur].values()), default=None)
        amounts = by_currency.get(currency) or {"debit": 0.0, "credit": 0.0}
        out.append({"name": name, "gstin": gstins.get(name), "invoices": c["invoices"],
                    "needs_review": c["needs_review"], "last_added": c["last_added"], "currency": currency,
                    "debit": round(amounts["debit"], 2), "credit": round(amounts["credit"], 2),
                    "other_currencies": sorted(set(by_currency) - {currency})})
    out.sort(key=lambda f: (f["name"] is None, (f["name"] or "").lower()))
    return out
