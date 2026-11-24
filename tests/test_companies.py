"""Company folders: names, your companies, the company's own GSTIN, purchase / sale and the dashboard
figures. Made-up firms and GSTINs only."""
from datetime import date

from invoice_parser.companies import (NO_COMPANY, PURCHASE, SALE, build_dashboard, clean_company,
                                      company_gstins, folders, guess_company, known_companies, names_match,
                                      own_companies_text, parse_own_companies, side_of)

OWN, SUPPLIER, CUSTOMER = "06BKXPT4822Q3ZY", "09AGHPR3344C1ZG", "06AJKPM6655F1ZZ"


def _row(n, company, seller, seller_gstin, buyer, buyer_gstin, total, tax, day="2026-10-05", currency="INR"):
    return {"id": n, "company": company, "entry_type": None, "currency": currency, "invoice_date": day,
            "total": total, "tax": tax, "seller_name": seller, "seller_tax_id": seller_gstin,
            "buyer_name": buyer, "buyer_tax_id": buyer_gstin}


ROWS = [
    _row(1, "JP", "OM TRADERS", SUPPLIER, "J.P. Enterprises Pvt Ltd", OWN, 1180.0, 180.0),
    _row(2, "JP", "OM TRADERS", SUPPLIER, "JP Enterprises", OWN, 590.0, 90.0, day="2026-09-12"),
    _row(3, "JP", "J P ENTERPRISES", OWN, "Shri SunElectricals", CUSTOMER, 2360.0, 360.0),
    _row(4, None, "Kavya Infotech", None, "SURESH MITTAL", None, 100.0, 0.0),
    _row(5, "JP", "Kavya Infotech", None, "Someone", None, 50.0, 0.0, currency="USD"),
]


def test_names_match():
    assert names_match("JP", "J.P. Enterprises Pvt Ltd") and names_match("JP", "JP Traders")
    assert names_match("Kavya Infotech", "M/s Kavya Infotech Pvt. Ltd.")
    assert names_match("Ganesha Softech", "Ganesba Soflech")  # small misreading
    assert not names_match("JP", "OM TRADERS") and not names_match("Kavya Infotech", "Kavita Traders")
    assert not names_match("", "anything") and not names_match("JP", None)


def test_clean_company():
    assert clean_company("  jp   company ") == "jp company"
    assert clean_company("jp", ["JP", "OM"]) == "JP"
    assert clean_company("") is None and clean_company(NO_COMPANY) is None


def test_the_company_gstin_and_the_side_of_each_invoice():
    gstins = company_gstins(ROWS)
    assert gstins == {"JP": OWN}
    assert [side_of(r, gstins.get(r["company"])) for r in ROWS] == [PURCHASE, PURCHASE, SALE, PURCHASE, PURCHASE]
    assert side_of({**ROWS[0], "entry_type": SALE}, OWN) == SALE  # set by hand wins
    # no GSTINs at all: the name says which side the company is on
    assert side_of(_row(9, "Kavya Infotech", "Kavya Infotech", None, "OM", None, 1.0, 0.0)) == SALE
    # the GSTIN on most invoices is the company's even when its name differs from the folder's
    rows = [_row(i, "Folder X", "Supplier", f"07AAAAA000{i}A1Z5", "Owner Co", OWN, 1.0, 0.0) for i in range(3)]
    assert company_gstins(rows) == {"Folder X": OWN}


def test_guess_company():
    gstins = {"JP": OWN, "OM TRADERS": None}
    both = ["JP", "OM TRADERS"]
    assert guess_company(_row(1, None, "Anyone", SUPPLIER, "Somebody", OWN, 1.0, 0.0), both, gstins) == "JP"
    assert guess_company(_row(1, None, "OM Traders", None, "X", None, 1.0, 0.0), both, gstins) == "OM TRADERS"
    # both companies on one invoice (one sells to the other): the seller's; none: no guess
    assert guess_company(_row(1, None, "OM Traders", None, "JP", OWN, 1.0, 0.0), both, gstins) == "OM TRADERS"
    assert guess_company(_row(1, None, "J.P. Enterprises", OWN, "OM Traders", None, 1.0, 0.0), both, gstins) == "JP"
    assert guess_company(_row(1, None, "Nobody", None, "X", None, 1.0, 0.0), ["JP"], gstins) is None


def test_dashboard_figures():
    d = build_dashboard([dict(r) for r in ROWS], currency="INR", period="fy", today=date(2026, 11, 17))
    s = d["summary"]
    assert (s["debit"], s["credit"], s["debit_n"], s["credit_n"]) == (1870.0, 2360.0, 3, 1)
    assert (s["net"], s["tax_in"], s["tax_out"], s["gst_net"]) == (490.0, 270.0, 360.0, 90.0)
    assert s["companies"] == 1 and s["vendors"] == 2
    assert [c["name"] for c in d["company_rows"]] == ["JP", None]
    assert d["company_rows"][0] == {"name": "JP", "invoices": 3, "debit": 1770.0, "credit": 2360.0, "debit_n": 2,
                                    "credit_n": 1, "net": 590.0, "gstin": OWN}
    oct_ = next(b for b in d["buckets"] if b["key"] == "2026-10")
    assert (oct_["debit"], oct_["credit"], oct_["invoices"]) == (1280.0, 2360.0, 3)
    assert [v["name"] for v in d["vendors"]] == ["OM TRADERS", "Kavya Infotech"]
    assert [v["name"] for v in d["customers"]] == ["Shri SunElectricals"]

    jp = build_dashboard([dict(r) for r in ROWS], period="all", company="JP")
    assert jp["company"] == "JP" and jp["company_gstin"] == OWN and jp["summary"]["invoices"] == 3
    assert [c["currency"] for c in jp["currencies"]] == ["INR", "USD"]
    none = build_dashboard([dict(r) for r in ROWS], period="all", company=NO_COMPANY)
    assert none["summary"]["invoices"] == 1 and none["has_unassigned"]
    unknown = build_dashboard([dict(r) for r in ROWS], period="all", company="Nope")
    assert unknown["company"] is None and unknown["summary"]["invoices"] == 4


def test_folders():
    counts = {"JP": {"invoices": 4, "needs_review": 1, "last_added": "2026-11-01 10:00:00"},
              None: {"invoices": 1, "needs_review": 0, "last_added": "2026-11-02 10:00:00"}}
    out = folders(counts, ROWS)
    assert [f["name"] for f in out] == ["JP", None]
    assert (out[0]["currency"], out[0]["debit"], out[0]["credit"]) == ("INR", 1770.0, 2360.0)
    assert out[0]["other_currencies"] == ["USD"] and out[0]["gstin"] == OWN and out[1]["debit"] == 100.0


def test_a_folder_of_uploads_reads_the_saved_invoices_once():
    """Thousands of invoices uploaded without a company: the companies' GSTINs are worked out once a
    minute, not again for every invoice (that read every saved invoice for each one)."""
    from invoice_parser.companies import auto_company

    class Store:
        reads, moved = 0, {}

        def money_rows(self):
            Store.reads += 1
            return ROWS

        def company_names(self):
            return ["JP"]

        def own_companies(self):
            return []

        def get(self, n):
            return {**_row(n, None, "OM TRADERS", SUPPLIER, "Anyone", OWN, 1.0, 0.0), "status": "done"}

        def set_company(self, ids, company):
            Store.moved.update({i: company for i in ids})

    store = Store()
    for n in range(10, 60):
        assert auto_company(store, n) == "JP"
    assert Store.reads == 1 and len(Store.moved) == 50


def test_your_companies_are_read_from_the_box():
    text = ("Kavya Infotech Pvt Ltd, GSTIN/UIN: 06bkxpt4822q3zy\n  om   traders \n\n"
            "09AGHPR3344C1ZG\nOM Traders\nGST Seva Kendra")
    firms = parse_own_companies(text)
    assert firms == [{"name": "Kavya Infotech Pvt Ltd", "gstin": OWN}, {"name": "om traders", "gstin": None},
                     {"name": "GST Seva Kendra", "gstin": None}]  # a GSTIN alone or a name twice: left out
    assert parse_own_companies(own_companies_text(firms)) == firms
    assert parse_own_companies("") == [] and parse_own_companies(None) == []
    assert known_companies(["kavya infotech pvt ltd", "JP"], firms) == ["GST Seva Kendra", "JP",
                                                                         "kavya infotech pvt ltd", "om traders"]


def test_your_companies_decide_purchase_or_sale():
    """Bills uploaded without a company: your company is the seller -> a sale, the buyer -> a purchase,
    found by the GSTIN typed for it even when its name was misread."""
    own = [{"name": "Kavya Infotech", "gstin": OWN}, {"name": "OM TRADERS", "gstin": None}]
    gstins = company_gstins([], own)
    assert gstins == {"Kavya Infotech": OWN, "OM TRADERS": None}
    names = known_companies([], own)
    sale = _row(1, None, "Kavya lnfotecb", OWN, "Shri SunElectricals", CUSTOMER, 2360.0, 360.0)
    bought = _row(2, None, "Ganesha Softech", SUPPLIER, "M/s Kavya Infotech Pvt Ltd", OWN, 1180.0, 180.0)
    other = _row(3, None, "Ganesha Softech", SUPPLIER, "OM Traders", None, 590.0, 90.0)
    rows = [sale, bought, other]
    for r in rows:
        r["company"] = guess_company(r, names, gstins)
    assert [r["company"] for r in rows] == ["Kavya Infotech", "Kavya Infotech", "OM TRADERS"]
    assert [side_of(r, gstins.get(r["company"])) for r in rows] == [SALE, PURCHASE, PURCHASE]
    # one of your companies selling to the other: a sale of the seller
    between = _row(4, None, "Kavya Infotech Pvt Ltd", OWN, "OM TRADERS", None, 118.0, 18.0)
    assert guess_company(between, names, gstins) == "Kavya Infotech"
    assert side_of({**between, "company": "Kavya Infotech"}, OWN) == SALE
    # the GSTIN typed for a company wins over the one learned from its invoices
    learned = [_row(i, "JP", "OM TRADERS", SUPPLIER, "JP Enterprises", CUSTOMER, 1.0, 0.0) for i in range(3)]
    assert company_gstins(learned) == {"JP": CUSTOMER}
    assert company_gstins(learned, [{"name": "JP", "gstin": OWN}]) == {"JP": OWN}
    d = build_dashboard([dict(r) for r in rows], period="all", known=names, own=own)
    assert (d["summary"]["credit"], d["summary"]["debit"], d["own_companies"]) == (2360.0, 1770.0, True)
    assert not build_dashboard([dict(r) for r in ROWS], period="all")["own_companies"]
    # your company has a folder even before any invoice is in it
    shown = folders({"Kavya Infotech": {"invoices": 2, "needs_review": 0, "last_added": "2026-11-20"}},
                    [dict(sale), dict(bought)], own)
    assert [(f["name"], f["invoices"], f["credit"]) for f in shown] == [("Kavya Infotech", 2, 2360.0),
                                                                        ("OM TRADERS", 0, 0.0)]


def test_invoices_without_a_currency_are_counted_as_missing_not_hidden():
    rows = ROWS + [_row(6, "JP", "OM TRADERS", SUPPLIER, "JP Enterprises", OWN, 999.0, 0.0, currency=None)]
    d = build_dashboard([dict(r) for r in rows], period="all", today=date(2026, 11, 18))
    assert d["no_currency"] == 1 and d["summary"]["total"] == build_dashboard(
        [dict(r) for r in ROWS], period="all", today=date(2026, 11, 18))["summary"]["total"]
    only = build_dashboard([dict(rows[-1])], period="all", today=date(2026, 11, 18))
    assert only["currencies"] == [] and only["no_currency"] == 1
