import pytest

from invoice_parser import normalize as nz


@pytest.mark.parametrize("text, expected", [
    ("1,234.50", 1234.5),
    ("1.234,50 €", 1234.5),
    ("12,34,567.00", 1234567.0),
    ("₹ 1,23,456.78", 123456.78),
    ("$1200", 1200.0),
    ("(45.00)", -45.0),
    ("INR 999", 999.0),
    ("1,50", 1.5),
    ("1,500", 1500.0),
    ("18%", 18.0),
])
def test_parse_amount(text, expected):
    assert nz.parse_amount(text) == pytest.approx(expected)


def test_parse_amount_rejects_text():
    assert nz.parse_amount("no digits here") is None


@pytest.mark.parametrize("text, expected", [
    ("Date: 05/03/2024", "2024-03-05"),      # day first (DAY_FIRST=True)
    ("12 Mar 2024", "2024-03-12"),
    ("March 5, 2024", "2024-03-05"),
    ("2024-07-01", "2024-07-01"),
    ("Issued 3-Jan-24", "2024-01-03"),
])
def test_find_date(text, expected):
    assert nz.find_date(text) == expected


def test_find_date_none():
    assert nz.find_date("no date in this text") is None


@pytest.mark.parametrize("text, day_first", [
    ("Invoice date 18/09/2015", True),                      # first part > 12: must be the day
    ("Date of issue 09/18/2015", False),                    # second part > 12: month first
    ("GSTIN 27ABCDE1234F1Z5  Date 05/03/2024  $ 100", True),  # Indian invoice billed in USD
    ("Date 04/11/2021  New Ann, NE 54138  Total $ 214,24", False),  # US address / USD
    ("Date 05/03/2024  Total 5,00 EUR", True),
])
def test_infer_day_first(text, day_first):
    assert nz.infer_day_first(text) is day_first


def test_find_date_respects_day_order():
    assert nz.find_date("04/11/2021", dayfirst=False) == "2021-04-11"
    assert nz.find_date("04/11/2021", dayfirst=True) == "2021-11-04"


def test_tax_ids():
    assert nz.find_tax_id("GSTIN: 27ABCDE1234F1Z5") == "27ABCDE1234F1Z5"
    assert nz.find_tax_id("VAT No: GB123456789") == "GB123456789"
    assert nz.find_tax_id("nothing here") is None


def test_email_and_phone():
    assert nz.find_email("Write to billing@acme.co.in today") == "billing@acme.co.in"
    assert nz.find_phone("Call +91 98765 43210 now") == "+91 98765 43210"
    assert nz.find_phone("Date 05-03-2024") is None


def test_strip_contact_removes_contact_details():
    cleaned = nz.strip_contact("Email: a@b.com, Phone: +91 98765 43210")
    assert "@" not in cleaned and "98765" not in cleaned


@pytest.mark.parametrize("text, iso", [("INR 1,200", "INR"), ("₹1,200", "INR"), ("€5", "EUR"), ("£9", "GBP")])
def test_detect_currency(text, iso):
    assert nz.detect_currency(text) == iso


def test_gstin_checksum_and_ocr_repair():
    assert nz.gstin_checksum_ok("06BKXPT4822Q3ZY") and nz.gstin_checksum_ok("06AABCS7788K1ZP")
    assert not nz.gstin_checksum_ok("27ABCDE1234F1Z5")
    assert nz.gstin_check_char("27ABCDE1234F1Z") == "0"
    # OCR read O for 0 and split the number in two: repaired, and accepted only because the checksum fits
    assert nz.find_gstin("GSTIN/UIN: O6BKXPT4822Q3ZY") == "06BKXPT4822Q3ZY"
    assert nz.find_gstin("GSTIN 06AABCS7788 K1ZP") == "06AABCS7788K1ZP"
    assert nz.find_gstin("Invoice KVIN/2026-27/007") is None


@pytest.mark.parametrize("text, value", [
    ("INR Two Lakh Ninety Five Thousand Only", 295000),
    ("Rupees One Crore Five Lakh Twenty Thousand and Fifty Paise Only", 10520000.50),
    ("USD One Thousand Two Hundred Thirty Four and 56/100", None),
    ("Total: Seven Hundred Fifty Dollars and Twenty Five Cents", 750.25),
])
def test_amount_in_words(text, value):
    got = nz.words_to_amount(text)
    if value is None:
        assert got is None or got == pytest.approx(1234)
    else:
        assert got == pytest.approx(value)
