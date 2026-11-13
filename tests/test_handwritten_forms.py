"""Bill-book invoices filled in by hand: a printed form (seller, labels, boxes) with the buyer, number,
date, items and amounts written in. The OCR side is tested in test_fast_photo_ocr.py; these tests feed
positioned words straight into the layout and extraction rules."""
import numpy as np
import pytest

from invoice_parser import normalize as nz
from invoice_parser.extract import extract_invoice
from invoice_parser.layout import build_lines, join_paise
from invoice_parser.matcher import LabelMatcher
from invoice_parser.models import Page, Word


@pytest.fixture(scope="module")
def matcher():
    return LabelMatcher(use_embeddings=False)


def _row(y, *cells, conf=0.9, h=0.014):
    """Words of one printed / written row: cells are (text, x0); each word ~0.011 of the page per letter."""
    words = []
    for text, x in cells:
        for token in text.split():
            width = 0.011 * len(token)
            words.append(Word(token, x, y, x + width, y + h, conf, 0))
            x += width + 0.008
    return words


def _fields(words, matcher):
    result = extract_invoice(build_lines([Page(0, words, "ocr", 0.9)]), matcher)
    f = result["fields"]
    v = lambda k: (f.get(k) or {}).get("value")
    return {"invoice_number": v("invoice_number"), "invoice_date": v("invoice_date"),
            "seller": (v("seller") or {}).get("name"), "seller_tax_id": (v("seller") or {}).get("tax_id"),
            "buyer": (v("buyer") or {}).get("name"), "buyer_tax_id": (v("buyer") or {}).get("tax_id"),
            "subtotal": v("subtotal"), "tax": v("tax"), "total": v("total")}, result


def test_a_bill_book_with_rupee_and_paise_columns(matcher):
    words = (_row(0.02, ("GSTIN : 09DFIPA7756B1ZH", 0.03), ("Tax Invoice", 0.45), ("Mob. : 8535000042", 0.78))
             + _row(0.05, ("M/S YASH TRADERS", 0.28))
             + _row(0.09, ("Add. 01, Krishna Vihar, Meerut Road, Hapur-245101 (U.P.)", 0.15))
             + _row(0.14, ("Invoice No.", 0.03), ("State : UP / State Code : 09", 0.36), ("Date :", 0.66))
             + _row(0.155, ("14", 0.16), ("11/09/2026", 0.73))
             + _row(0.19, ("Name :", 0.03), ("Deepnap Softech", 0.10), ("Transport Name :", 0.60))
             + _row(0.23, ("Address :", 0.03), ("121 B, Industrial estate", 0.12), ("Vehicle No. :", 0.60))
             + _row(0.27, ("GSTIN :", 0.03), ("06BMxPS9096Q3ZB", 0.10), ("Place of Supply :", 0.60))
             + _row(0.32, ("S.No.", 0.03), ("DESCRIPTION OF GOODS", 0.09), ("HSN CODE", 0.46), ("QTY.", 0.58),
                    ("Rate", 0.66), ("Rs.", 0.80), ("P.", 0.94))
             + _row(0.36, ("1", 0.04), ("Carrier 2TN Split A/c", 0.10), ("8415", 0.47), ("1", 0.60),
                    ("42,288.14", 0.66), ("42288", 0.80), ("14", 0.95))
             + _row(0.70, ("Total Amount in words", 0.03), ("Total Amount Before Tax", 0.57), ("42288", 0.80),
                    ("14", 0.95))
             + _row(0.715, ("Forty nine thousand", 0.19))
             + _row(0.73, ("Add : CGST@ ....%", 0.57), ("3805", 0.80), ("93", 0.95))
             + _row(0.745, ("nine hundred only", 0.04))
             + _row(0.76, ("Add : SGST@ ....%", 0.57), ("3805=93", 0.80))
             + _row(0.78, ("Bank Name : Union Bank of India", 0.03), ("Add : IGST@ ....%", 0.57))
             + _row(0.80, ("A/c No. : 306201010036594", 0.03), ("Round Off", 0.57))
             + _row(0.83, ("Total Amount After Tax", 0.57), ("49900t", 0.80))
             + _row(0.90, ("For : M/S YASH TRADERS", 0.62))
             + _row(0.95, ("Authorised Signatory", 0.80)))
    got, result = _fields(words, matcher)
    assert got == {"invoice_number": "14", "invoice_date": "2026-09-11", "seller": "M/S YASH TRADERS",
                   "seller_tax_id": "09DFIPA7756B1ZH", "buyer": "Deepnap Softech", "buyer_tax_id": "06BMXPS9096Q3ZB",
                   "subtotal": 42288.14, "tax": 7611.86, "total": 49900.0}
    assert result["checks"]["total_matches_amount_in_words"]  # 'Forty nine thousand' + 'nine hundred only'
    assert result["checks"]["subtotal_plus_tax_minus_discount_equals_total"]


def test_a_dealer_bill_book_with_total_and_grand_total(matcher):
    words = (_row(0.02, ("GSTIN : 06CAEPG1459F1Z0", 0.03), ("TAX INVOICE", 0.44), ("M : 8851510153", 0.80))
             + _row(0.05, ("Shri RamElectricals", 0.26))
             + _row(0.10, ("1st Floor, 1D-43, Opp. Milp Dawakhana, Faridabad, Haryana, 121001", 0.20))
             + _row(0.145, ("No.", 0.03), ("Date", 0.60))
             + _row(0.155, ("695", 0.09), ("9-9-2026", 0.71))
             + _row(0.175, ("Tax is payable on Reverse Charge : (yes/No)", 0.60))
             + _row(0.19, ("M/s.", 0.03), ("DEEPAK SHARMA", 0.11), ("P.O. No.", 0.60), ("Date", 0.72))
             + _row(0.22, ("Sec-31 HSIIDC FBD", 0.06), ("Ch. No.", 0.60), ("Date", 0.71))
             + _row(0.25, ("Industrial estate", 0.06), ("Vehicle No.", 0.60))
             + _row(0.27, ("Party GST No.", 0.03), ("06BMXPS9096Q3ZB", 0.17), ("State Code. 06", 0.45))
             + _row(0.33, ("S.No. PARTICULARS", 0.03), ("HSN CODE", 0.46), ("QTY.", 0.58), ("RATE", 0.66),
                    ("Rs.", 0.78), ("P.", 0.93))
             + _row(0.36, ("1)", 0.04), ('48" Ceiling Fan Crompton', 0.11), ("8414", 0.47), ("7 nos", 0.58),
                    ("1986.0", 0.67), ("13902.1", 0.79))
             + _row(0.65, ("TOTAL", 0.66), ("13902.1", 0.78))
             + _row(0.68, ("Mobile No :", 0.03), ("8368416280-9315550183", 0.16), ("R/OFF", 0.66))
             + _row(0.71, ("CGST @ 9 %", 0.66), ("1251.1", 0.79))
             + _row(0.73, ("Rupees (in words)", 0.03), ("Sixteen thousand four", 0.21), ("SGST @ 9 %", 0.66),
                    ("1251.1", 0.78))
             + _row(0.77, ("hundred five only", 0.05), ("IGST @ %", 0.66))
             + _row(0.80, ("GRAND TOTAL", 0.66), ("16405", 0.80))
             + _row(0.92, ("E. & O.E.", 0.03), ("For Shri RamElectricals", 0.68))
             + _row(0.96, ("Prop./(Auth. Signatory)", 0.74)))
    got, result = _fields(words, matcher)
    assert got == {"invoice_number": "695", "invoice_date": "2026-09-09", "seller": "Shri RamElectricals",
                   "seller_tax_id": "06CAEPG1459F1Z0", "buyer": "DEEPAK SHARMA", "buyer_tax_id": "06BMXPS9096Q3ZB",
                   "subtotal": 13902.1, "tax": 2502.2, "total": 16405.0}
    # the 0.70 between 13902.1 + 2502.2 and 16405 is the round-off, not a misread tax
    assert not any("corrected" in w for w in result["warnings"])


def test_rupees_and_paise_columns_become_one_amount():
    words = (_row(0.40, ("Rs.", 0.80), ("P.", 0.94)) + _row(0.70, ("Total", 0.60), ("42288", 0.80), ("14", 0.95))
             + _row(0.75, ("3805=93", 0.80)) + _row(0.80, ("42288/14", 0.84)))
    assert [w.text for w in join_paise(words)] == ["Rs.", "P.", "Total", "42288.14", "3805.93", "42288.14"]


def test_amounts_are_left_alone_without_a_paise_column_or_in_pdf_text():
    words = _row(0.70, ("Total", 0.60), ("1", 0.70), ("NOS", 0.75), ("42288", 0.80), ("14", 0.95))
    assert [w.text for w in join_paise(words)] == ["Total", "1", "NOS", "42288", "14"]
    pdf = _row(0.40, ("Rs.", 0.80), ("P.", 0.94), conf=1.0) + _row(0.70, ("42288", 0.80), ("14", 0.95), conf=1.0)
    assert [w.text for w in join_paise(pdf)] == ["Rs.", "P.", "42288", "14"]


def test_handwritten_gstin_and_ocr_tax_id_label():
    assert nz.find_gstin("06BmxPS9096032B") == "06BMXPS9096Q3ZB"  # a handwritten Q read as 0, Z as 2
    assert nz.find_tax_id("Tax ld: 994-72-1270") == "994-72-1270"  # 'Id' read as 'ld'


def test_words_written_without_a_space_are_split_at_the_gap():
    from rapidocr.ch_ppocr_rec.typings import WordInfo, WordType

    from invoice_parser import fast_ocr

    info = WordInfo(words=[list("DEEPAKSHARMA")], word_cols=[[3, 5, 8, 11, 14, 16, 21, 24, 27, 29, 33, 35]],
                    word_types=[WordType.EN_NUM])
    text, _, info = fast_ocr._with_gaps("DEEPAKSHARMA", 0.97, info)
    assert text == "DEEPAK SHARMA" and info.words == [list("DEEPAK"), list("SHARMA")]
    printed = WordInfo(words=[list("RamElectricals")], word_cols=[[2, 4, 9, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]],
                       word_types=[WordType.EN_NUM])
    assert fast_ocr._with_gaps("RamElectricals", 0.99, printed)[0] == "RamElectricals"  # printed: kept


def test_a_box_printed_on_a_scan_is_not_taken_for_the_page_edge():
    from invoice_parser.preprocess import _find_document_quad, _paper_around

    page = np.full((1400, 1000), 245, np.uint8)
    page[40:60, 100:600] = 20  # the header printed above the form's box
    page[150:1350, 50:950] = 245
    for y0, y1, x0, x1 in ((150, 154, 50, 950), (1346, 1350, 50, 950), (150, 1350, 50, 54), (150, 1350, 946, 950)):
        page[y0:y1, x0:x1] = 20
    quad = _find_document_quad(page)
    assert quad is not None and _paper_around(page, quad)  # found, but white paper all round it: keep the page
