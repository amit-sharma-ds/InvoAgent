from io import BytesIO

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from invoice_parser.matcher import LabelMatcher
from invoice_parser.pipeline import InvoiceParser


@pytest.fixture(scope="module")
def parser():
    return InvoiceParser(matcher=LabelMatcher(use_embeddings=False))


def _to_pdf(draw):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    draw(c)
    c.save()
    return buf.getvalue()


def _colon_layout(c):
    """'Vendor:' / 'Bill To:' with values on the same line; details on the right."""
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, 790, "TAX INVOICE")
    c.setFont("Helvetica", 10)
    c.drawString(40, 760, "Vendor: Acme Traders Pvt Ltd")
    c.drawString(40, 747, "12 MG Road, Pune 411001")
    c.drawString(40, 734, "GSTIN: 27ABCDE1234F1Z0")
    c.drawString(330, 760, "Invoice No: INV-2024-0042")
    c.drawString(330, 747, "Invoice Date: 12/03/2024")
    c.drawString(330, 734, "Due Date: 11/04/2024")
    c.drawString(40, 690, "Bill To: Rahul Electronics")
    c.drawString(40, 677, "45 Park Street, Kolkata 700016")
    for x, h in zip([40, 340, 410, 490], ["Description", "Qty", "Rate", "Amount"]):
        c.drawString(x, 630, h)
    y = 614
    for row in [("Laptop Stand", "2", "400.00", "800.00"), ("USB-C Cable", "4", "25.00", "100.00")]:
        for x, cell in zip([40, 340, 410, 490], row):
            c.drawString(x, y, cell)
        y -= 16
    c.drawString(380, 540, "Subtotal:")
    c.drawRightString(540, 540, "900.00")
    c.drawString(380, 526, "Tax (18%):")
    c.drawRightString(540, 526, "162.00")
    c.drawString(380, 512, "Grand Total:")
    c.drawRightString(540, 512, "$ 1,062.00")


def _label_only_layout(c):
    """Headings without colons: 'BILL FROM' / 'BILL TO' columns, then 'Subtotal' 'Tax 18%' 'Total'."""
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, 800, "INVOICE")
    c.setFont("Helvetica-Bold", 9)
    c.drawString(40, 765, "BILL FROM")
    c.drawString(330, 765, "BILL TO")
    c.setFont("Helvetica", 10)
    c.drawString(40, 751, "Bluepeak Solutions LLP")
    c.drawString(330, 751, "Metro Retail Store")
    c.drawString(40, 738, "221 Baker Street, London NW1")
    c.drawString(330, 738, "City Hardware")
    c.drawString(330, 790, "Invoice Number")
    c.drawString(440, 790, "INV-5521")
    c.drawString(330, 776, "Date")
    c.drawString(440, 776, "07/07/2024")
    for x, h in zip([40, 340, 410, 490], ["Description", "Qty", "Rate", "Amount"]):
        c.drawString(x, 700, h)
    c.drawString(40, 684, "Keyboard")
    c.drawString(340, 684, "3")
    c.drawString(410, 684, "150.00")
    c.drawString(490, 684, "450.00")
    c.drawString(380, 640, "Subtotal")
    c.drawRightString(540, 640, "£450.00")
    c.drawString(380, 626, "Tax 18%")
    c.drawRightString(540, 626, "£81.00")
    c.drawString(380, 612, "Total")
    c.drawRightString(540, 612, "£531.00")


def _summary_table_layout(c):
    """Style of the public katanaml invoice set: Seller/Client columns, tax IDs after a blank line,
    item table with wrapped descriptions, and totals in a Net worth / VAT / Gross worth summary table."""
    c.setFont("Helvetica-Bold", 14)
    c.drawString(45, 800, "Invoice no: 65321852")
    c.setFont("Helvetica", 11)
    c.drawString(45, 775, "Date of issue:")
    c.drawString(330, 775, "04/11/2021")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(45, 700, "Seller:")
    c.drawString(330, 700, "Client:")
    c.setFont("Helvetica", 11)
    for y, left, right in [(680, "Kaufman, Cooper and Young", "Wells-Carlson"),
                           (666, "33451 Johnson Lake", "148 Carroll Village Suite 393"),
                           (652, "New Ann, NE 54138", "South Allisonstad, TX 72090")]:
        c.drawString(48, y, left)
        c.drawString(333, y, right)
    c.drawString(48, 620, "Tax Id: 930-79-7845")  # after a blank gap
    c.drawString(333, 620, "Tax Id: 957-82-7504")
    c.drawString(48, 606, "IBAN: GB85XPMM58300597200061")

    c.setFont("Helvetica-Bold", 8)
    c.drawString(45, 580, "No.")
    c.drawString(70, 580, "Description")
    c.drawRightString(270, 580, "Qty")
    c.drawString(280, 580, "UM")
    c.drawRightString(360, 580, "Net price")
    c.drawRightString(420, 580, "Net worth")
    c.drawRightString(480, 580, "VAT [%]")
    c.drawRightString(545, 580, "Gross worth")
    c.setFont("Helvetica", 8)
    rows = [(560, "1.", "New KID CUDI Rap Hip", "Hop Soul T-Shirt Size 3", "2,00", "22,49", "44,98", "49,48"),
            (530, "2.", "boys bogs waterproof boots", "youth size 3", "1,00", "9,80", "9,80", "10,78")]
    for y, no, desc, wrapped, qty, price, net, gross in rows:
        c.drawString(47, y, no)
        c.drawString(70, y, desc)
        c.drawString(70, y - 10, wrapped)  # description continues on a second line
        c.drawRightString(270, y, qty)
        c.drawString(280, y, "each")
        c.drawRightString(360, y, price)
        c.drawRightString(420, y, net)
        c.drawRightString(480, y, "10%")
        c.drawRightString(545, y, gross)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(45, 480, "SUMMARY")
    c.setFont("Helvetica", 9)
    for x, text in [(300, "VAT [%]"), (420, "Net worth"), (480, "VAT"), (545, "Gross worth")]:
        c.drawRightString(x, 460, text)
    for x, text in [(300, "10%"), (420, "54,78"), (480, "5,48"), (545, "60,26")]:
        c.drawRightString(x, 445, text)
    c.drawString(200, 430, "Total")
    for x, text in [(420, "$ 54,78"), (480, "$ 5,48"), (545, "$ 60,26")]:
        c.drawRightString(x, 430, text)


def test_summary_table_tax_ids_and_wrapped_items(parser):
    result = parser.parse_bytes("katanaml-style.pdf", _to_pdf(_summary_table_layout))
    f = result["fields"]
    assert f["invoice_number"]["value"] == "65321852"
    assert f["invoice_date"]["value"] == "2021-04-11"  # US invoice: month first
    assert result["date_order"] == "month-first"
    assert f["seller"]["value"]["name"] == "Kaufman, Cooper and Young"
    assert f["seller"]["value"]["tax_id"] == "930-79-7845"
    assert f["buyer"]["value"]["name"] == "Wells-Carlson"
    assert f["buyer"]["value"]["tax_id"] == "957-82-7504"
    assert f["subtotal"]["value"] == pytest.approx(54.78)
    assert f["tax"]["value"] == pytest.approx(5.48)
    assert f["total"]["value"] == pytest.approx(60.26)
    items = result["line_items"]
    assert len(items) == 2  # "youth size 3" is a wrapped description, not a third item
    assert items[0]["quantity"] == 2 and items[0]["unit_price"] == pytest.approx(22.49)
    assert items[0]["amount"] == pytest.approx(44.98) and "Size 3" in items[0]["description"]
    assert items[1]["description"].endswith("youth size 3")
    assert all(result["checks"].values())


def test_colon_labels_and_totals(parser):
    result = parser.parse_bytes("invoice.pdf", _to_pdf(_colon_layout))
    f = result["fields"]
    assert f["invoice_number"]["value"] == "INV-2024-0042"
    assert f["invoice_date"]["value"] == "2024-03-12"
    assert f["due_date"]["value"] == "2024-04-11"
    assert f["seller"]["value"]["name"] == "Acme Traders Pvt Ltd"
    assert f["seller"]["value"]["tax_id"] == "27ABCDE1234F1Z0"
    assert f["seller"]["value"]["address"] == "12 MG Road, Pune 411001"
    assert f["buyer"]["value"]["name"] == "Rahul Electronics"
    assert f["subtotal"]["value"] == pytest.approx(900.0)
    assert f["tax"]["value"] == pytest.approx(162.0)
    assert f["total"]["value"] == pytest.approx(1062.0)
    assert f["currency"]["value"] == "USD"
    assert len(result["line_items"]) == 2
    assert result["line_items"][0]["quantity"] == 2
    assert result["line_items"][0]["amount"] == pytest.approx(800.0)
    assert result["checks"]["subtotal_plus_tax_minus_discount_equals_total"] is True
    assert result["checks"]["line_items_sum_equals_subtotal"] is True
    assert result["warnings"] == []


def test_label_only_two_column_layout(parser):
    result = parser.parse_bytes("labels.pdf", _to_pdf(_label_only_layout))
    f = result["fields"]
    assert f["seller"]["value"]["name"] == "Bluepeak Solutions LLP"
    assert f["buyer"]["value"]["name"] == "Metro Retail Store"
    assert f["invoice_number"]["value"] == "INV-5521"
    assert f["invoice_date"]["value"] == "2024-07-07"
    assert f["tax"]["value"] == pytest.approx(81.0)
    assert f["total"]["value"] == pytest.approx(531.0)
    assert f["currency"]["value"] == "GBP"
    assert result["checks"]["subtotal_plus_tax_minus_discount_equals_total"] is True


def _tally_layout(c, sgst="22,500.00"):
    """Tally ERP 'Tax Invoice': boxed labels with the value BELOW them, Indian number format, CGST + SGST rows,
    amount in words and the seller named only in the 'for X / Authorised Signatory' block and the header."""
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(300, 810, "Tax Invoice")
    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, 785, "Kavya Infotech")
    c.setFont("Helvetica", 9)
    c.drawString(40, 772, "Plot 12, Sector 5, Gurugram")
    c.drawString(40, 759, "GSTIN/UIN: 06BKXPT4822Q3ZY")
    c.drawString(40, 746, "State Name : Haryana, Code : 06")
    for x, y, text in [(320, 785, "Invoice No."), (450, 785, "Dated"),
                       (320, 771, "KVIN/2026-27/007"), (450, 771, "23-Sep-26"),
                       (320, 752, "Delivery Note"), (450, 752, "Mode/Terms of Payment"),
                       (320, 733, "Buyer's Order No."), (450, 733, "Dated"),
                       (320, 719, "4500012345"), (450, 719, "15-Sep-26")]:
        c.drawString(x, y, text)
    c.drawString(40, 715, "Buyer (Bill to)")
    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, 702, "Shakti Steel Tubes Limited")
    c.setFont("Helvetica", 9)
    c.drawString(40, 689, "Industrial Area, Phase 2, Panipat")
    c.drawString(40, 676, "GSTIN/UIN : 06AABCS7788K1ZP")
    for x, h in [(40, "Sl No."), (80, "Description of Goods"), (300, "HSN/SAC"), (370, "Quantity"),
                 (430, "Rate"), (470, "per"), (510, "Amount")]:
        c.drawString(x, 640, h)
    c.drawString(40, 624, "1")
    c.drawString(80, 624, "Software Development Services")
    c.drawString(300, 624, "998314")
    c.drawString(370, 624, "1 Nos")
    c.drawString(430, 624, "2,50,000.00")
    c.drawString(470, 624, "Nos")
    c.drawRightString(565, 624, "2,50,000.00")
    c.drawRightString(300, 600, "CGST @ 9%")
    c.drawRightString(565, 600, "22,500.00")
    c.drawRightString(300, 586, "SGST @ 9%")
    c.drawRightString(565, 586, sgst)
    c.drawRightString(300, 560, "Total")
    c.drawRightString(565, 560, "2,95,000.00")
    c.drawString(40, 535, "Amount Chargeable (in words)")
    c.drawString(40, 522, "INR Two Lakh Ninety Five Thousand Only")
    c.drawRightString(565, 480, "for Kavya Infotech")
    c.drawRightString(565, 450, "Authorised Signatory")


def test_tally_gst_invoice(parser):
    result = parser.parse_bytes("tally.pdf", _to_pdf(_tally_layout))
    f = result["fields"]
    assert f["invoice_number"]["value"] == "KVIN/2026-27/007"
    assert f["invoice_date"]["value"] == "2026-09-23"
    assert f["po_number"]["value"] == "4500012345"
    assert f["seller"]["value"]["name"] == "Kavya Infotech"
    assert f["seller"]["value"]["tax_id"] == "06BKXPT4822Q3ZY"
    assert f["buyer"]["value"]["name"] == "Shakti Steel Tubes Limited"
    assert f["buyer"]["value"]["tax_id"] == "06AABCS7788K1ZP"
    assert f["tax"]["value"] == pytest.approx(45000)
    assert f["total"]["value"] == pytest.approx(295000)
    assert f["currency"]["value"] == "INR"
    assert len(result["line_items"]) == 1 and result["line_items"][0]["amount"] == pytest.approx(250000)
    checks = result["checks"]
    assert checks["seller_gstin_valid"] and checks["buyer_gstin_valid"] and checks["cgst_equals_sgst"]
    assert checks["gst_type_matches_states"] and checks["total_matches_amount_in_words"]


def _superstore_layout(c):
    """'SuperStore' style (invoice-generator template): the seller is only a name at the top, '# 5873' under
    INVOICE, Bill To / Ship To side by side with 'Ship Mode' on the right at the same height, white text on a
    dark item-table header, and Subtotal / Discount / Shipping / Total."""
    Y = lambda top: 842 - top - 8
    c.setFont("Helvetica", 27.8)
    c.drawRightString(575, 842 - 51, "INVOICE")
    c.setFont("Helvetica", 12)
    c.drawRightString(571, Y(60), "# 5873")
    c.setFont("Helvetica-Bold", 9.8)
    c.drawString(48, Y(26.6), "SuperStore")
    c.setFont("Helvetica", 9.8)
    c.drawString(48, Y(124), "Bill To:")
    c.drawString(182, Y(124), "Ship To:")
    c.setFont("Helvetica-Bold", 9.8)
    c.drawString(48, Y(140), "Alan Haines")
    c.drawString(182, Y(140), "San Salvador, San")
    c.drawString(182, Y(152), "Salvador, El Salvador")
    c.setFont("Helvetica", 9.8)
    c.drawRightString(462, Y(110.6), "Date:")
    c.drawRightString(567, Y(110.6), "Jan 17 2012")
    c.drawRightString(462, Y(132.4), "Ship Mode:")
    c.drawRightString(567, Y(132.4), "Standard Class")
    c.setFont("Helvetica-Bold", 11.2)
    c.drawRightString(462, Y(155.3), "Balance Due:")
    c.drawRightString(567, Y(155.3), "$15,655.53")
    c.setFillGray(0.2)
    c.rect(36, 842 - 228, 540, 18, fill=1, stroke=0)
    c.setFillGray(1)
    c.setFont("Helvetica", 9.8)
    c.drawString(44, Y(214), "Item")
    c.drawString(373, Y(214), "Quantity")
    c.drawRightString(493, Y(214), "Rate")
    c.drawRightString(567, Y(214), "Amount")
    c.setFillGray(0)
    c.setFont("Helvetica-Bold", 9.8)
    c.drawString(44, Y(239.6), "Hewlett Fax and Copier, Laser")
    c.setFont("Helvetica", 9.8)
    c.drawString(373, Y(239.6), "11")
    c.drawRightString(492, Y(239.6), "$1,411.55")
    c.drawRightString(567, Y(239.6), "$15,527.06")
    c.drawString(44, Y(255.4), "Copiers, Technology, TEC-CO-4574")
    for top, label, value in [(319, "Subtotal:", "$15,527.06"), (341, "Discount (0.2%):", "$31.05"),
                              (362.6, "Shipping:", "$159.52"), (384.4, "Total:", "$15,655.53")]:
        c.drawRightString(462, Y(top), label)
        c.drawRightString(567, Y(top), value)
    c.drawString(48, Y(439), "Notes:")
    c.drawString(48, Y(455.6), "Thanks for your business!")
    c.drawString(48, Y(483.4), "Terms:")
    c.drawString(48, Y(500), "Order ID : MX-2012-AH1019539-40925")


def test_superstore_layout(parser):
    result = parser.parse_bytes("invoice_Alan Haines_5873.pdf", _to_pdf(_superstore_layout))
    f = result["fields"]
    assert f["invoice_number"]["value"] == "5873"
    assert f["invoice_date"]["value"] == "2012-01-17"
    assert f["po_number"]["value"] == "MX-2012-AH1019539-40925"
    assert f["seller"]["value"]["name"] == "SuperStore"
    assert f["buyer"]["value"]["name"] == "Alan Haines"  # not "Ship Mode", not the Ship-To address
    assert f["subtotal"]["value"] == pytest.approx(15527.06)
    assert f["discount"]["value"] == pytest.approx(31.05)
    assert f["shipping"]["value"] == pytest.approx(159.52)
    assert f["total"]["value"] == pytest.approx(15655.53)
    assert f["currency"]["value"] == "USD"
    assert len(result["line_items"]) == 1 and result["line_items"][0]["quantity"] == 11
    assert all(result["checks"].values())  # subtotal - discount + shipping = total


def test_superstore_layout_as_a_scan(parser):
    """Same invoice as an image: the dark header bar must be readable and the letterhead found."""
    import fitz
    page = fitz.open(stream=_to_pdf(_superstore_layout), filetype="pdf")[0]
    png = page.get_pixmap(dpi=200).tobytes("png")
    result = parser.parse_bytes("superstore.png", png)
    f = result["fields"]
    assert f["invoice_number"]["value"] == "5873"
    assert f["seller"]["value"]["name"].lower() == "superstore"
    assert f["buyer"]["value"]["name"] == "Alan Haines"
    assert f["total"]["value"] == pytest.approx(15655.53)
    assert f["shipping"]["value"] == pytest.approx(159.52)
    assert len(result["line_items"]) == 1


def test_ocr_digit_error_in_sgst_is_corrected_by_the_invoice_arithmetic(parser):
    """Phone photo of the Tally invoice: SGST was read as 22,600 instead of 22,500. CGST = SGST and
    total (confirmed by the amount in words) - taxable value = 45,000 prove the right value."""
    result = parser.parse_bytes("tally-misread.pdf", _to_pdf(lambda c: _tally_layout(c, sgst="22,600.00")))
    f = result["fields"]
    assert f["tax"]["components"] == {"CGST": 22500.0, "SGST": 22500.0}
    assert f["tax"]["value"] == pytest.approx(45000)
    assert f["total"]["value"] == pytest.approx(295000)
    assert all(result["checks"].values())
    assert any("SGST was read as 22,600.00 and corrected to 22,500.00" in w for w in result["warnings"])


def test_a_real_mismatch_is_not_hidden(parser):
    """When nothing proves the right value (two digits differ), the amounts are left as read and the
    check fails, so a person looks at it."""
    result = parser.parse_bytes("tally-wrong.pdf", _to_pdf(lambda c: _tally_layout(c, sgst="27,900.00")))
    assert result["fields"]["tax"]["components"]["SGST"] == pytest.approx(27900)
    assert result["checks"]["cgst_equals_sgst"] is False


def test_quantity_and_rate_from_a_merged_cell():
    from invoice_parser.extract import _qty_rate_from_text
    assert _qty_rate_from_text("1| SR-SWT-XX-91407920 998311 1 NOS 2,50,000.00 al 2,50,000.00", 250000, None, None) == (1, 250000)
    assert _qty_rate_from_text("Widget 2 150.00 300.00", 300, None, None) == (2, 150)
    assert _qty_rate_from_text("Widget 7 999.00", 300, None, None) == (None, None)  # nothing fits: unchanged


def _ocr_page(rows):
    """Pages as OCR would give them: (text, x0, y0, x1, y1) in 0..1 page coordinates."""
    from invoice_parser.models import Page, Word

    return [Page(0, [Word(t, x0, y0, x1, y1, 0.9, 0) for t, x0, y0, x1, y1 in rows], "ocr", 0.9)]


def _words(text, x, y, h=0.01, w=0.012):
    """One OCR word per space-separated token, left to right from x."""
    out = []
    for token in text.split():
        out.append((token, x, y, x + w * len(token), y + h))
        x += w * (len(token) + 1)
    return out


def test_total_row_split_by_a_tilted_photo():
    """Real phone photo of a Tally invoice: OCR returned the Total row as 'Total  1 NOS' and the amount
    2,95,000.00 as a line of its own (the right end of the row sits higher on a curved page). The
    total quantity must not be taken as the total; the line items were unreadable, so the subtotal
    comes from total - tax, which is printed on the invoice and confirmed by the amount in words."""
    from invoice_parser.extract import extract_invoice
    from invoice_parser.layout import build_lines

    rows = (_words("Tax Invoice", 0.40, 0.05)
            + _words("1 SR-SWT-XX-91407920", 0.03, 0.41) + _words("1 NOS 2,50,000.00 NOS", 0.47, 0.41)
            + _words("Output CGST 9%", 0.20, 0.45) + _words("22,500.00", 0.80, 0.45)
            + _words("Output SGST 9%", 0.20, 0.47) + _words("22,500.00", 0.80, 0.47)
            + _words("Total", 0.30, 0.614) + _words("1 NOS", 0.47, 0.614)
            + _words(">", 0.62, 0.600) + _words("2,95,000.00", 0.80, 0.601)
            + _words("Amount Chargeable (in words)", 0.04, 0.636)
            + _words("INR Two Lakh Ninety Five Thousand Only", 0.04, 0.648))
    result = extract_invoice(build_lines(_ocr_page(rows)), LabelMatcher(use_embeddings=False))
    f = result["fields"]
    assert f["total"]["value"] == pytest.approx(295000)
    assert f["tax"]["value"] == pytest.approx(45000)
    assert f["subtotal"]["value"] == pytest.approx(250000)
    assert result["checks"]["total_matches_amount_in_words"]


def test_a_quantity_with_its_unit_is_never_an_amount():
    from invoice_parser.extract import _QTY_UNIT_RE

    assert _QTY_UNIT_RE.sub(" ", "Total 1 NOS 2,95,000.00").split() == ["Total", "2,95,000.00"]
    assert _QTY_UNIT_RE.sub(" ", "5 pcs").strip() == ""
    assert _QTY_UNIT_RE.sub(" ", "1,234.00") == "1,234.00"  # plain amounts stay


def _tally_photo(turns):
    """The Tally invoice as a picture turned by `turns` quarter turns (np.rot90), the way a phone
    stores a photo of a page lying flat on a desk when it has no rotation tag."""
    import cv2
    import fitz
    import numpy as np

    page = fitz.open(stream=_to_pdf(_tally_layout), filetype="pdf")[0]
    pix = page.get_pixmap(dpi=200)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]
    return cv2.imencode(".png", np.ascontiguousarray(np.rot90(img, turns)))[1].tobytes()


@pytest.mark.parametrize("turns", [1, 2, 3])
def test_sideways_and_upside_down_pictures_are_turned_upright(parser, turns):
    result = parser.parse_bytes("tally-turned.png", _tally_photo(turns))
    f = result["fields"]
    assert f["invoice_number"]["value"] == "KVIN/2026-27/007"
    assert f["buyer"]["value"]["name"] == "Shakti Steel Tubes Limited"
    assert f["total"]["value"] == pytest.approx(295000)
