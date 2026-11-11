#!/usr/bin/env python
"""Create varied synthetic invoices with ground truth, for tests and accuracy checks.

Five layouts use different labels (Seller / Vendor / Sold By / From / Bill From ...), two-column
headers, label-only headings, date formats and amount formats (US, EU, Indian).

Usage (from the project folder, with the venv active):
    python scripts/make_sample_invoices.py                 # 4 PDFs per layout -> data/samples/
    python scripts/make_sample_invoices.py --count 8 --scans
"""
import argparse
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from invoice_parser.normalize import gstin_check_char  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "samples"

COMPANIES = [
    "Acme Traders Pvt Ltd", "Bluepeak Solutions LLP", "Sunrise Industries Pvt. Ltd.",
    "Nimbus Technologies Inc", "Global Parts Co.", "Kaveri Enterprises", "Orion Supplies GmbH",
    "Fresh Harvest Foods Ltd",
]
BUYERS = ["Rahul Electronics", "Metro Retail Store", "Delta Logistics Pvt Ltd",
          "Greenleaf Pharma", "Northwind Traders LLC", "City Hardware"]
ADDRESSES = ["12 MG Road, Pune 411001", "45 Park Street, Kolkata 700016",
             "221 Baker Street, London NW1", "Plot 7, Sector 18, Noida 201301",
             "900 Market St, San Francisco CA 94103"]
ITEMS = ["Laptop Stand", "USB-C Cable", "Office Chair", "A4 Paper Ream", "Wireless Mouse",
         "Printer Toner", "Monitor 24 inch", "Keyboard"]


def amount_text(value, style):
    if style == "eu":  # 1.234,50
        return f"{value:,.2f}".replace(",", "#").replace(".", ",").replace("#", ".")
    if style == "in":  # 1,23,456.00
        whole, frac = f"{value:.2f}".split(".")
        if len(whole) > 3:
            head, tail = whole[:-3], whole[-3:]
            parts = []
            while len(head) > 2:
                parts.insert(0, head[-2:])
                head = head[:-2]
            if head:
                parts.insert(0, head)
            whole = ",".join(parts + [tail])
        return f"{whole}.{frac}"
    return f"{value:,.2f}"  # 1,234.50


def random_date(rng):
    return date(2023, 1, 1) + timedelta(days=rng.randint(0, 900))


def make_items(rng):
    items = []
    for _ in range(rng.randint(2, 4)):
        qty = rng.randint(1, 9)
        rate = round(rng.uniform(40, 4000), 2)
        items.append({"description": rng.choice(ITEMS), "quantity": qty,
                      "rate": rate, "amount": round(qty * rate, 2)})
    return items


def compute_totals(items, tax_pct, discount_pct=0):
    sub = round(sum(i["amount"] for i in items), 2)
    disc = round(sub * discount_pct / 100, 2)
    tax = round((sub - disc) * tax_pct / 100, 2)
    total = round(sub - disc + tax, 2)
    return sub, disc, tax, total


def gstin(rng):
    """A GSTIN with a correct check character, like a real one."""
    first14 = f"{rng.randint(10, 35)}ABCDE{rng.randint(1000, 9999)}F1Z"
    return first14 + gstin_check_char(first14)


def draw_table(c, y, headers, xs, rows):
    c.setFont("Helvetica-Bold", 10)
    for x, h in zip(xs, headers):
        c.drawString(x, y, h)
    c.setFont("Helvetica", 10)
    for row in rows:
        y -= 16
        for x, cell in zip(xs, row):
            c.drawString(x, y, cell)
    return y


def truth(file_hint, currency, inv_no, inv_date, due, seller, buyer, items, sub, disc, tax, total):
    return {
        "invoice_number": inv_no,
        "invoice_date": inv_date.isoformat(),
        "due_date": due.isoformat() if due else None,
        "seller": seller,
        "buyer": buyer,
        "currency": currency,
        "subtotal": sub,
        "discount": disc,
        "tax": tax,
        "total": total,
        "line_items": [{"description": i["description"], "quantity": i["quantity"],
                        "unit_price": i["rate"], "amount": i["amount"]} for i in items],
    }


# --- Layouts -------------------------------------------------------------------

def layout_classic(c, rng):
    """'Seller:' / 'Bill To:' with values on the same line; invoice details on the right."""
    seller, buyer = rng.choice(COMPANIES), rng.choice(BUYERS)
    inv_no = f"INV-{rng.randint(2024, 2025)}-{rng.randint(1, 9999):04d}"
    inv_date = random_date(rng)
    due = inv_date + timedelta(days=30)
    items = make_items(rng)
    tax_pct = rng.choice([8, 10, 18])
    sub, _, tax, total = compute_totals(items, tax_pct)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, 790, "TAX INVOICE")
    c.setFont("Helvetica", 10)
    c.drawString(40, 760, f"Seller: {seller}")
    c.drawString(40, 747, rng.choice(ADDRESSES))
    c.drawString(40, 734, f"GSTIN: {gstin(rng)}")
    c.drawString(330, 760, f"Invoice No: {inv_no}")
    c.drawString(330, 747, f"Invoice Date: {inv_date.strftime('%d/%m/%Y')}")
    c.drawString(330, 734, f"Due Date: {due.strftime('%d/%m/%Y')}")
    c.drawString(40, 690, f"Bill To: {buyer}")
    c.drawString(40, 677, rng.choice(ADDRESSES))
    rows = [[i["description"], str(i["quantity"]), amount_text(i["rate"], "us"),
             amount_text(i["amount"], "us")] for i in items]
    y = draw_table(c, 630, ["Description", "Qty", "Rate", "Amount"], [40, 340, 410, 490], rows) - 30
    c.drawString(380, y, "Subtotal:")
    c.drawRightString(540, y, f"$ {amount_text(sub, 'us')}")
    c.drawString(380, y - 14, f"Tax ({tax_pct}%):")
    c.drawRightString(540, y - 14, f"$ {amount_text(tax, 'us')}")
    c.drawString(380, y - 28, "Grand Total:")
    c.drawRightString(540, y - 28, f"$ {amount_text(total, 'us')}")
    return truth("classic", "USD", inv_no, inv_date, due, seller, buyer, items, sub, None, tax, total)


def layout_vendor(c, rng):
    """Company name at the top with no label; 'Customer:' and 'Invoice #:' labels."""
    seller, buyer = rng.choice(COMPANIES), rng.choice(BUYERS)
    inv_no = f"INV-{rng.randint(100000, 999999)}"
    inv_date = random_date(rng)
    items = make_items(rng)
    sub, _, tax, total = compute_totals(items, rng.choice([5, 12, 18]))

    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(297, 800, seller)
    c.setFont("Helvetica", 10)
    c.drawCentredString(297, 786, rng.choice(ADDRESSES))
    c.drawString(40, 740, f"Customer: {buyer}")
    c.drawString(40, 727, rng.choice(ADDRESSES))
    c.drawString(330, 740, f"Invoice #: {inv_no}")
    c.drawString(330, 727, f"Date: {inv_date.isoformat()}")
    rows = [[i["description"], str(i["quantity"]), amount_text(i["rate"], "us"),
             amount_text(i["amount"], "us")] for i in items]
    y = draw_table(c, 660, ["Item", "Quantity", "Unit Price", "Total"], [40, 300, 390, 490], rows) - 30
    c.drawString(380, y, "Subtotal:")
    c.drawRightString(540, y, f"$ {amount_text(sub, 'us')}")
    c.drawString(380, y - 14, "Tax:")
    c.drawRightString(540, y - 14, f"$ {amount_text(tax, 'us')}")
    c.drawString(380, y - 28, "Amount Due:")
    c.drawRightString(540, y - 28, f"$ {amount_text(total, 'us')}")
    return truth("vendor", "USD", inv_no, inv_date, None, seller, buyer, items, sub, None, tax, total)


def layout_eu(c, rng):
    """'From:' / 'To:' blocks, EU number format (1.234,50 EUR) and '3 Mar 2024' dates."""
    seller, buyer = rng.choice(COMPANIES), rng.choice(BUYERS)
    bill_no = f"{rng.randint(2024, 2025)}/{rng.randint(1, 999):04d}"
    inv_date = random_date(rng)
    due = inv_date + timedelta(days=21)
    items = make_items(rng)
    sub, _, tax, total = compute_totals(items, 19)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, 800, "INVOICE")
    c.setFont("Helvetica", 10)
    c.drawString(40, 765, f"From: {seller}")
    c.drawString(40, 752, rng.choice(ADDRESSES))
    c.drawString(40, 700, f"To: {buyer}")
    c.drawString(40, 687, rng.choice(ADDRESSES))
    c.drawString(330, 765, f"Bill No.: {bill_no}")
    c.drawString(330, 752, f"Date of issue: {inv_date.day} {inv_date.strftime('%b %Y')}")
    c.drawString(330, 739, f"Payment due: {due.isoformat()}")
    rows = [[i["description"], str(i["quantity"]), amount_text(i["rate"], "eu"),
             amount_text(i["amount"], "eu")] for i in items]
    y = draw_table(c, 630, ["Description", "Qty", "Unit price", "Amount"], [40, 340, 410, 490], rows) - 30
    c.drawString(380, y, "Net Amount:")
    c.drawRightString(540, y, f"{amount_text(sub, 'eu')} EUR")
    c.drawString(380, y - 14, "VAT 19%:")
    c.drawRightString(540, y - 14, f"{amount_text(tax, 'eu')} EUR")
    c.drawString(380, y - 28, "Total Amount:")
    c.drawRightString(540, y - 28, f"{amount_text(total, 'eu')} EUR")
    return truth("eu", "EUR", bill_no, inv_date, due, seller, buyer, items, sub, None, tax, total)


def layout_india(c, rng):
    """GST invoice: 'Sold By' / 'Bill To' with GSTIN, CGST + SGST lines, INR amounts."""
    seller, buyer = rng.choice(COMPANIES), rng.choice(BUYERS)
    inv_no = f"GST/24-25/{rng.randint(1, 999):04d}"
    inv_date = random_date(rng)
    items = make_items(rng)
    sub, _, _, _ = compute_totals(items, 18)
    half = round(sub * 9 / 100, 2)
    tax, total = round(2 * half, 2), round(sub + 2 * half, 2)

    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(297, 800, "TAX INVOICE")
    c.setFont("Helvetica", 10)
    c.drawString(40, 770, f"Sold By: {seller}")
    c.drawString(40, 757, rng.choice(ADDRESSES))
    c.drawString(40, 744, f"GSTIN: {gstin(rng)}")
    c.drawString(330, 770, f"Invoice No: {inv_no}")
    c.drawString(330, 757, f"Invoice Date: {inv_date.strftime('%d-%m-%Y')}")
    c.drawString(40, 700, f"Bill To: {buyer}")
    c.drawString(40, 687, rng.choice(ADDRESSES))
    c.drawString(40, 674, f"GSTIN: {gstin(rng)}")
    rows = []
    for n, i in enumerate(items, 1):
        rows.append([str(n), i["description"], str(rng.randint(8000, 9999)), str(i["quantity"]),
                     amount_text(i["rate"], "in"), amount_text(i["amount"], "in")])
    y = draw_table(c, 630, ["Sr", "Description", "HSN", "Qty", "Rate", "Amount"],
                   [40, 70, 300, 370, 420, 490], rows) - 30
    c.drawString(330, y, "Taxable Value:")
    c.drawRightString(540, y, f"INR {amount_text(sub, 'in')}")
    c.drawString(330, y - 14, "CGST 9%:")
    c.drawRightString(540, y - 14, f"INR {amount_text(half, 'in')}")
    c.drawString(330, y - 28, "SGST 9%:")
    c.drawRightString(540, y - 28, f"INR {amount_text(half, 'in')}")
    c.drawString(330, y - 42, "Total Amount:")
    c.drawRightString(540, y - 42, f"INR {amount_text(total, 'in')}")
    return truth("gst", "INR", inv_no, inv_date, None, seller, buyer, items, sub, None, tax, total)


def layout_labels(c, rng):
    """Label-only headings ('BILL FROM', 'BILL TO', 'Subtotal' with no colon), GBP symbols."""
    seller, buyer = rng.choice(COMPANIES), rng.choice(BUYERS)
    inv_no = f"INV-{rng.randint(1000, 9999)}"
    inv_date = random_date(rng)
    items = make_items(rng)
    sub, _, tax, total = compute_totals(items, 18)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, 800, "INVOICE")
    c.setFont("Helvetica-Bold", 9)
    c.drawString(40, 765, "BILL FROM")
    c.drawString(330, 765, "BILL TO")
    c.setFont("Helvetica", 10)
    c.drawString(40, 751, seller)
    c.drawString(330, 751, buyer)
    c.drawString(40, 738, rng.choice(ADDRESSES))
    c.drawString(330, 738, rng.choice(ADDRESSES))
    c.drawString(330, 790, "Invoice Number")
    c.drawString(440, 790, inv_no)
    c.drawString(330, 776, "Date")
    c.drawString(440, 776, inv_date.strftime("%d/%m/%Y"))
    rows = [[i["description"], str(i["quantity"]), amount_text(i["rate"], "us"),
             amount_text(i["amount"], "us")] for i in items]
    y = draw_table(c, 700, ["Description", "Qty", "Rate", "Amount"], [40, 340, 410, 490], rows) - 30
    c.drawString(380, y, "Subtotal")
    c.drawRightString(540, y, f"GBP {amount_text(sub, 'us')}")
    c.drawString(380, y - 14, "Tax 18%")
    c.drawRightString(540, y - 14, f"GBP {amount_text(tax, 'us')}")
    c.drawString(380, y - 28, "Total")
    c.drawRightString(540, y - 28, f"GBP {amount_text(total, 'us')}")
    return truth("labels", "GBP", inv_no, inv_date, None, seller, buyer, items, sub, None, tax, total)


LAYOUTS = {
    "classic": layout_classic,
    "vendor": layout_vendor,
    "eu_from_to": layout_eu,
    "gst_india": layout_india,
    "label_only": layout_labels,
}


def make_scan(pdf_path: Path, out_path: Path, seed: int):
    """Simulate a photocopy: 200 DPI render, slight rotation, blur and sensor noise."""
    from io import BytesIO

    import fitz
    import numpy as np
    from PIL import Image, ImageFilter

    rng = random.Random(seed)
    doc = fitz.open(pdf_path)
    pix = doc[0].get_pixmap(dpi=200, alpha=False)
    doc.close()
    img = Image.open(BytesIO(pix.tobytes("png"))).convert("L")
    img = img.rotate(rng.uniform(-1.5, 1.5), resample=Image.BICUBIC, expand=True, fillcolor=255)
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    arr = np.asarray(img).astype(np.float32)
    arr += np.random.default_rng(seed).normal(0, 8, arr.shape)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(out_path)


def make_scanned_pdf(png_path: Path, out_path: Path):
    """An image-only PDF (no text layer), like the output of an office scanner."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, filename=str(png_path))
    doc.save(out_path)
    doc.close()


def make_photo(pdf_path: Path, out_path: Path, seed: int):
    """Simulate a phone photo: page lying on a table, perspective tilt, uneven light, JPEG."""
    import cv2
    import fitz
    import numpy as np

    rng = random.Random(seed)
    doc = fitz.open(pdf_path)
    pix = doc[0].get_pixmap(dpi=200, alpha=False)
    doc.close()
    page = cv2.cvtColor(np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, 3), cv2.COLOR_RGB2BGR)
    h, w = page.shape[:2]
    H, W = int(h * 1.25), int(w * 1.3)
    ox, oy = (W - w) // 2, (H - h) // 2

    def jitter():
        return rng.uniform(-0.05, 0.05)
    corners = np.float32([[ox + jitter() * w, oy + jitter() * h], [ox + w + jitter() * w, oy + jitter() * h],
                          [ox + w + jitter() * w, oy + h + jitter() * h], [ox + jitter() * w, oy + h + jitter() * h]])
    matrix = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), corners)
    warped = cv2.warpPerspective(page, matrix, (W, H))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), matrix, (W, H))
    table = np.zeros((H, W, 3), np.uint8)
    table[:] = (70, 90, 110)
    photo = np.where(mask[..., None] > 0, warped, table).astype(np.float32)
    light = np.linspace(0.6, 1.0, W, dtype=np.float32)
    if rng.random() < 0.5:
        light = light[::-1]
    photo = photo * light[None, :, None] + np.random.default_rng(seed).normal(0, 6, photo.shape)
    photo = cv2.GaussianBlur(np.clip(photo, 0, 255).astype(np.uint8), (3, 3), 0.8)
    cv2.imwrite(str(out_path), photo, [cv2.IMWRITE_JPEG_QUALITY, 75])


def _write_truth(gt, file_path: Path, **extra):
    data = dict(gt, file=file_path.name, **extra)
    file_path.with_name(file_path.stem + ".json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=4, help="invoices per layout")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scans", action="store_true",
                        help="also write noisy PNG scans and image-only scanned PDFs")
    parser.add_argument("--photos", action="store_true",
                        help="also write simulated phone photos (tilted page, uneven light)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for name, layout in LAYOUTS.items():
        for n in range(1, args.count + 1):
            stem = f"{name}_{n:02d}"
            pdf_path = OUT_DIR / f"{stem}.pdf"
            c = canvas.Canvas(str(pdf_path), pagesize=A4)
            gt = layout(c, rng)
            c.save()
            gt["template"] = name
            _write_truth(gt, pdf_path, kind="digital pdf")
            written += 1
            if args.scans:
                png_path = OUT_DIR / f"{stem}_scan.png"
                make_scan(pdf_path, png_path, seed=args.seed * 1000 + written)
                _write_truth(gt, png_path, kind="scan image")
                scan_pdf = OUT_DIR / f"{stem}_scanpdf.pdf"
                make_scanned_pdf(png_path, scan_pdf)
                _write_truth(gt, scan_pdf, kind="scanned pdf")
                written += 2
            if args.photos:
                photo_path = OUT_DIR / f"{stem}_photo.jpg"
                make_photo(pdf_path, photo_path, seed=args.seed * 1000 + written)
                _write_truth(gt, photo_path, kind="phone photo")
                written += 1
    print(f"Wrote {written} files to {OUT_DIR}")


if __name__ == "__main__":
    main()
