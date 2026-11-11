"""CSV and Excel exports of the invoice table.

Missing values are written as empty cells (a real NULL), not the text "null", so the files
sort, filter and sum correctly in Excel and import cleanly into other systems.
"""
import csv
import io
from datetime import date

DATE_COLUMNS = {"invoice_date", "due_date"}


def to_csv(rows, columns) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    for row in rows:
        writer.writerow(["" if row.get(c) is None else row.get(c) for c in columns])
    return buf.getvalue().encode("utf-8-sig")  # BOM: Excel then shows ₹, € and accents correctly


def build_xlsx(invoice_rows, invoice_columns, item_rows, item_columns) -> bytes:
    from openpyxl import Workbook

    wb = Workbook()
    _fill_sheet(wb.active, "Invoices", invoice_rows, invoice_columns)
    _fill_sheet(wb.create_sheet(), "Line items", item_rows, item_columns)
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


def _fill_sheet(ws, title, rows, columns):
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    ws.title = title
    ws.append(columns)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for row in rows:
        values = []
        for c in columns:
            v = row.get(c)
            if isinstance(v, str):
                v = ILLEGAL_CHARACTERS_RE.sub("", v)  # OCR text can contain control characters
                if c in DATE_COLUMNS:
                    try:
                        v = date.fromisoformat(v)  # real Excel dates, so they sort and filter
                    except ValueError:
                        pass
            values.append(v)
        ws.append(values)
    ws.freeze_panes = "A2"
    if rows:
        ws.auto_filter.ref = ws.dimensions
    for i, c in enumerate(columns, 1):
        ws.column_dimensions[get_column_letter(i)].width = min(40, max(10, len(c) + 4))
