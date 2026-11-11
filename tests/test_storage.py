import io
import time

import pytest
from openpyxl import load_workbook
from test_pipeline import _colon_layout, _label_only_layout, _to_pdf

from invoice_parser.exporting import build_xlsx, to_csv
from invoice_parser.matcher import LabelMatcher
from invoice_parser.pipeline import InvoiceParser
from invoice_parser.storage import EXPORT_COLUMNS, ITEM_COLUMNS, InvoiceStore, flatten
from invoice_parser.worker import BatchWorker


@pytest.fixture()
def store(tmp_path):
    return InvoiceStore(db_path=tmp_path / "test.db", files_dir=tmp_path / "files")


@pytest.fixture(scope="module")
def parser():
    return InvoiceParser(matcher=LabelMatcher(use_embeddings=False))


def _unique_pdf(layout, ref):
    """Same layout, different bytes (so files are not treated as duplicates)."""
    def draw(c):
        layout(c)
        c.drawString(40, 30, f"Ref {ref}")
    return _to_pdf(draw)


def _run_until_done(store, worker, batch_id, timeout=120):
    worker.start()
    deadline = time.time() + timeout
    while not store.batch_summary(batch_id)["complete"] and time.time() < deadline:
        time.sleep(0.2)
    worker.stop()


def test_duplicate_file_is_stored_once_but_listed_in_both_batches(store):
    data = _to_pdf(_colon_layout)
    b1, b2 = store.create_batch("first"), store.create_batch("second")
    id1, new1 = store.add_file("a.pdf", data, batch_id=b1)
    id2, new2 = store.add_file("copy-of-a.pdf", data, batch_id=b2)
    assert new1 and not new2 and id1 == id2
    assert store.batch_summary(b1)["total"] == store.batch_summary(b2)["total"] == 1


def test_missing_fields_become_null_columns():
    row = flatten({"fields": {"invoice_number": {"value": "X-1", "confidence": 1.0}},
                   "line_items": [], "overall_confidence": 0.3})
    assert row["invoice_number"] == "X-1"
    assert row["seller_name"] is None and row["total"] is None and row["due_date"] is None
    assert row["needs_review"] == 1


def test_parallel_workers_process_a_batch(store, parser):
    batch = store.create_batch("bulk")
    for i in range(6):
        store.add_file(f"inv{i}.pdf", _unique_pdf(_colon_layout if i % 2 else _label_only_layout, i), batch_id=batch)
    _run_until_done(store, BatchWorker(store, workers=3, parser=parser), batch)
    summary = store.batch_summary(batch)
    assert summary["done"] == 6 and summary["failed"] == 0
    _, rows = store.search(batch_id=batch)
    assert all(r["total"] is not None and r["invoice_number"] for r in rows)
    first = store.get(rows[0]["id"])
    assert first["line_items"] and first["line_items"][0]["amount"] is not None


def test_a_broken_file_does_not_stop_the_batch(store, parser):
    batch = store.create_batch("mixed")
    bad_id, _ = store.add_file("broken.pdf", b"this is not a pdf", batch_id=batch)
    store.add_file("good.pdf", _unique_pdf(_colon_layout, "good"), batch_id=batch)
    _run_until_done(store, BatchWorker(store, workers=2, parser=parser), batch)
    summary = store.batch_summary(batch)
    assert summary["done"] == 1 and summary["failed"] == 1
    assert store.get(bad_id)["error"]


def test_requeue_and_reprocess(store, parser):
    invoice_id, _ = store.add_file("x.pdf", _unique_pdf(_colon_layout, "rq"), queue=False)
    worker = BatchWorker(store, workers=1, parser=parser)
    worker.process_now(invoice_id)
    assert store.get(invoice_id)["status"] == "done"
    store.requeue(invoice_id)
    assert store.claim_next()["id"] == invoice_id


def test_csv_and_excel_exports(store, parser):
    invoice_id, _ = store.add_file("x.pdf", _unique_pdf(_colon_layout, "exp"), queue=False)
    BatchWorker(store, workers=1, parser=parser).process_now(invoice_id)
    rows, items = store.export_rows(), store.export_line_items()

    csv_bytes = to_csv(rows, EXPORT_COLUMNS)
    assert csv_bytes.startswith(b"\xef\xbb\xbf")  # BOM so Excel reads UTF-8

    wb = load_workbook(io.BytesIO(build_xlsx(rows, EXPORT_COLUMNS, items, ITEM_COLUMNS)))
    assert wb.sheetnames == ["Invoices", "Line items"]
    sheet = wb["Invoices"]
    header = [c.value for c in sheet[1]]
    row = {h: c.value for h, c in zip(header, sheet[2])}
    assert row["invoice_number"] == "INV-2024-0042"
    assert row["invoice_date"].year == 2024  # a real Excel date, not text
    assert row["po_number"] is None          # missing field -> empty cell
    assert wb["Line items"].max_row == 3     # header + 2 items


def test_duplicate_copies_are_not_counted_twice_in_totals(tmp_path):
    from invoice_parser.storage import InvoiceStore
    store = InvoiceStore(db_path=tmp_path / "t.db", files_dir=tmp_path / "files")
    result = {"fields": {"invoice_number": {"value": "INV-9"}, "total": {"value": 100.0},
                         "currency": {"value": "INR"}, "seller": {"value": {"name": "Acme"}}},
              "line_items": [], "checks": {}, "warnings": [], "overall_confidence": 0.9}
    for name in ("a.pdf", "b.jpg"):  # the same invoice as a PDF and as a photo
        invoice_id, _ = store.add_file(name, name.encode(), queue=False)
        store.save_result(invoice_id, result, 10)
    over = store.overview()
    assert over["duplicates"] == 1
    assert [t["total"] for t in over["totals"]] == [100.0]
    assert [v["total"] for v in store.vendors()] == [100.0]
