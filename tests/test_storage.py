import io
import os
import time
import uuid

import pytest
from openpyxl import load_workbook
from test_pipeline import _colon_layout, _label_only_layout, _to_pdf

from invoice_parser.exporting import build_xlsx, to_csv
from invoice_parser.matcher import LabelMatcher
from invoice_parser.pipeline import InvoiceParser
from invoice_parser.storage import EXPORT_COLUMNS, ITEM_COLUMNS, InvoiceStore, flatten
from invoice_parser.worker import BatchWorker

# Every storage test runs on SQLite, and on MongoDB too when TEST_MONGODB_URI names a throw-away server.
MONGO_URI = os.environ.get("TEST_MONGODB_URI", "")
BACKENDS = ["sqlite"] + (["mongo"] if MONGO_URI else [])
needs_mongo = pytest.mark.skipif(not MONGO_URI, reason="TEST_MONGODB_URI not set")


def _open(backend, tmp_path, **kwargs):
    if backend == "mongo":
        from invoice_parser.mongo_store import MongoInvoiceStore

        return MongoInvoiceStore(uri=MONGO_URI, db_name=f"invoagent_t_{uuid.uuid4().hex[:10]}",
                                 files_dir=tmp_path / "files", **kwargs)
    return InvoiceStore(db_path=tmp_path / "test.db", files_dir=tmp_path / "files")


def _close(store):
    if hasattr(store, "client"):
        store.wait_for_uploads()
        store.client.drop_database(store.db.name)


@pytest.fixture(params=BACKENDS)
def store(request, tmp_path):
    s = _open(request.param, tmp_path)
    yield s
    _close(s)


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


def test_duplicate_copies_are_not_counted_twice_in_totals(store):
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


# --- MongoDB store: same answers as SQLite, files kept small, space limit -------------------------
def _result(number, total, currency="INR", date="2026-05-10", seller="Acme", gstin=None, tax=None, items=1):
    return {"fields": {"invoice_number": {"value": number}, "total": {"value": total},
                       "tax": {"value": tax} if tax is not None else None, "currency": {"value": currency},
                       "invoice_date": {"value": date} if date else None,
                       "seller": {"value": {"name": seller, "tax_id": gstin}}},
            "line_items": [{"description": f"item {i}", "quantity": 1.0, "unit_price": 10.0, "amount": 10.0}
                           for i in range(items)],
            "checks": {}, "warnings": [], "overall_confidence": 0.9}


def _fill(store):
    """The same history in any store: invoices in two currencies and two years, a duplicate,
    approvals, a rejection, a correction, a failure and a deletion."""
    batch = store.create_batch("upload 1")
    rows = [("A-1", 1180.0, "INR", "2026-05-10", "Acme", "06AAAAA0000A1Z5", 180.0),
            ("A-2", 590.0, "INR", "2026-06-02", "Acme", "06AAAAA0000A1Z5", 90.0),
            ("A-1", 1180.0, "INR", "2026-05-10", "Acme", "06AAAAA0000A1Z5", 180.0),  # duplicate of #1
            ("B-7", 100.0, "USD", "2025-12-01", "Beta LLC", None, None),
            ("C-3", None, "INR", None, "  gamma traders ", None, None),
            ("D-9", 2000.0, "INR", "2025-04-15", "Delta", "07BBBBB1111B1Z2", 305.08),
            ("E-1", 50.0, "EUR", "2026-01-20", "Epsilon", None, 0.0)]
    ids = []
    for n, (number, total, cur, date, seller, gstin, tax) in enumerate(rows):
        invoice_id, _ = store.add_file(f"f{n}.pdf", f"file {n}".encode(), batch_id=batch, queue=False)
        store.save_result(invoice_id, _result(number, total, cur, date, seller, gstin, tax, items=n % 3), 10 + n)
        ids.append(invoice_id)
    failed, _ = store.add_file("bad.pdf", b"bad", batch_id=batch, queue=False)
    store.save_error(failed, "ValueError: broken")
    store.add_file("again.pdf", b"file 0", batch_id=store.create_batch("upload 2"))  # same bytes: not new
    store.set_review(ids[1], "approved", "ok")
    store.set_review(ids[3], "rejected")
    store.update_fields(ids[5], {"seller_name": "Delta Pvt Ltd", "total": "2,100.00"})
    store.delete([ids[6]])
    store.set_company([ids[0], ids[1], ids[5]], "Kavya Infotech")
    store.set_company([ids[3]], "OM TRADERS", "sale")
    store.add_file("again2.pdf", b"file 4", company="OM TRADERS")  # known file without a company: moved
    return ids, batch


_TIMES = {"created_at", "processed_at", "reviewed_at", "started_at"}


def _same(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: _same(v) for k, v in value.items() if k not in _TIMES}
    if isinstance(value, (list, tuple)):
        return [_same(v) for v in value]
    return value


def _without_item_ids(row):
    """SQLite numbers every line item row; MongoDB keeps the items inside the invoice (no own id)."""
    if row is None:
        return None
    return {**row, "line_items": [{k: v for k, v in it.items() if k != "id"} for it in row["line_items"]]}


@needs_mongo
def test_mongodb_store_answers_like_the_sqlite_store(tmp_path):
    from datetime import date

    sqlite, mongo = _open("sqlite", tmp_path / "s"), _open("mongo", tmp_path / "m")
    try:
        (ids, batch), (mids, mbatch) = _fill(sqlite), _fill(mongo)
        assert ids == mids and batch == mbatch
        today = date(2026, 6, 15)
        checks = [lambda s: s.stats(), lambda s: s.overview(), lambda s: s.vendors(),
                  lambda s: s.batch_summary(batch), lambda s: s.list_batches(),
                  lambda s: s.export_rows(), lambda s: s.export_line_items(),
                  lambda s: s.search(), lambda s: s.search(q="acme"), lambda s: s.search(q="gamma"),
                  lambda s: s.search(review=True), lambda s: s.search(status="failed"),
                  lambda s: s.search(review_status="pending"), lambda s: s.search(review_status="approved"),
                  lambda s: s.search(batch_id=batch, limit=3, offset=2),
                  lambda s: s.search(company="Kavya Infotech"), lambda s: s.search(company="-"),
                  lambda s: s.export_rows(company="OM TRADERS"), lambda s: s.money_rows(),
                  lambda s: s.money_rows(company="Kavya Infotech"), lambda s: s.company_names(),
                  lambda s: s.company_counts(), lambda s: s.files_of("Kavya Infotech"), lambda s: s.files_of("-"),
                  lambda s: [_without_item_ids(s.get(i)) for i in ids]]
        checks += [lambda s, p=p, c=c, co=co: s.dashboard(currency=c, period=p, today=today, company=co)
                   for p in ("12m", "fy", "lfy", "all") for c in (None, "INR", "USD", "XYZ")
                   for co in (None, "Kavya Infotech", "-")]
        for n, check in enumerate(checks):
            a, b = _same(check(sqlite)), _same(check(mongo))
            if n == 2:  # vendors: rows with the same total may come in either order
                a, b = sorted(a, key=repr), sorted(b, key=repr)
            assert a == b, f"check {n}"
    finally:
        _close(mongo)


def _photo(width, height):
    from PIL import Image

    exif = Image.Exif()
    exif[271], exif[272], exif[274] = "vivo", "vivo Y100A", 6
    buf = io.BytesIO()
    Image.radial_gradient("L").resize((width, height)).convert("RGB").save(buf, "JPEG", quality=95, exif=exif)
    return buf.getvalue()


@needs_mongo
def test_mongodb_keeps_a_smaller_copy_of_a_photo_and_fetches_it_back(tmp_path):
    from PIL import Image

    store = _open("mongo", tmp_path)
    try:
        data = _photo(4000, 3000)
        invoice_id, _ = store.add_file("IMG_1.jpg", data, queue=False)
        store.save_result(invoice_id, _result("P-1", 10.0), 5)
        store.wait_for_uploads()
        row = store.get(invoice_id)
        local = store.file_path(row)
        assert local.read_bytes() == data  # this computer keeps the full original
        local.unlink()                     # another computer: fetched from MongoDB
        copy = store.file_path(row).read_bytes()
        assert len(copy) < len(data)
        with Image.open(io.BytesIO(copy)) as im:
            assert max(im.size) == 2600
            assert im.getexif()[271] == "vivo" and im.getexif()[274] == 6  # camera details and turn kept
        pdf = _to_pdf(_colon_layout)
        pdf_id, _ = store.add_file("x.pdf", pdf, queue=False)
        store.save_error(pdf_id, "test")
        store.wait_for_uploads()
        pdf_row = store.get(pdf_id)
        store.file_path(pdf_row).unlink()
        assert store.file_path(pdf_row).read_bytes() == pdf  # PDFs are kept as they are
        store.delete([invoice_id])
        assert not store.file_path(row).exists() and store.db["files.files"].count_documents({}) == 1
    finally:
        _close(store)


@needs_mongo
def test_mongodb_stops_storing_files_before_the_cluster_is_full(tmp_path):
    store = _open("mongo", tmp_path, files_limit_mb=1)
    try:
        store._usage = (time.time(), 2**20)  # the cluster already holds 1 MB
        invoice_id, _ = store.add_file("IMG_2.jpg", _photo(800, 600), queue=False)
        store.save_result(invoice_id, _result("P-2", 20.0), 5)
        store.wait_for_uploads()
        assert store.db["files.files"].count_documents({}) == 0
        assert store.get(invoice_id)["total"] == 20.0  # the invoice data is still saved
        assert store.invoices.find_one({"id": invoice_id})["file_note"] == "not stored: database full"
    finally:
        _close(store)


def test_open_store_falls_back_to_sqlite_when_mongodb_fails(monkeypatch):
    """A wrong MongoDB password must not stop the app from starting."""
    from invoice_parser import config, mongo_store, storage

    def refuse(*args, **kwargs):
        raise RuntimeError("bad auth : authentication failed")

    monkeypatch.setattr(config, "MONGODB_URI", "mongodb://user:wrong@example.invalid/")
    monkeypatch.setattr(config, "MONGODB_ERROR", "")
    monkeypatch.setattr(mongo_store, "MongoInvoiceStore", refuse)
    monkeypatch.setattr(storage.InvoiceStore, "for_workspace", classmethod(lambda cls, name: "sqlite store"))
    assert storage.open_store() == "sqlite store"
    assert "bad auth" in config.MONGODB_ERROR


@needs_mongo
def test_a_photo_made_smaller_by_the_browser_is_in_mongodb_before_it_is_read(tmp_path):
    """A server that runs out of memory while reading a photo restarts without its local files: the
    photo is stored in the database as soon as it arrives (as it is, without decoding it)."""
    store = _open("mongo", tmp_path)
    try:
        data = _photo(3000, 2250)
        invoice_id, _ = store.add_file("IMG_2.jpg", data, queue=False)
        store.wait_for_uploads()
        row = store.get(invoice_id)
        assert row["status"] == "processing" and store.invoices.find_one({"id": invoice_id})["file_stored"]
        store.file_path(row).unlink()
        assert store.file_path(row).read_bytes() == data
    finally:
        _close(store)


def test_your_companies_in_the_store(store):
    """Your companies are kept with the invoices; saving them files the invoices without a company."""
    from invoice_parser.companies import file_without_company

    assert store.own_companies() == []
    sold, _ = store.add_file("s.pdf", b"s", queue=False)
    bought, _ = store.add_file("p.pdf", b"p", queue=False)
    misread, _ = store.add_file("m.pdf", b"m", queue=False)
    other, _ = store.add_file("o.pdf", b"o", queue=False)
    store.save_result(sold, _result("S-1", 1180.0, seller="Kavya Infotech Pvt Ltd", gstin="06BKXPT4822Q3ZY"), 5)
    store.save_result(bought, _result("P-1", 590.0, seller="OM TRADERS", gstin="09AGHPR3344C1ZG"), 5)
    to_us = _result("P-2", 236.0, seller="Ganesha Softech")
    to_us["fields"]["buyer"] = {"value": {"name": "Kavya lnfotecb", "tax_id": "06BKXPT4822Q3ZY"}}  # name misread
    store.save_result(misread, to_us, 5)
    store.save_result(other, _result("X-1", 100.0, seller="Ganesha Softech"), 5)
    store.set_company([bought], None, "sale")  # a side set by hand stays when the invoice is filed
    store.set_own_companies([{"name": "Kavya Infotech", "gstin": None}, {"name": "OM TRADERS", "gstin": None}])
    assert store.own_companies()[0] == {"name": "Kavya Infotech", "gstin": None}
    assert store.company_names() == ["Kavya Infotech", "OM TRADERS"]
    # the misread bill is found by Kavya Infotech's GSTIN, learned from the bill filed by its name
    assert file_without_company(store) == 3
    assert [store.get(i)["company"] for i in (sold, bought, misread, other)] == [
        "Kavya Infotech", "OM TRADERS", "Kavya Infotech", None]
    assert store.get(bought)["entry_type"] == "sale"
    dash = store.dashboard(period="all", company="Kavya Infotech")
    assert (dash["summary"]["credit"], dash["summary"]["debit"]) == (1180.0, 236.0) and dash["own_companies"]
    store.set_own_companies([])
    assert store.own_companies() == [] and store.company_names() == ["Kavya Infotech", "OM TRADERS"]  # folders stay


def test_companies_in_the_store(store):
    a, _ = store.add_file("a.pdf", b"a", queue=False, company="Kavya Infotech")
    b, _ = store.add_file("b.pdf", b"b", queue=False)
    for invoice_id, total in ((a, 1180.0), (b, 590.0)):
        store.save_result(invoice_id, _result(f"N-{invoice_id}", total, tax=round(total * 18 / 118, 2)), 5)
    assert store.company_names() == ["Kavya Infotech"]
    assert [r["id"] for r in store.search(company="Kavya Infotech")[1]] == [a]
    assert [r["id"] for r in store.search(company="-")[1]] == [b]
    assert store.add_file("again.pdf", b"b", company="OM TRADERS") == (b, False)
    assert store.get(b)["company"] == "OM TRADERS"
    assert store.add_file("again.pdf", b"b", company="Kavya Infotech") == (b, False)
    assert store.get(b)["company"] == "OM TRADERS"  # a file already in a company stays there
    assert store.set_company([a, b], "Kavya Infotech", "sale") == 2
    assert {r["entry_type"] for r in store.money_rows()} == {"sale"}
    counts = store.company_counts()
    assert counts["Kavya Infotech"]["invoices"] == 2 and None not in counts
    assert [f["id"] for f in store.files_of("Kavya Infotech")] == [a, b]
    dash = store.dashboard(period="all", company="Kavya Infotech")
    assert dash["summary"]["credit"] == 1770.0 and dash["summary"]["debit"] == 0.0
