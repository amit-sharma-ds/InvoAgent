import csv
import io
import socket
import zipfile

import pytest
from test_pipeline import _colon_layout, _label_only_layout, _to_pdf

import app as app_module
from conftest import TEST_PASSWORD, TEST_USER, login
from invoice_parser.matcher import LabelMatcher
from invoice_parser.pipeline import InvoiceParser


@pytest.fixture()
def client():
    parser = InvoiceParser(matcher=LabelMatcher(use_embeddings=False))  # fast, offline
    for w in app_module.workers.values():
        w._parser = parser
    app_module.app.config["TESTING"] = True
    return login(app_module.app.test_client())


def _drain_queue(name="live"):
    """The background threads are not started in tests: process the queue here."""
    while (job := app_module.stores[name].claim_next()) is not None:
        app_module.workers[name].process(job)


def _upload(client, data, name):
    resp = client.post("/upload", data={"files": (io.BytesIO(data), name)}, content_type="multipart/form-data")
    assert resp.status_code == 302 and "/invoices/" in resp.headers["Location"]
    return int(resp.headers["Location"].rstrip("/").rsplit("/", 1)[-1])


def test_pages_render(client):
    for url in ["/dashboard", "/upload", "/invoices", "/invoices?review=1", "/data", "/vendors",
                "/companies", "/dashboard?company=-", "/invoices?company=-"]:
        assert client.get(url).status_code == 200, url
    start = client.get("/")  # the Dashboard is the start page
    assert start.status_code == 302 and start.headers["Location"].endswith("/dashboard")


def test_sidebar_has_tabs_and_no_demo_workspace(client):
    html = client.get("/dashboard").data.decode("utf-8")
    for tab in ["Dashboard", "Companies", "Add invoices", "Invoices", "Review queue", "Data sheet", "Vendors"]:
        assert tab in html
    assert "Overview" not in html  # one start page: its counts and recent invoices are on the Dashboard
    assert "Phone camera</a>" not in html  # the camera page is part of Add invoices
    assert "/workspace/" not in html and "Demo" not in html  # the Demo workspace was removed
    assert client.post("/workspace/demo").status_code == 404
    assert client.get("/sample/anything.pdf").status_code == 404
    assert list(app_module.stores) == ["live"]


def test_reviewer_can_correct_and_approve(client):
    invoice_id = _upload(client, _to_pdf(_colon_layout), "review.pdf")
    resp = client.post(f"/invoices/{invoice_id}/edit", data={"invoice_number": "INV-FIXED-1", "total": "1,100.50",
                                                            "invoice_date": "2024-03-05"})
    assert resp.status_code == 302
    row = app_module.stores["live"].get(invoice_id)
    assert row["invoice_number"] == "INV-FIXED-1" and row["total"] == pytest.approx(1100.5) and row["edited"]
    page = client.get(f"/invoices/{invoice_id}").data.decode("utf-8")
    assert "INV-FIXED-1" in page and "corrected by reviewer" in page

    bad = client.post(f"/invoices/{invoice_id}/edit", data={"total": "abc"})
    assert bad.status_code == 302 and app_module.stores["live"].get(invoice_id)["total"] == pytest.approx(1100.5)

    client.post(f"/invoices/{invoice_id}/review", data={"decision": "approved", "note": "checked"})
    row = app_module.stores["live"].get(invoice_id)
    assert row["review_status"] == "approved" and row["review_note"] == "checked" and row["reviewed_at"]
    assert client.post(f"/invoices/{invoice_id}/review", data={"decision": "maybe"}).status_code == 400


def test_same_invoice_in_a_different_file_is_flagged_as_duplicate(client):
    first = _upload(client, _to_pdf(_colon_layout), "original.pdf")
    second = _upload(client, _to_pdf(_colon_layout) + b"\n% re-saved copy", "copy.pdf")  # other bytes, same invoice
    assert first != second
    row = app_module.stores["live"].get(second)
    original = app_module.stores["live"].get(row["duplicate_of"])  # points at the earliest copy in the database
    assert original["invoice_number"] == row["invoice_number"] and original["id"] <= first and row["needs_review"]
    assert b"possible duplicate of" in client.get(f"/invoices/{second}").data


def test_single_upload_is_parsed_saved_and_listed(client):
    resp = client.post("/upload", data={"files": (io.BytesIO(_to_pdf(_colon_layout)), "inv.pdf")},
                       content_type="multipart/form-data")
    assert resp.status_code == 302 and "/invoices/" in resp.headers["Location"]
    page = client.get(resp.headers["Location"])
    assert b"Acme Traders Pvt Ltd" in page.data and b"1,062.00" in page.data
    assert b"INV-2024-0042" in client.get("/invoices?q=INV-2024-0042").data


def test_same_file_is_not_processed_twice(client):
    data = _to_pdf(_label_only_layout)
    first = client.post("/upload", data={"files": (io.BytesIO(data), "a.pdf")}, content_type="multipart/form-data")
    second = client.post("/upload", data={"files": (io.BytesIO(data), "renamed.pdf")},
                         content_type="multipart/form-data")
    assert first.headers["Location"] == second.headers["Location"]


def test_zip_upload_becomes_a_batch_with_exports(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a.pdf", _to_pdf(_colon_layout))
        zf.writestr("folder/b.pdf", _to_pdf(_label_only_layout))
        zf.writestr("notes.txt", "not an invoice")
    buf.seek(0)
    resp = client.post("/upload", data={"files": (buf, "invoices.zip")}, content_type="multipart/form-data")
    assert resp.status_code == 302 and "/batches/" in resp.headers["Location"]
    batch_id = int(resp.headers["Location"].rstrip("/").rsplit("/", 1)[-1])

    _drain_queue()
    summary = client.get(f"/api/batches/{batch_id}").get_json()
    assert summary["total"] == 2 and summary["done"] == 2 and summary["complete"]
    assert client.get(f"/batches/{batch_id}").status_code == 200

    rows = list(csv.DictReader(io.StringIO(
        client.get(f"/export/invoices.csv?batch={batch_id}").data.decode("utf-8-sig"))))
    assert len(rows) == 2
    assert {r["po_number"] for r in rows} == {""}  # not on these invoices -> empty (NULL), same column for all
    xlsx = client.get(f"/export/invoices.xlsx?batch={batch_id}")
    assert xlsx.status_code == 200 and xlsx.data[:2] == b"PK"


def test_data_sheet_shows_all_invoices_and_line_items(client):
    client.post("/upload", data={"files": (io.BytesIO(_to_pdf(_colon_layout)), "sheet.pdf")},
                content_type="multipart/form-data")
    page = client.get("/data")
    assert page.status_code == 200
    html = page.data.decode("utf-8")
    assert "Data sheet" in html and "INV-2024-0042" in html
    assert 'data-sheet="sheet-items"' in html and "Laptop Stand" in html  # line-item sheet
    assert html.count('<details class="menu">') == 1 and "/export/invoices.xlsx" in html  # one Download menu
    assert "/export/invoices.csv" in html and "/export/line_items.csv" in html
    assert '<span class="null" title="Not found">–</span>' in html  # a missing field is shown as a dash
    assert "seller_tax_id" in html and "buyer_tax_id" in html  # every export column is there


def test_long_lists_come_in_pages(client):
    """Invoices: 20 rows a page, Rows per page 10 / 20 / 50 / 100 with the filters kept, a page past the end
    shows the last page. The Data sheet, Vendors and batch tables are paged in the browser (data-pages)."""
    store = app_module.store()
    added = [_upload(client, _to_pdf(_colon_layout) + f"%paged {n}".encode(), f"paged{n}.pdf")
             for n in range(max(0, 12 - store.stats()["total"]))]
    try:
        total = store.stats()["total"]
        page = client.get("/invoices").data.decode("utf-8")
        assert page.count('name="ids"') == min(20, total) and f" of {total}</span>" in page
        assert '<option value="20" selected>' in page and 'name="per"' in page
        pages = -(-total // 10)
        last = client.get("/invoices?per=10&page=999&q=").data.decode("utf-8")
        assert f'<span class="btn page-num" aria-current="page">{pages}</span>' in last  # numbered pages 1 2 3
        assert 'aria-label="Page 1">1</a>' in last and '<option value="10" selected>' in last
        assert last.count('name="ids"') == total - 10 * (pages - 1)
        assert '<input type="hidden" name="q" value="">' in last  # the filters stay when Rows per page changes
        assert '<option value="20" selected>' in client.get("/invoices?per=7").data.decode("utf-8")  # not offered
        for url in ["/data", "/vendors"]:
            assert "data-pages" in client.get(url).data.decode("utf-8"), url
    finally:
        store.delete(added)


def test_line_items_open_at_the_side_and_notes_are_folded(client):
    """Line items: a button on the invoice page and 'N items' in a Data sheet row open them in a side panel
    (without the script the invoice page shows them in place). Notes on an invoice are folded."""
    first = _upload(client, _to_pdf(_colon_layout) + b"%side 1", "side1.pdf")
    second = _upload(client, _to_pdf(_colon_layout) + b"%side 2", "side2.pdf")  # a possible duplicate: a note
    page = client.get(f"/invoices/{second}").data.decode("utf-8")
    assert 'data-drawer="line-items"' in page and '<aside class="drawer" id="line-items"' in page
    assert "Laptop Stand" in page and "Total of the items" in page
    assert '<details class="notes">' in page and "to check before you approve</summary>" in page
    assert "processed in" not in page.split("Technical details")[0]  # with the technical details at the end
    sheet = client.get("/data").data.decode("utf-8")
    assert f'data-items="{second}"' in sheet and f'data-invoice="{second}"' in sheet and 'id="sheet-drawer"' in sheet
    assert f'href="/invoices/{second}#line-items"' in sheet  # without the script: the invoice page's items
    assert 'data-all-columns="#sheet-invoices table"' in sheet and 'class="extra"' in sheet
    app_module.store().delete([first, second])


def test_notes_on_an_invoice_are_short():
    short = app_module.short_note
    assert short("Total was read as 271,000,008,513,000,227,004,118,656.00 and corrected to 236,000.00: the amount "
                 "in words says so. Please check.") == ("Total was misread and corrected to 236,000.00: the amount "
                                                        "in words says so.")
    assert short("Seller was inferred (no 'Seller/Vendor' label): please check it.") == \
        "Seller was inferred (no 'Seller/Vendor' label)"
    assert short("Subtotal could not be read; the line items give 200,000.00. Please check.") == \
        "Subtotal could not be read; the line items give 200,000.00."
    assert short("SGST was read as 22,600.00 and corrected to 22,500.00: CGST says so. Please check.") == \
        "SGST was read as 22,600.00 and corrected to 22,500.00: CGST says so."  # a real amount is kept
    assert short("Words read unclearly were set to their common spelling (Buyer address: Naggr → Nagar): "
                 "please check them.") == ("Words read unclearly were set to their common spelling (Buyer address: "
                                           "Naggr → Nagar)")
    assert short("Could not find invoice date.") == "Could not find invoice date."


def test_api_parse_and_list(client):
    resp = client.post("/api/parse", data={"file": (io.BytesIO(_to_pdf(_colon_layout)), "inv.pdf")},
                       content_type="multipart/form-data")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["invoice_id"] and data["fields"]["total"]["value"] == pytest.approx(1062.0)
    listing = client.get("/api/invoices?limit=5").get_json()
    assert listing["total"] >= 1 and len(listing["items"]) <= 5


def test_api_errors(client):
    assert client.post("/api/parse").status_code == 400
    resp = client.post("/api/parse", data={"file": (io.BytesIO(b"hello"), "notes.txt")},
                       content_type="multipart/form-data")
    assert resp.status_code == 422


def test_session_cookie_has_its_own_name(client):
    resp = client.post("/invoices/move", data={"company": "x"})  # nothing selected: a message in the session
    assert "invoice_parser_session=" in resp.headers.get("Set-Cookie", "")


def test_upload_page_has_one_uploader(client):
    html = client.get("/upload").data.decode("utf-8")
    assert 'id="uploader"' in html and 'id="drop-zone"' in html and 'id="upload-company"' in html
    assert 'capture="environment"' in html and "webkitdirectory" in html and "Ctrl" in html
    assert 'id="phone-camera"' in html and "<strong>Phone camera</strong>" in html  # shown on a PC too
    assert "Recent uploads" in html and "DroidCam" not in html and "/camera/phone" not in html
    old = client.get("/camera")  # old links and bookmarks open Add invoices
    assert old.status_code == 302 and old.headers["Location"].endswith("/upload")


def test_the_uploader_gets_json_and_keeps_the_message(client):
    data = _to_pdf(_colon_layout) + b"%xhr"
    first = client.post("/upload", data={"files": (io.BytesIO(data), "xhr.pdf"), "company": ""},
                        content_type="multipart/form-data", headers={"X-Upload": "1"})
    assert first.status_code == 200 and first.get_json()["next"].startswith("/invoices/")
    again = client.post("/upload", data={"files": (io.BytesIO(data), "xhr.pdf")},
                        content_type="multipart/form-data", headers={"X-Upload": "1"})
    assert again.get_json()["next"] == first.get_json()["next"]
    assert "processed before" in client.get(again.get_json()["next"]).data.decode("utf-8")  # message kept
    empty = client.post("/upload", data={}, content_type="multipart/form-data", headers={"X-Upload": "1"})
    assert empty.get_json()["next"] == "/upload"


def test_port_guard_detects_busy_port():
    busy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    busy.bind(("127.0.0.1", 0))
    busy.listen(1)
    try:
        assert app_module._port_is_free(busy.getsockname()[1]) is False
    finally:
        busy.close()


def test_rows_are_clickable_and_invoice_page_has_back_and_delete(client):
    invoice_id = _upload(client, _to_pdf(_colon_layout), "rows.pdf")
    listing = client.get("/invoices?q=rows").data.decode("utf-8")
    assert f'data-href="/invoices/{invoice_id}"' in listing and 'id="bulk-delete"' in listing
    page = client.get(f"/invoices/{invoice_id}").data.decode("utf-8")
    assert 'class="back-link" href="/invoices?q=rows"' in page  # back to the list with its filter
    assert f'action="/invoices/{invoice_id}/delete"' in page and "data-confirm" in page


def test_delete_one_and_selected(client):
    store = app_module.stores["live"]
    a = _upload(client, _to_pdf(_colon_layout) + b"%del-a", "del-a.pdf")
    b = _upload(client, _to_pdf(_label_only_layout) + b"%del-b", "del-b.pdf")
    c = _upload(client, _to_pdf(_colon_layout) + b"%del-c", "del-c.pdf")
    file_a = store.file_path(store.get(a))
    assert file_a.exists()
    resp = client.post(f"/invoices/{a}/delete")
    assert resp.status_code == 302 and store.get(a) is None and not file_a.exists()
    resp = client.post("/invoices/delete", data={"ids": [str(b), str(c)]})
    assert resp.status_code == 302 and store.get(b) is None and store.get(c) is None
    assert client.post("/invoices/999999/delete").status_code == 404


def test_deleting_the_original_clears_the_duplicate_pointer(client):
    store = app_module.stores["live"]
    first = _upload(client, _to_pdf(_label_only_layout) + b"%orig", "orig.pdf")
    second = _upload(client, _to_pdf(_label_only_layout) + b"%copy", "copy2.pdf")
    if store.get(second)["duplicate_of"] is None:
        pytest.skip("the earlier copy of this invoice is another row")
    store.delete([store.get(second)["duplicate_of"]])
    assert store.get(second)["duplicate_of"] is None


def test_the_sidebar_has_no_books_of_choice(client):
    """Your two companies: a bill from one to the other is the seller's sale. The sidebar has no 'Books of'
    choice; one company's books are seen by choosing it on the Dashboard."""
    store = app_module.store()
    invoice_id = _upload(client, _to_pdf(_colon_layout) + b"%books", "books.pdf")  # Acme Traders -> Rahul Electronics
    try:
        client.post("/companies/own", data={"companies": "Acme Traders, 27ABCDE1234F1Z0\nRahul Electronics"})
        assert store.get(invoice_id)["company"] == "Acme Traders"  # the seller's
        page = client.get(f"/invoices/{invoice_id}").data.decode("utf-8")
        assert "Books of" not in page and 'id="view-company"' not in page and 'value="sale" selected' in page
        assert client.post("/view-company", data={"company": "Rahul Electronics"}).status_code == 404
        assert "Dashboard · " not in client.get("/dashboard?period=all").data.decode("utf-8")  # all companies
        dash = client.get("/dashboard?period=all&currency=USD&company=Rahul%20Electronics").data.decode("utf-8")
        assert "Dashboard · Rahul Electronics" in dash and '<option value="Rahul Electronics" selected>' in dash
    finally:
        client.post("/companies/own", data={"companies": ""})
        store.delete([invoice_id])
        store.set_company([r["id"] for r in store.export_rows(company="Acme Traders")], None)


def test_the_original_file_json_and_exports_are_shown_before_downloading(client):
    invoice_id = _upload(client, _to_pdf(_colon_layout) + b"%view", "view.pdf")
    page = client.get(f"/invoices/{invoice_id}").data.decode("utf-8")
    assert f'href="/invoices/{invoice_id}/view"' in page  # Original file: the View page, not the file
    assert 'id="view-json"' in page and 'id="json-view" hidden' in page and 'id="download-json"' in page
    view = client.get(f"/invoices/{invoice_id}/view").data.decode("utf-8")
    assert f'src="/invoices/{invoice_id}/view/1.jpg"' in view and f"/invoices/{invoice_id}/file?download=1" in view
    picture = client.get(f"/invoices/{invoice_id}/view/1.jpg")
    assert picture.status_code == 200 and picture.mimetype == "image/jpeg" and picture.data[:2] == b"\xff\xd8"
    assert client.get(f"/invoices/{invoice_id}/view/2.jpg").status_code == 404  # the PDF has one page
    assert client.get(f"/invoices/{invoice_id}/file").headers["Content-Disposition"].startswith("inline")
    assert client.get(f"/invoices/{invoice_id}/file?download=1").headers["Content-Disposition"].startswith(
        "attachment")
    assert client.get("/invoices/999999/view").status_code == 404

    # the export buttons open the data sheet with the same filters; the downloads are there
    listing = client.get("/invoices?company=-").data.decode("utf-8")
    assert 'href="/data?company=-"' in listing and "/export/" not in listing
    sheet = client.get("/data?company=-&sheet=items").data.decode("utf-8")
    assert "Showing: company No company" in sheet and "/export/invoices.xlsx?company=-" in sheet
    assert 'id="sheet-invoices" hidden' in sheet and f'href="/invoices/{invoice_id}"' in sheet
    assert "Showing:" not in client.get("/data").data.decode("utf-8")


def test_a_picture_saved_as_jfif_is_read(client):
    """Chrome on Windows saves a picture from a web page as .jfif: it is a JPEG and is read like one."""
    import fitz

    with fitz.open(stream=_to_pdf(_colon_layout), filetype="pdf") as doc:
        jpeg = doc[0].get_pixmap(dpi=150).tobytes("jpeg")
    invoice_id = _upload(client, jpeg, "screenshot.jfif")
    assert app_module.store().get(invoice_id)["status"] == "done"
    assert client.get(f"/invoices/{invoice_id}/view/1.jpg").status_code == 200


def test_a_photo_is_shown_the_right_way_up(tmp_path):
    from PIL import Image

    from invoice_parser import file_view

    photo = Image.new("RGB", (40, 20), "white")
    exif = photo.getexif()
    exif[0x0112] = 6  # taken with the phone turned: the picture must be turned to be upright
    path = tmp_path / "photo.jpg"
    photo.save(path, "JPEG", exif=exif)
    assert file_view.page_count(path, "photo.jpg") == 1
    assert Image.open(io.BytesIO(file_view.page_jpeg(path, "photo.jpg", 1))).size == (20, 40)
    assert file_view.page_jpeg(path, "photo.jpg", 2) is None
    clear = tmp_path / "logo.png"
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(clear)
    assert Image.open(io.BytesIO(file_view.page_jpeg(clear, "logo.png", 1))).getpixel((5, 5)) == (255, 255, 255)
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    assert file_view.page_count(broken, "broken.pdf") == 0


def test_dashboard(client):
    _upload(client, _to_pdf(_colon_layout) + b"%dash", "dash.pdf")
    page = client.get("/dashboard?period=all&currency=USD")
    html = page.data.decode("utf-8")
    assert page.status_code == 200 and "Purchases and sales by year" in html and "Top suppliers" in html
    assert "Acme Traders Pvt Ltd" in html and 'class="vbar"' in html and "Sales</li>" in html
    assert "Debit" not in html and "Credit" not in html
    # what the Overview tab had: the counts of all invoices, the recent invoices
    assert '<span>need review</span>' in html and '<span>possible duplicates</span>' in html
    assert "Recent invoices" in html and "Acme Traders Pvt Ltd<br>" in html
    # a working app, not a demo: no help texts on how the app works
    assert "How invoices flow through this app" not in html and "Purchases</strong> = bills" not in html
    for url in ["/upload", "/data", "/invoices", "/vendors", "/companies"]:
        page = client.get(url).data.decode("utf-8")
        assert "For the best results" not in page and "like a spreadsheet" not in page, url
        assert "Click a row to open" not in page and "One folder per company" not in page, url
    assert client.get("/dashboard?period=fy&currency=XYZ").status_code == 200  # unknown currency: falls back


def test_indian_amount_format():
    assert app_module.format_amount(295000, "INR") == "2,95,000.00"
    assert app_module.format_amount(10520000.5, "INR") == "1,05,20,000.50"
    assert app_module.format_amount(295000, "USD") == "295,000.00"
    assert app_module._nice_max(295000) == 500000 and app_module._nice_max(15655.53) == 20000


def test_everything_needs_sign_in():
    anon = app_module.app.test_client()
    resp = anon.get("/invoices?q=x")
    assert resp.status_code == 302 and "/login?next=/invoices?q%3Dx" in resp.headers["Location"].replace("%3F", "?")
    for url in ["/", "/dashboard", "/data", "/export/invoices.csv", "/camera", "/invoices/1/file"]:
        assert anon.get(url).status_code == 302, url
    assert anon.post("/invoices/1/delete").status_code == 302  # to the sign-in page, nothing deleted
    assert anon.get("/api/invoices").status_code == 401
    assert anon.get("/health").status_code == 200 and anon.get("/static/style.css").status_code == 200
    assert anon.get("/login").status_code == 200


def test_sign_in_wrong_password_logout_and_safe_next():
    from invoice_parser import auth
    login(app_module.app.test_client())  # makes sure the test user exists
    anon = app_module.app.test_client()
    bad = anon.post("/login", data={"username": TEST_USER, "password": "wrong-password"})
    assert bad.status_code == 401 and b"Wrong user name or password" in bad.data
    auth.record_success("127.0.0.1")
    ok = anon.post("/login", data={"username": TEST_USER, "password": TEST_PASSWORD, "next": "//evil.example/x"})
    assert ok.status_code == 302 and ok.headers["Location"] == "/"  # never sent to another site
    page = anon.get("/dashboard").data.decode("utf-8")
    assert f"Signed in as <strong>{TEST_USER}</strong>" in page
    anon.post("/logout")
    assert anon.get("/").status_code == 302


def test_too_many_wrong_passwords_lock_the_address():
    from invoice_parser import auth
    anon = app_module.app.test_client()
    try:
        for _ in range(auth.MAX_FAILURES):
            anon.post("/login", data={"username": TEST_USER, "password": "nope-nope"})
        locked = anon.post("/login", data={"username": TEST_USER, "password": TEST_PASSWORD})
        assert locked.status_code == 401 and b"Too many wrong passwords" in locked.data
    finally:
        auth.record_success("127.0.0.1")


def test_api_accepts_basic_auth():
    import base64
    login(app_module.app.test_client())
    anon = app_module.app.test_client()
    token = base64.b64encode(f"{TEST_USER}:{TEST_PASSWORD}".encode()).decode()
    assert anon.get("/api/invoices", headers={"Authorization": f"Basic {token}"}).status_code == 200
    bad = base64.b64encode(f"{TEST_USER}:wrong".encode()).decode()
    assert anon.get("/api/invoices", headers={"Authorization": f"Basic {bad}"}).status_code == 401
    from invoice_parser import auth
    auth.record_success("127.0.0.1")


def test_passwords_are_stored_as_hashes(tmp_path):
    from invoice_parser import auth
    path = tmp_path / "users.json"
    auth.set_password("someone", "a-long-password", path=path)
    assert "a-long-password" not in path.read_text() and auth.verify("someone", "a-long-password", path=path)
    assert auth.verify("someone", "wrong", path=path) is None
    with pytest.raises(ValueError):
        auth.set_password("someone", "short", path=path)


def test_first_user_from_server_settings(tmp_path, monkeypatch):
    """Render / Docker: the first user comes from INVOICE_ADMIN_USER / INVOICE_ADMIN_PASSWORD, and an
    existing user is never overwritten by them."""
    from invoice_parser import auth

    path = tmp_path / "users.json"
    monkeypatch.setenv("INVOICE_ADMIN_USER", "boss")
    monkeypatch.setenv("INVOICE_ADMIN_PASSWORD", "long-enough-1")
    assert auth.bootstrap_admin_from_env(path) is True
    assert auth.verify("boss", "long-enough-1", path)
    monkeypatch.setenv("INVOICE_ADMIN_PASSWORD", "another-pass-2")
    assert auth.bootstrap_admin_from_env(path) is False
    assert auth.verify("boss", "long-enough-1", path)


def test_scanner_page_and_folder_import(client, tmp_path):
    from invoice_parser.scanner import import_folder_once

    assert client.get("/scanner").status_code == 200 and "Scanner" in client.get("/dashboard").data.decode("utf-8")
    resp = client.post("/scanner", data={"folder": str(tmp_path / "missing"), "enabled": "1"})
    assert resp.status_code == 302 and app_module.load_settings().get("scan_folder_on") is not True
    (tmp_path / "scan-001.pdf").write_bytes(_to_pdf(_colon_layout) + b"%scanner")
    (tmp_path / "notes.txt").write_text("not an invoice")
    store = app_module.stores["live"]
    assert import_folder_once(tmp_path, store, settle=0) == 1
    assert (tmp_path / "imported" / "scan-001.pdf").exists() and (tmp_path / "notes.txt").exists()
    _drain_queue()
    _, rows = store.search(q="scan-001")
    assert rows and rows[0]["status"] == "done" and rows[0]["invoice_number"] == "INV-2024-0042"
    assert client.post("/scanner", data={"folder": str(tmp_path), "enabled": "1"}).status_code == 302
    assert app_module.load_settings()["scan_folder_on"] is True
    client.post("/scanner", data={"folder": str(tmp_path)})  # switched off again


def _upload_to(client, data, name, company):
    resp = client.post("/upload", data={"files": (io.BytesIO(data), name), "company": company},
                       content_type="multipart/form-data")
    return int(resp.headers["Location"].rstrip("/").rsplit("/", 1)[-1])


def test_company_folders(client):
    """Upload into a company: its folder, dashboard, filter, move, side by hand and ZIP of its files."""
    store = app_module.store()
    first = _upload_to(client, _to_pdf(_colon_layout) + b"%co1", "co1.pdf", "  Kavya   Infotech ")
    assert store.get(first)["company"] == "Kavya Infotech"
    assert 'value="Kavya Infotech"' in client.get("/upload").data.decode("utf-8")  # offered next time
    second = _upload_to(client, _to_pdf(_colon_layout) + b"%co2", "co2.pdf", "kavya infotech")
    assert store.get(second)["company"] == "Kavya Infotech"  # same folder, whatever the case

    html = client.get("/companies").data.decode("utf-8")
    assert "Kavya Infotech" in html and "Files (ZIP)" in html
    listed = client.get("/invoices?company=Kavya+Infotech").data.decode("utf-8")
    assert f'value="{first}"' in listed and f'value="{second}"' in listed

    # the sample's parties are not the company: purchase unless set by hand (no 'Auto' choice)
    page = client.get(f"/invoices/{first}").data.decode("utf-8")
    assert 'value="purchase" selected' in page and "Auto" not in page
    shown = {"company": "Kavya Infotech", "side_shown": "purchase"}
    client.post(f"/invoices/{first}/company", data={**shown, "entry_type": "purchase"})
    assert store.get(first)["entry_type"] is None  # left as shown: still found from the bill
    client.post(f"/invoices/{first}/company", data={**shown, "entry_type": "sale"})
    assert store.get(first)["entry_type"] == "sale"
    assert 'value="sale" selected' in client.get(f"/invoices/{first}").data.decode("utf-8")
    client.post(f"/invoices/{first}/company", data={"company": "Kavya Infotech", "side_shown": "sale",
                                                     "entry_type": "sale"})
    assert store.get(first)["entry_type"] == "sale"  # left as shown: the side set by hand stays
    assert "Kavya Infotech" in client.get("/dashboard?period=all").data.decode("utf-8")  # in the company filter

    archive = client.get("/companies/files.zip?company=Kavya+Infotech")
    names = zipfile.ZipFile(io.BytesIO(archive.data)).namelist()
    assert archive.status_code == 200 and sorted(names) == [f"Kavya Infotech/{first:05d}-co1.pdf",
                                                            f"Kavya Infotech/{second:05d}-co2.pdf"]
    text = client.get("/export/invoices.csv?company=Kavya+Infotech").data.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    assert {r["id"]: (r["company"], r["entry_type"]) for r in rows}[str(first)] == ("Kavya Infotech", "sale")
    assert len(rows) == 2

    client.post("/invoices/move", data={"ids": [str(second)], "company": "OM TRADERS"})
    assert store.get(second)["company"] == "OM TRADERS" and store.get(first)["company"] == "Kavya Infotech"
    client.post("/invoices/move", data={"ids": [str(first), str(second)], "company": ""})
    assert store.get(first)["company"] is None and store.get(first)["entry_type"] is None


def test_projects(client):
    """Bills put in a project when uploading, on the invoice page and from the list; the dashboard shows
    each project's sales, purchases and margin without GST."""
    store = app_module.store()
    resp = client.post("/upload", data={"files": (io.BytesIO(_to_pdf(_colon_layout) + b"%pr1"), "pr1.pdf"),
                                        "project": " Racking   Line "}, content_type="multipart/form-data")
    first = int(resp.headers["Location"].rstrip("/").rsplit("/", 1)[-1])
    second = None
    try:
        assert store.get(first)["project"] == "Racking Line"
        assert 'id="upload-project" value="Racking Line"' in client.get("/upload").data.decode("utf-8")
        second = _upload(client, _to_pdf(_colon_layout) + b"%pr2", "pr2.pdf")  # the project offered is used
        assert store.get(second)["project"] == "Racking Line"
        assert 'id="invoice-project" value="Racking Line"' in client.get(f"/invoices/{first}").data.decode("utf-8")

        shown = {"company": "", "side_shown": "purchase", "entry_type": "purchase"}
        client.post(f"/invoices/{second}/company", data={**shown, "project": ""})
        assert store.get(second)["project"] is None
        client.post(f"/invoices/{second}/company", data={**shown, "project": "racking line"})
        assert store.get(second)["project"] == "Racking Line"  # the known spelling, whatever the case
        client.post("/invoices/project", data={"ids": [str(second)], "project": "Office"})
        assert store.get(second)["project"] == "Office" and store.get(first)["project"] == "Racking Line"
        listed = client.get("/invoices?project=Office").data.decode("utf-8")
        assert f'name="ids" value="{second}"' in listed and f'name="ids" value="{first}"' not in listed

        store.set_review(first, "approved")  # the sample is a possible duplicate of the earlier tests' copies
        row = store.get(first)
        d = store.dashboard(period="all", currency=row["currency"], project="Racking Line")
        assert [(p["name"], p["purchases"]) for p in d["project_rows"]] == [
            ("Racking Line", round(row["total"] - (row["tax"] or 0), 2))]
        html = client.get(f"/dashboard?period=all&currency={row['currency']}&project=Racking+Line").data.decode("utf-8")
        assert "Dashboard · Racking Line</h1>" in html and "Project-wise" in html and "Margin&nbsp;%" in html
        assert "Project-wise" in client.get(f"/dashboard?period=all&currency={row['currency']}").data.decode("utf-8")

        text = client.get("/export/invoices.csv?project=Office").data.decode("utf-8-sig")
        assert [(r["id"], r["project"]) for r in csv.DictReader(io.StringIO(text))] == [(str(second), "Office")]
        client.post("/invoices/project", data={"ids": [str(first), str(second)], "project": ""})
        assert store.get(first)["project"] is None and store.get(second)["project"] is None
    finally:
        store.delete([i for i in (first, second) if i])


def test_your_companies_make_their_bills_sales(client):
    """Your companies typed once: their sales bills count as sales without picking a folder."""
    store = app_module.store()
    first = _upload(client, _to_pdf(_colon_layout) + b"%own1", "own1.pdf")  # seller: Acme Traders Pvt Ltd
    try:
        assert store.get(first)["company"] is None
        assert "Your companies are not set" in client.get("/dashboard?period=all").data.decode("utf-8")
        resp = client.post("/companies/own", data={"companies": "Acme Traders, GSTIN 27ABCDE1234F1Z0\nOM TRADERS"})
        assert resp.status_code == 302 and store.get(first)["company"] == "Acme Traders"
        html = client.get("/companies").data.decode("utf-8")
        assert "Acme Traders, 27ABCDE1234F1Z0\nOM TRADERS</textarea>" in html
        assert " without a company moved into their folder." in html  # also the earlier tests' samples
        assert '<strong class="folder-name">OM TRADERS</strong>' in html  # an empty folder for your company
        assert 'value="sale" selected' in client.get(f"/invoices/{first}").data.decode("utf-8")
        second = _upload(client, _to_pdf(_colon_layout) + b"%own2", "own2.pdf")
        assert store.get(second)["company"] == "Acme Traders"  # new invoices go there by themselves
        dash = store.dashboard(period="all", company="Acme Traders")
        assert dash["summary"]["credit_n"] >= 1 and dash["summary"]["debit_n"] == 0  # all its bills are sales
        assert "Your companies are not set" not in client.get("/dashboard?period=all").data.decode("utf-8")
    finally:  # the other tests' invoices go back to no company
        client.post("/companies/own", data={"companies": ""})
        store.delete([i for i in (first, locals().get("second")) if i])
        store.set_company([r["id"] for r in store.export_rows(company="Acme Traders")], None)
    assert store.own_companies() == [] and "Acme Traders" not in store.company_names()
