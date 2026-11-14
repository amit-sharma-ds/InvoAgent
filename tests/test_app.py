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
    for url in ["/", "/dashboard", "/upload", "/invoices", "/invoices?review=1", "/data", "/vendors", "/camera"]:
        assert client.get(url).status_code == 200, url


def test_sidebar_has_tabs_and_no_demo_workspace(client):
    html = client.get("/").data.decode("utf-8")
    for tab in ["Overview", "Upload", "Invoices", "Review queue", "Data sheet", "Vendors", "Phone camera"]:
        assert tab in html
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
    assert '<span class="null">null</span>' in html  # missing fields are shown as null
    assert "seller_tax_id" in html and "buyer_tax_id" in html  # every export column is there


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
    resp = client.post("/camera/capture", data={"source": "ip", "ip_url": "http://8.8.8.8:8080"})
    assert "invoice_parser_session=" in resp.headers.get("Set-Cookie", "")


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


def test_dashboard(client):
    _upload(client, _to_pdf(_colon_layout) + b"%dash", "dash.pdf")
    page = client.get("/dashboard?period=all&currency=USD")
    html = page.data.decode("utf-8")
    assert page.status_code == 200 and "Invoice value by year" in html and "Top vendors" in html
    assert "Acme Traders Pvt Ltd" in html and 'class="vbar"' in html
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
    page = anon.get("/").data.decode("utf-8")
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

    assert client.get("/scanner").status_code == 200 and "Scanner" in client.get("/").data.decode("utf-8")
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
