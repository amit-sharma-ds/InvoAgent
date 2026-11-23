"""Flask web app for the invoice parser.

Sidebar tabs: Overview, Dashboard, Companies, Upload (files, phone camera photo, paste), Invoices,
Review queue, Data sheet, Vendors, Scanner.
Data: MongoDB when MONGODB_URI is set (shared by every computer and the server), else SQLite in
data/live (see config.py).
JSON API: POST /api/parse, GET /api/invoices, GET /api/batches/<id>, GET /health.

Run:  python app.py
  This PC:            http://localhost:5077
  Phone (same Wi-Fi): http://<this-PC-IP>:5077

Port 5077 is used instead of Flask's usual 5000 so this app never collides with other
local Flask projects. Change it with the INVOICE_PARSER_PORT environment variable.
"""
import io
import json
import logging
import os
import re
import socket
import sys
import tempfile
import threading
import time
import zipfile
from datetime import timedelta
from datetime import datetime
from pathlib import Path

from flask import (Flask, Response, abort, flash, jsonify, redirect, render_template, request,
                   send_file, session, url_for)

from invoice_parser import auth, config, fast_ocr
from invoice_parser.companies import (ENTRY_TYPES, NO_COMPANY, clean_company, company_gstins,
                                      file_without_company, folders, own_companies_text, parse_own_companies,
                                      side_of, with_sides)
from invoice_parser.settings import load_settings, save_settings
from invoice_parser.exporting import build_xlsx, to_csv
from invoice_parser.pipeline import get_parser
from invoice_parser.scanner import FolderWatcher
from invoice_parser.storage import (EDITABLE_FIELDS, EXPORT_COLUMNS, ITEM_COLUMNS, REVIEW_STATUSES, STATUSES,
                                    TABLE_COLUMNS, move_legacy_data_to_demo, open_store)
from invoice_parser.worker import BatchWorker

log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
app.secret_key = auth.secret_key()  # random, kept in data/secret_key: session cookies cannot be forged
# Browsers share cookies across every localhost port. A unique cookie name stops this app
# and another local Flask project from overwriting each other's "session" cookie.
app.config["SESSION_COOKIE_NAME"] = "invoice_parser_session"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"  # other sites cannot send forms (e.g. delete) with this login
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=12)  # sign in again after 12 hours

# INVOICE_PARSER_PORT on this PC; PORT is what cloud hosts such as Render set
PORT = int(os.environ.get("INVOICE_PARSER_PORT") or os.environ.get("PORT") or "5077")
ALLOWED_EXTENSIONS = config.PDF_EXTENSIONS | config.IMAGE_EXTENSIONS
MAX_ZIP_MEMBER = 50 * 1024 * 1024        # skip single files above 50 MB inside a ZIP
MAX_ZIP_TOTAL = 2 * 1024 * 1024 * 1024   # stop unpacking after 2 GB (ZIP-bomb guard)
PAGE_SIZE = 50
SHEET_ROW_LIMIT = 5000  # rows shown in the browser; the Excel / CSV downloads always have everything
MONEY_COLUMNS = {"subtotal", "tax", "discount", "shipping", "total"}
EDIT_LABELS = [("invoice_number", "Invoice number"), ("invoice_date", "Invoice date"), ("due_date", "Due date"),
               ("po_number", "PO number"), ("currency", "Currency"), ("seller_name", "Seller name"),
               ("seller_tax_id", "Seller GSTIN / tax ID"), ("buyer_name", "Buyer name"),
               ("buyer_tax_id", "Buyer GSTIN / tax ID"), ("subtotal", "Subtotal"), ("tax", "Tax"),
               ("discount", "Discount"), ("shipping", "Shipping / freight"), ("total", "Total")]

try:  # older versions kept all data in data/invoices.db: that becomes the Demo workspace
    if move_legacy_data_to_demo():
        log.info("Moved the earlier data to the Demo workspace.")
except OSError as exc:
    log.warning("Could not move the earlier data to the Demo workspace: %s", exc)

stores = {"live": open_store()}
workers = {"live": BatchWorker(stores["live"])}
scan_watcher = FolderWatcher(stores["live"], workers["live"].notify, load_settings)  # scanner's folder


# --- workspace and helpers ---------------------------------------------------------------
def workspace():
    return "live"  # the only workspace (the Demo workspace was removed)


def store():
    return stores[workspace()]


def worker():
    return workers[workspace()]


# Pages that list invoices: the Back button on an invoice returns to the last one of these (with its filters).
LIST_PAGES = {"index", "upload_page", "invoices", "data_sheet", "vendors", "batch_detail", "dashboard", "companies"}


# --- sign-in ---------------------------------------------------------------------------------
PUBLIC_ENDPOINTS = {"login", "static", "health"}


def _client_address():
    return request.remote_addr or "unknown"


@app.before_request
def require_login():
    """Every page, download and API call needs a signed-in user. The JSON API also
    accepts the same user name and password as HTTP Basic auth (for other company systems)."""
    if request.endpoint in PUBLIC_ENDPOINTS or session.get("user"):
        return None
    if request.path.startswith("/api/"):
        creds = request.authorization
        if creds and creds.type == "basic":
            if auth.locked_out(_client_address()):
                return jsonify(error="Too many wrong passwords. Try again later."), 429
            if auth.verify(creds.username, creds.password):
                auth.record_success(_client_address())
                return None
            auth.record_failure(_client_address())
        return jsonify(error="Sign in required (session or HTTP Basic auth)."), 401, {
            "WWW-Authenticate": 'Basic realm="InvoAgent"'}
    if request.method == "GET":
        return redirect(url_for("login", next=request.full_path.rstrip("?")))
    return redirect(url_for("login"))


def _safe_next(url):
    return url if url and url.startswith("/") and not url.startswith("//") else url_for("index")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user"):
        return redirect(_safe_next(request.args.get("next")))
    error, users_exist = None, auth.has_users()
    if request.method == "POST":
        wait = auth.locked_out(_client_address())
        if wait:
            error = f"Too many wrong passwords. Try again in {max(1, wait // 60)} minute(s)."
        elif auth.verify(request.form.get("username"), request.form.get("password")):
            auth.record_success(_client_address())
            session.clear()  # a fresh session at sign-in
            session.permanent = True
            session["user"] = request.form.get("username", "").strip()
            return redirect(_safe_next(request.form.get("next")))
        else:
            auth.record_failure(_client_address())
            error = "Wrong user name or password."
            log.warning("Failed sign-in for %r from %s", request.form.get("username"), _client_address())
    return render_template("login.html", error=error, users_exist=users_exist,
                           next=request.values.get("next", "")), (401 if error else 200)


@app.post("/logout")
def logout():
    session.clear()
    flash("You are signed out.")
    return redirect(url_for("login"))


@app.before_request
def remember_list_page():
    if request.method == "GET" and request.endpoint in LIST_PAGES:
        session["back_url"] = request.full_path.rstrip("?")


def back_url():
    url = session.get("back_url") or ""
    return url if url.startswith("/") and not url.startswith("//") else url_for("invoices")


@app.context_processor
def inject_layout():
    if not session.get("user"):
        return {}
    return {"workspace": workspace(), "nav_stats": store().stats(), "endpoint": request.endpoint,
            "back_url": back_url(), "current_user": session.get("user"), "db_error": config.MONGODB_ERROR,
            "company_names": lambda: store().company_names(),  # called only by the pages that list them
            "current_company": session.get("company", ""), "NO_COMPANY": NO_COMPANY}


@app.template_global()
def url_with(endpoint, **changes):
    """URL for endpoint keeping the current filters (search, status, page ...)."""
    args = request.args.to_dict()
    args.update({k: v for k, v in changes.items() if v is not None})
    return url_for(endpoint, **args)


def _back_to(endpoint, message, **values):
    flash(message)
    return redirect(url_for(endpoint, **values))


def _filters():
    return {
        "q": request.args.get("q", "").strip() or None,
        "status": request.args.get("status") or None,
        "review": request.args.get("review") == "1",
        "batch_id": request.args.get("batch", type=int),
        "review_status": request.args.get("rstatus") or None,
        "company": request.args.get("company") or None,
    }


def _chosen_company(value):
    """A company typed in a form; the known spelling when it differs only in case. None: no company."""
    return clean_company(value, store().company_names())


def _remember_company():
    """The company field of an upload form: used for this upload and offered next time.
    Empty: the invoice goes to the known company it names, if any."""
    if "company" not in request.form:
        return clean_company(session.get("company"))
    company = _chosen_company(request.form.get("company"))
    session["company"] = company or ""
    return company


def _render_result(result, invoice=None):
    # Escape "</" so the embedded JSON can never close the <script> tag early.
    payload = json.dumps(result, ensure_ascii=False, indent=2).replace("</", "<\\/")
    return render_template("result.html", result=result, result_json=payload, invoice=invoice,
                           edit_labels=EDIT_LABELS)


def _expand_uploads(uploads):
    """Uploaded files -> [(name, bytes)], unpacking ZIP archives. Also returns skipped names."""
    files, skipped = [], []
    for up in uploads:
        if not up or not up.filename:
            continue
        name = Path(up.filename).name
        data = up.read()
        ext = Path(name).suffix.lower()
        if ext == ".zip":
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    unpacked = 0
                    for info in zf.infolist():
                        member = Path(info.filename).name
                        if info.is_dir() or not member or member.startswith("._"):
                            continue
                        if (Path(member).suffix.lower() not in ALLOWED_EXTENSIONS
                                or info.file_size > MAX_ZIP_MEMBER or unpacked + info.file_size > MAX_ZIP_TOTAL):
                            skipped.append(member)
                            continue
                        unpacked += info.file_size
                        files.append((member, zf.read(info)))
            except zipfile.BadZipFile:
                skipped.append(name)
        elif ext in ALLOWED_EXTENSIONS:
            files.append((name, data))
        else:
            skipped.append(name)
    return files, skipped


def _store_and_process(name, data, company=None):
    """Save one file and parse it right away. Returns (invoice_id, is_new)."""
    invoice_id, is_new = store().add_file(name, data, queue=False, company=company)
    if is_new:
        worker().process_now(invoice_id)
    return invoice_id, is_new


# --- pages -------------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("overview.html", overview=store().overview())


@app.get("/upload")
def upload_page():
    return render_template("upload.html", batches=store().list_batches(limit=5),
                           recent=store().search(limit=8)[1], max_mb=config.MAX_UPLOAD_MB)


def _after_upload(url, message=None):
    """Where the browser goes next. The Upload page's own uploader (header X-Upload) gets JSON
    {"next": url} instead of a redirect, so a message for that page is not used up on the way."""
    if message:
        flash(message)
    if request.headers.get("X-Upload") == "1":
        return jsonify(next=url)
    return redirect(url)


@app.post("/upload")
def upload():
    uploads = request.files.getlist("files") + request.files.getlist("file")
    has_zip = any(u.filename and u.filename.lower().endswith(".zip") for u in uploads)
    company = _remember_company()
    files, skipped = _expand_uploads(uploads)
    if not files:
        if skipped:
            return _after_upload(url_for("upload_page"), f"No supported files found. Skipped: {', '.join(skipped[:5])}")
        return _after_upload(url_for("upload_page"), "Choose a PDF, an image or a ZIP file.")

    if len(files) == 1 and not has_zip:  # single invoice: process now and show the result
        invoice_id, is_new = _store_and_process(*files[0], company=company)
        return _after_upload(url_for("invoice_detail", invoice_id=invoice_id),
                             None if is_new else "This exact file was processed before. Showing the saved result.")

    batch_id = store().create_batch(f"Upload of {len(files)} files, {datetime.now():%d %b %Y %H:%M}")
    duplicates = sum(1 for name, data in files
                     if not store().add_file(name, data, batch_id=batch_id, company=company)[1])
    worker().notify()
    note = f"{len(files) - duplicates} files queued for processing" + (f" in {company}." if company else ".")
    if duplicates:
        note += f" {duplicates} were already in the database and were skipped."
    if skipped:
        note += f" {len(skipped)} unsupported files were ignored."
    return _after_upload(url_for("batch_detail", batch_id=batch_id), note)


@app.get("/invoices")
def invoices():
    filters = _filters()
    page = max(1, request.args.get("page", 1, type=int))
    total, rows = store().search(**filters, limit=PAGE_SIZE, offset=(page - 1) * PAGE_SIZE)
    return render_template("invoices.html", companies=store().company_names(), rows=rows, columns=TABLE_COLUMNS, total=total, page=page,
                           pages=max(1, -(-total // PAGE_SIZE)), filters=filters, stats=store().stats(),
                           statuses=STATUSES, review_statuses=REVIEW_STATUSES, money_columns=MONEY_COLUMNS)


@app.get("/data")
def data_sheet():
    """Spreadsheet view: every invoice (and every line item) in one table."""
    rows = with_sides(store().export_rows(), store().money_rows(), store().own_companies())
    items = store().export_line_items()
    return render_template("data.html", rows=rows[:SHEET_ROW_LIMIT], items=items[:SHEET_ROW_LIMIT * 5],
                           total_rows=len(rows), total_items=len(items), columns=EXPORT_COLUMNS,
                           item_columns=ITEM_COLUMNS, money_columns=MONEY_COLUMNS | {"unit_price", "amount"},
                           yes_no_columns={"needs_review", "totals_ok", "line_items_ok", "edited"})


@app.template_filter("amount")
def format_amount(value, currency=None):
    """2,95,000.00 for INR (lakh grouping), 295,000.00 otherwise."""
    if value is None:
        return ""
    if currency != "INR":
        return f"{value:,.2f}"
    whole, frac = f"{abs(value):.2f}".split(".")
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    text = ",".join(([head] if head else []) + groups + [tail])
    return f"{'-' if value < 0 else ''}{text}.{frac}"


def _nice_max(value):
    """Round an axis maximum up to 1, 2, 2.5 or 5 x a power of ten."""
    if not value or value <= 0:
        return 1
    power = 10 ** (len(str(int(value))) - 1)
    for step in (1, 2, 2.5, 5, 10):
        if value <= step * power:
            return step * power
    return 10 * power


@app.get("/dashboard")
def dashboard():
    """Debit (purchases) and credit (sales) by month and by company, GST, top suppliers and customers,
    for one currency, one period and all or one company."""
    data = store().dashboard(currency=request.args.get("currency"), period=request.args.get("period", "12m"),
                             company=request.args.get("company") or None)
    data["axis_max"] = _nice_max(max((max(b["debit"], b["credit"]) for b in data["buckets"]), default=0))
    data["vendor_max"] = max((v["total"] or 0 for v in data["vendors"]), default=0) or 1
    data["customer_max"] = max((v["total"] or 0 for v in data["customers"]), default=0) or 1
    data["company_max"] = max((max(c["debit"], c["credit"]) for c in data["company_rows"]), default=0) or 1
    return render_template("dashboard.html", d=data, periods=[("12m", "Last 12 months"),
                           ("fy", "This financial year"), ("lfy", "Last financial year"), ("all", "All years")])


@app.get("/vendors")
def vendors():
    return render_template("vendors.html", vendors=store().vendors())


@app.get("/invoices/<int:invoice_id>")
def invoice_detail(invoice_id):
    row = store().get(invoice_id)
    if row is None:
        abort(404)
    if row["status"] != "done":
        return render_template("invoice_status.html", invoice=row)
    gstin = (company_gstins(store().money_rows(company=row["company"]), store().own_companies()).get(row["company"])
             if row["company"] else None)
    row["side"] = side_of(row, gstin)
    return _render_result(json.loads(row["result_json"]), invoice=row)


@app.post("/invoices/<int:invoice_id>/company")
def set_invoice_company(invoice_id):
    """Move one invoice to another company and / or set its side (purchase or sale)."""
    if store().get(invoice_id) is None:
        abort(404)
    company = _chosen_company(request.form.get("company"))
    entry_type = request.form.get("entry_type")
    store().set_company([invoice_id], company, entry_type if entry_type in ENTRY_TYPES else None)
    flash(f"Invoice #{invoice_id} is in {company or 'no company'}.")
    return redirect(url_for("invoice_detail", invoice_id=invoice_id))


@app.post("/invoices/move")
def move_invoices():
    """Move the invoices selected in the list to a company."""
    ids = [int(i) for i in request.form.getlist("ids") if i.isdigit()]
    if not ids:
        return _back_to("invoices", "Select one or more invoices to move.")
    company = _chosen_company(request.form.get("company"))
    moved = store().set_company(ids, company)
    flash(f"{moved} invoice{'s' if moved != 1 else ''} moved to {company or 'no company'}.")
    return redirect(back_url())


# --- companies (one folder each) -----------------------------------------------------------
@app.get("/companies")
def companies():
    own = store().own_companies()
    return render_template("companies.html", folders=folders(store().company_counts(), store().money_rows(), own),
                           own_text=own_companies_text(own))


@app.post("/companies/own")
def save_own_companies():
    """Your companies: a bill where one of them is the seller counts as a sale, where it is the buyer as a
    purchase. The invoices without a company that name one of them are put into its folder."""
    folders_now = [c for c in store().company_counts() if c]
    firms = [{**f, "name": clean_company(f["name"], folders_now)}  # the folder's spelling when it exists
             for f in parse_own_companies(request.form.get("companies"))]
    store().set_own_companies(firms)
    moved = file_without_company(store()) if firms else 0
    message = f"Your companies: {', '.join(f['name'] for f in firms) or 'none'}."
    if moved:
        message += (f" {moved} invoice{'s' if moved != 1 else ''} without a company moved into "
                    f"{'their' if len(firms) > 1 else 'its'} folder.")
    flash(message)
    return redirect(url_for("companies"))


def _safe_file_name(name):
    return re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", name or "").strip(" .") or "file"


@app.get("/companies/files.zip")
def company_files_zip():
    """A company's folder as a ZIP: every original file it holds, in a folder named after it."""
    company = request.args.get("company") or NO_COMPANY
    rows = store().files_of(company)
    if not rows:
        return _back_to("companies", "This folder has no files.")
    folder = _safe_file_name("No company" if company == NO_COMPANY else company)
    out, missing = tempfile.TemporaryFile(), 0  # on disk: a large folder does not fill the memory
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:  # PDFs and photos are compressed already
        for r in rows:
            path = store().file_path(r)
            if path.exists():
                zf.write(path, f"{folder}/{r['id']:05d}-{_safe_file_name(r['file_name'])}")
            else:
                missing += 1
        if missing:
            zf.writestr(f"{folder}/MISSING FILES.txt", f"{missing} original file(s) were not available "
                        "(the database had no space left for them); their invoice data is saved.\n")
    out.seek(0)
    return send_file(out, mimetype="application/zip", as_attachment=True,
                     download_name=f"{folder}-{datetime.now():%Y%m%d-%H%M}.zip")


@app.get("/invoices/<int:invoice_id>/file")
def invoice_file(invoice_id):
    row = store().get(invoice_id)
    if row is None:
        abort(404)
    path = store().file_path(row)
    if not path.exists():
        return _back_to("invoice_detail", "The original file is not available (the database had no space "
                        "left for it); the invoice data is saved.", invoice_id=invoice_id)
    return send_file(path, download_name=row["file_name"])


@app.post("/invoices/<int:invoice_id>/reprocess")
def reprocess(invoice_id):
    if store().get(invoice_id) is None:
        abort(404)
    store().requeue(invoice_id)
    worker().notify()
    flash("Queued for processing again.")
    return redirect(url_for("invoice_detail", invoice_id=invoice_id))


@app.post("/invoices/<int:invoice_id>/edit")
def edit_invoice(invoice_id):
    """A reviewer corrects extracted values."""
    if store().get(invoice_id) is None:
        abort(404)
    values = {col: request.form.get(col) for col in EDITABLE_FIELDS if col in request.form}
    try:
        store().update_fields(invoice_id, values)
    except ValueError as exc:
        return _back_to("invoice_detail", str(exc), invoice_id=invoice_id)
    flash("Corrections saved.")
    return redirect(url_for("invoice_detail", invoice_id=invoice_id))


@app.post("/invoices/<int:invoice_id>/review")
def review(invoice_id):
    """Approve or reject an invoice (and optionally open the next one that needs review)."""
    decision = request.form.get("decision", "")
    if decision not in REVIEW_STATUSES or store().get(invoice_id) is None:
        abort(400)
    store().set_review(invoice_id, decision, request.form.get("note"))
    flash(f"Invoice #{invoice_id} marked as {decision}.")
    if request.form.get("next") == "1":
        _, waiting = store().search(review=True, status="done", review_status="pending", limit=1)
        if waiting:
            return redirect(url_for("invoice_detail", invoice_id=waiting[0]["id"]))
        flash("No more invoices waiting for review.")
        return redirect(url_for("invoices", review="1"))
    return redirect(url_for("invoice_detail", invoice_id=invoice_id))


@app.post("/invoices/<int:invoice_id>/delete")
def delete_invoice(invoice_id):
    """Delete one invoice (row, line items and the stored original file)."""
    if store().get(invoice_id) is None:
        abort(404)
    store().delete([invoice_id])
    flash(f"Invoice #{invoice_id} was deleted.")
    target = back_url()
    return redirect(url_for("invoices") if target.startswith(f"/invoices/{invoice_id}") else target)


@app.post("/invoices/delete")
def delete_invoices():
    """Delete the invoices selected in the list."""
    ids = [int(i) for i in request.form.getlist("ids") if i.isdigit()]
    if not ids:
        return _back_to("invoices", "Select one or more invoices to delete.")
    deleted = store().delete(ids)
    flash(f"{deleted} invoice{'s' if deleted != 1 else ''} deleted.")
    return redirect(back_url())


@app.post("/invoices/reprocess-failed")
def reprocess_failed():
    store().requeue(failed_only=True)
    worker().notify()
    flash("All failed invoices were queued again.")
    return redirect(url_for("invoices"))


@app.get("/batches/<int:batch_id>")
def batch_detail(batch_id):
    summary = store().batch_summary(batch_id)
    if summary is None:
        abort(404)
    _, rows = store().search(batch_id=batch_id, limit=1000)
    return render_template("batch.html", batch=summary, rows=rows, money_columns=MONEY_COLUMNS)


# --- the old Phone camera page: its photo and paste options are on the Upload page now -------------
@app.get("/camera")
def camera():
    return redirect(url_for("upload_page"))


@app.get("/scanner")
def scanner():
    saved = load_settings()
    folder = saved.get("scan_folder", "")
    waiting = imported = None
    if folder and Path(folder).is_dir():
        allowed = config.PDF_EXTENSIONS | config.IMAGE_EXTENSIONS
        waiting = sum(1 for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in allowed)
        done = Path(folder) / "imported"
        imported = sum(1 for p in done.iterdir() if p.is_file()) if done.is_dir() else 0
    return render_template("scanner.html", folder=folder, enabled=bool(saved.get("scan_folder_on")),
                           folder_ok=bool(folder) and Path(folder).is_dir(), waiting=waiting, imported=imported,
                           last=scan_watcher.last_import, api_url=request.host_url.rstrip("/") + url_for("api_parse"))


@app.post("/scanner")
def scanner_save():
    folder = (request.form.get("folder") or "").strip().strip('"')
    enabled = request.form.get("enabled") == "1"
    if enabled and not (folder and Path(folder).is_dir()):
        return _back_to("scanner", f"Folder not found on this computer: {folder or '(empty)'}")
    save_settings(scan_folder=folder, scan_folder_on=enabled)
    flash("Scanner folder is on: new files are imported every few seconds." if enabled
          else "Scanner folder is off.")
    return redirect(url_for("scanner"))


# --- exports -----------------------------------------------------------------------------
def _download(data, name, mimetype):
    stem, ext = Path(name).stem, Path(name).suffix
    return Response(data, mimetype=mimetype, headers={
        "Content-Disposition": f'attachment; filename="{stem}-{workspace()}-{datetime.now():%Y%m%d-%H%M}{ext}"'})


@app.get("/export/invoices.csv")
def export_invoices_csv():
    rows = with_sides(store().export_rows(**_filters()), store().money_rows(), store().own_companies())
    return _download(to_csv(rows, EXPORT_COLUMNS), "invoices.csv", "text/csv")


@app.get("/export/line_items.csv")
def export_line_items_csv():
    return _download(to_csv(store().export_line_items(**_filters()), ITEM_COLUMNS), "line_items.csv", "text/csv")


@app.get("/export/invoices.xlsx")
def export_invoices_xlsx():
    filters = _filters()
    rows = with_sides(store().export_rows(**filters), store().money_rows(), store().own_companies())
    data = build_xlsx(rows, EXPORT_COLUMNS, store().export_line_items(**filters), ITEM_COLUMNS)
    return _download(data, "invoices.xlsx",
                     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# --- JSON API ----------------------------------------------------------------------------------
@app.post("/api/parse")
def api_parse():
    """POST a multipart form with field 'file'. The invoice is saved and returned as JSON."""
    upload_file = request.files.get("file")
    if upload_file is None or not upload_file.filename:
        return jsonify(error="Send the invoice as multipart form field 'file'."), 400
    name = Path(upload_file.filename).name
    if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
        return jsonify(error=f"Unsupported file type '{Path(name).suffix}'. Use PDF or an image."), 422
    invoice_id, is_new = _store_and_process(name, upload_file.read())
    row = store().get(invoice_id)
    if row["status"] == "failed":
        return jsonify(error=row["error"], invoice_id=invoice_id), 422
    if row["status"] != "done":
        return jsonify(invoice_id=invoice_id, status=row["status"]), 202
    result = json.loads(row["result_json"])
    result.update(invoice_id=invoice_id, duplicate=not is_new, review_status=row["review_status"])
    return jsonify(result)


@app.get("/api/invoices")
def api_invoices():
    limit = min(500, max(1, request.args.get("limit", 100, type=int)))
    offset = max(0, request.args.get("offset", 0, type=int))
    total, rows = store().search(**_filters(), limit=limit, offset=offset)
    return jsonify(total=total, limit=limit, offset=offset, items=rows)


@app.get("/api/batches/<int:batch_id>")
def api_batch(batch_id):
    summary = store().batch_summary(batch_id)
    return jsonify(summary) if summary else (jsonify(error="Batch not found."), 404)


@app.get("/health")
def health():
    database = ("mongodb" if hasattr(stores["live"], "client")
                else f"sqlite (MongoDB not reachable: {config.MONGODB_ERROR})" if config.MONGODB_ERROR else "sqlite")
    return jsonify(status="ok", label_matcher=get_parser().matcher.backend, database=database,
                   workers=workers["live"].workers, queue=stores["live"].stats())


@app.errorhandler(413)
def file_too_large(_error):
    return _after_upload(url_for("upload_page"), f"The upload is larger than {config.MAX_UPLOAD_MB} MB. "
                         "For large volumes use the folder import command (see README).")


# --- startup -----------------------------------------------------------------------------
def _port_is_free(port: int) -> bool:
    """True when no other program listens on the port. Uses an exclusive bind so Windows
    cannot let two servers share one port (which would mix up their requests)."""
    for host in ("127.0.0.1", "0.0.0.0"):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
        finally:
            probe.close()
    return True


def start_background_work():
    """Start the worker threads and load the label model in the background."""
    auth.bootstrap_admin_from_env()  # servers: first user from INVOICE_ADMIN_USER / INVOICE_ADMIN_PASSWORD
    for w in workers.values():
        w.start()
    if hasattr(stores["live"], "resume_uploads"):  # MongoDB: files of this computer not stored there yet
        threading.Thread(target=stores["live"].resume_uploads, daemon=True).start()
    scan_watcher.start()  # imports from the scanner folder when one is switched on
    threading.Thread(target=lambda: get_parser().matcher.warm_up(), daemon=True).start()
    threading.Thread(target=fast_ocr.warm_up, daemon=True).start()  # phone-photo OCR models


if __name__ == "__main__":
    # a restarted server (Render, after a crash) finds the port held for a few seconds by the old process:
    # wait for it there instead of stopping, which made the restarts fail again and again
    for _ in range(30 if os.environ.get("RENDER") else 1):
        if _port_is_free(PORT):
            break
        time.sleep(1)
    else:
        sys.exit(f"Port {PORT} is already used by another program. "
                 "Set INVOICE_PARSER_PORT to a free port (for example 5078) and start again.")
    start_background_work()
    print("InvoAgent running.")
    print(f"  This PC:            http://localhost:{PORT}")
    print(f"  Phone (same Wi-Fi): http://<this-PC-IP>:{PORT}")
    print(f"  Data:               " + (f"MongoDB database '{config.MONGODB_DB}'" if hasattr(stores['live'], 'client')
                                       else f"{config.DATA_DIR / 'live'} (SQLite)"))
    print(f"  Worker threads:     {workers['live'].workers}")
    from waitress import serve  # production-grade WSGI server (works on Windows)

    serve(app, host="0.0.0.0", port=PORT, threads=8)
