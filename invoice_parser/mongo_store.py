"""MongoDB storage for parsed invoices, with the same methods as storage.InvoiceStore (SQLite).

Used when MONGODB_URI is set (config.py), so every computer and the Render server see the same
invoices, approvals and corrections, and a restart of the server loses nothing.

Database MONGODB_DB (default "invoagent"):
* invoices : one document per invoice with the same fields as the SQLite columns, its line items
             inside it, and batch_ids (the uploads that listed the file)
* batches  : uploads / folder imports
* counters : the last invoice / batch number, so invoices keep short numbers (#1, #2 ...)
* files    : GridFS bucket with the original files, named by their SHA-256

Original files: a photo is kept as a JPEG of at most 2600 px (quality 90, camera details and
rotation kept), so the free 512 MB cluster holds about a thousand invoices; a PDF is kept as it is.
The invoice is always read from the full original (a local copy in files_dir); the smaller copy is
uploaded in the background: after the invoice is read for a single upload (so it never takes CPU
from the reading), at once for files waiting in a batch, and at start-up for anything missed. Nothing is uploaded once the whole cluster would
pass MONGODB_FILES_LIMIT_MB, so the cluster's other databases never run out of space; the invoice
data is still saved and the original stays on the computer that received it.
"""
import hashlib
import io
import logging
import mimetypes
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path

from . import config
from .storage import (EXPORT_COLUMNS, INVOICE_COLUMNS, REVIEW_STATUSES, STATUSES, TABLE_COLUMNS, _minutes_ago,
                      _now, corrections, flatten, invoice_key)

log = logging.getLogger(__name__)

KEEP_SIDE = 2600     # longest side of a stored photo (the photo reader works on 2600 px)
JPEG_QUALITY = 90
SYSTEM_DATABASES = {"admin", "local", "config"}

# rejected invoices and duplicate copies never count in money totals (an approved "duplicate" does)
_COUNTED = {"status": "done", "review_status": {"$ne": "rejected"},
            "$or": [{"duplicate_of": None}, {"review_status": "approved"}]}
_COUNTED_EXPR = {"$and": [{"$ne": [{"$ifNull": ["$review_status", "pending"]}, "rejected"]},
                          {"$or": [{"$eq": [{"$ifNull": ["$duplicate_of", None]}, None]},
                                   {"$eq": ["$review_status", "approved"]}]}]}
# a seller: its GSTIN, else its name in capitals
_VENDOR_KEY = {"$cond": [{"$ne": [{"$ifNull": ["$seller_tax_id", ""]}, ""]}, "$seller_tax_id",
                         {"$toUpper": {"$trim": {"input": {"$ifNull": ["$seller_name", ""]}}}}]}


def _and(*conds):
    conds = [c for c in conds if c]
    return {"$and": conds} if len(conds) > 1 else (conds[0] if conds else {})


def _sum(expr):
    """$sum plus a count of the values that are present, so an all-empty sum can be None (like SQL)."""
    return {"$sum": expr}, {"$sum": {"$cond": [{"$ne": [{"$ifNull": [expr, None]}, None]}, 1, 0]}}


def _none_if_empty(doc, *names):
    for name in names:
        if not doc.pop(f"{name}_n", 0):
            doc[name] = None
    return doc


def _by_total_desc(rows):
    return sorted(rows, key=lambda r: (r["total"] is None, -(r["total"] or 0)))


def compact_file(data: bytes, file_name: str):
    """(bytes, content type) to keep in the database: a photo or image as a JPEG of at most
    KEEP_SIDE px with its EXIF (camera details, rotation), when that is smaller; anything else as it is."""
    ext = Path(file_name).suffix.lower()
    guessed = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
    if ext not in config.IMAGE_EXTENSIONS or ext in (".tif", ".tiff"):  # PDFs, multi-page TIFFs
        return data, guessed
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            exif = im.info.get("exif") or b""
            im.load()
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                rgba = im.convert("RGBA")
                im = Image.new("RGB", rgba.size, "white")
                im.paste(rgba, mask=rgba.split()[-1])
            elif im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            if max(im.size) > KEEP_SIDE:
                im.thumbnail((KEEP_SIDE, KEEP_SIDE), Image.LANCZOS)
            out = io.BytesIO()
            im.save(out, "JPEG", quality=JPEG_QUALITY, optimize=True, exif=exif)
        small = out.getvalue()
    except Exception:
        return data, guessed
    return (small, "image/jpeg") if len(small) < len(data) else (data, guessed)


class MongoInvoiceStore:
    def __init__(self, uri=None, db_name=None, files_dir=None, files_limit_mb=None, client=None):
        import gridfs
        from pymongo import MongoClient

        self.client = client or MongoClient(uri or config.MONGODB_URI, serverSelectionTimeoutMS=20000,
                                            appname="InvoAgent")
        self.db = self.client[db_name or config.MONGODB_DB]
        self.files_dir = Path(files_dir or config.FILES_DIR)  # local copies of the originals
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.files_limit = (files_limit_mb or config.MONGODB_FILES_LIMIT_MB) * 2**20
        self.invoices, self.batches, self.counters = self.db.invoices, self.db.batches, self.db.counters
        self.fs = gridfs.GridFSBucket(self.db, bucket_name="files")
        for field, unique in (("id", True), ("file_sha256", True), ("status", False), ("invoice_key", False),
                              ("batch_ids", False)):
            self.invoices.create_index(field, unique=unique)
        self.batches.create_index("id", unique=True)
        self._uploads = queue.Queue()
        self._uploader = None
        self._uploader_lock = threading.Lock()
        self._usage = (0.0, 0)  # (when measured, bytes used by the whole cluster)

    def _next_id(self, name):
        from pymongo import ReturnDocument

        return self.counters.find_one_and_update({"_id": name}, {"$inc": {"value": 1}}, upsert=True,
                                                 return_document=ReturnDocument.AFTER)["value"]

    @staticmethod
    def _row(doc, columns=INVOICE_COLUMNS):
        return {c: doc.get(c) for c in columns}

    # --- adding files ------------------------------------------------------------------
    def create_batch(self, name, source="upload"):
        batch_id = self._next_id("batches")
        self.batches.insert_one({"id": batch_id, "name": name, "source": source, "created_at": _now()})
        return batch_id

    def add_file(self, file_name, data, batch_id=None, queue=True):
        """Store a file. Returns (invoice_id, is_new). A file whose content is already stored is not
        added again (is_new=False) but is still recorded as part of the batch. queue=False keeps a
        new file out of the background queue so the caller can process it right away."""
        from pymongo.errors import DuplicateKeyError

        sha = hashlib.sha256(data).hexdigest()
        existing = self.invoices.find_one({"file_sha256": sha}, {"id": 1})
        is_new = existing is None
        if is_new:
            rel = f"{sha[:2]}/{sha}{Path(file_name).suffix.lower()}"
            path = self.files_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(data)
            doc = dict.fromkeys(INVOICE_COLUMNS)
            doc.update(id=self._next_id("invoices"), batch_id=batch_id, file_name=Path(file_name).name,
                       file_sha256=sha, stored_path=rel, status="queued" if queue else "processing",
                       created_at=_now(), started_at=None if queue else _now(), review_status="pending",
                       edited=0, line_items=[], batch_ids=[batch_id] if batch_id else [], file_stored=False)
            try:
                self.invoices.insert_one(doc)
            except DuplicateKeyError:  # the same file added at the same moment elsewhere
                is_new = False
                existing = self.invoices.find_one({"file_sha256": sha}, {"id": 1})
            else:
                if queue:  # waits in a batch: store it now (a single upload: after it is read)
                    self._queue_upload(doc["id"])
                return doc["id"], True
        if batch_id:
            self.invoices.update_one({"id": existing["id"]}, {"$addToSet": {"batch_ids": batch_id}})
        return existing["id"], False

    def file_path(self, row) -> Path:
        """The local copy of the invoice's file, fetched from the database when this computer has none.
        The path does not exist when the file is in neither place."""
        path = self.files_dir / row["stored_path"]
        if not path.exists():
            sha = row.get("file_sha256") or Path(row["stored_path"]).stem
            try:
                data = self.fs.open_download_stream_by_name(sha).read()
            except Exception:  # gridfs.errors.NoFile: not stored (database space limit)
                return path
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(path)
        return path

    # --- original files to the database (background) ----------------------------------------
    def _queue_upload(self, invoice_id):
        self._uploads.put(invoice_id)
        with self._uploader_lock:
            if self._uploader is None or not self._uploader.is_alive():
                self._uploader = threading.Thread(target=self._upload_loop, name="invoice-file-upload", daemon=True)
                self._uploader.start()

    def _upload_loop(self):
        while True:
            invoice_id = self._uploads.get()
            try:
                self.upload_file(invoice_id)
            except Exception:
                log.exception("Could not store the file of invoice %s in MongoDB (kept on this computer)",
                              invoice_id)
            finally:
                self._uploads.task_done()

    def wait_for_uploads(self):
        """Block until every queued file is in the database (scripts and tests)."""
        self._uploads.join()

    def upload_file(self, invoice_id):
        """Put the invoice's file (the smaller copy) in GridFS. Returns True when the file is stored."""
        doc = self.invoices.find_one({"id": invoice_id}, {"stored_path": 1, "file_sha256": 1, "file_name": 1,
                                                          "file_stored": 1})
        if doc is None or doc.get("file_stored"):
            return bool(doc)
        path = self.files_dir / doc["stored_path"]
        if not path.exists():
            return False
        data, content_type = compact_file(path.read_bytes(), doc["file_name"])
        sha = doc["file_sha256"]
        if not self._in_gridfs(sha):
            if self._cluster_bytes() + len(data) > self.files_limit:
                log.warning("MongoDB is nearly full: the file of invoice %s stays on this computer only", invoice_id)
                self.invoices.update_one({"id": invoice_id}, {"$set": {"file_note": "not stored: database full"}})
                return False
            self.fs.upload_from_stream(sha, data, metadata={"invoice_id": invoice_id, "file_name": doc["file_name"],
                                                            "content_type": content_type})
            self._usage = (self._usage[0], self._usage[1] + len(data))
        self.invoices.update_one({"id": invoice_id}, {"$set": {"file_stored": True, "stored_bytes": len(data)},
                                                      "$unset": {"file_note": ""}})
        return True

    def resume_uploads(self):
        """At start-up: queue the files of this computer that are not in the database yet."""
        for doc in self.invoices.find({"file_stored": {"$ne": True}}, {"id": 1, "stored_path": 1}):
            if (self.files_dir / doc["stored_path"]).exists():
                self._queue_upload(doc["id"])

    def _in_gridfs(self, sha):
        return self.db["files.files"].find_one({"filename": sha}, {"_id": 1}) is not None

    def _cluster_bytes(self):
        """Space used by every database of the cluster (measured at most once a minute)."""
        measured, used = self._usage
        if time.time() - measured > 60:
            used = 0
            for d in self.client.list_databases():
                size = int(d.get("sizeOnDisk") or 0)
                if d["name"] not in SYSTEM_DATABASES:
                    try:
                        s = self.client[d["name"]].command("dbStats")
                        size = max(size, int(s.get("dataSize", 0)) + int(s.get("indexSize", 0)))
                    except Exception:
                        pass
                used += size
            self._usage = (time.time(), used)
        return used

    def _delete_file(self, sha, stored_path):
        (self.files_dir / stored_path).unlink(missing_ok=True)
        for f in self.db["files.files"].find({"filename": sha}, {"_id": 1}):
            self.fs.delete(f["_id"])

    # --- queue -------------------------------------------------------------------------
    def claim_next(self):
        """Atomically take the oldest queued invoice and mark it 'processing'."""
        from pymongo import ReturnDocument

        doc = self.invoices.find_one_and_update(
            {"status": "queued"}, {"$set": {"status": "processing", "started_at": _now()}}, sort=[("id", 1)],
            projection={"_id": 0, "id": 1, "file_name": 1, "stored_path": 1, "file_sha256": 1},
            return_document=ReturnDocument.AFTER)
        return doc

    def _find_duplicate(self, invoice_id, row):
        """Another invoice with the same number from the same seller (GSTIN or name)."""
        if not row["invoice_key"] or len(row["invoice_key"]) < 3:
            return None
        others = self.invoices.find({"invoice_key": row["invoice_key"], "id": {"$ne": invoice_id}, "status": "done"},
                                    {"id": 1, "seller_name": 1, "seller_tax_id": 1}).sort("id", 1)
        for other in others:
            if row["seller_tax_id"] and other.get("seller_tax_id"):
                same = row["seller_tax_id"] == other["seller_tax_id"]
            else:
                a, b = invoice_key(row["seller_name"]), invoice_key(other.get("seller_name"))
                same = not a or not b or a == b
            if same:
                return other["id"]
        return None

    def save_result(self, invoice_id, result, processing_ms):
        row = flatten(result)
        duplicate = self._find_duplicate(invoice_id, row)
        if duplicate:
            result.setdefault("warnings", []).append(
                f"Possible duplicate of invoice #{duplicate}: same seller and invoice number.")
            result["duplicate_of"] = duplicate
            row = flatten(result)
            row["needs_review"] = 1
        row.update(status="done", error=None, processing_ms=processing_ms, processed_at=_now(),
                   duplicate_of=duplicate, review_status="pending", review_note=None, reviewed_at=None, edited=0)
        items = [{"line_no": n, "description": it.get("description"), "quantity": it.get("quantity"),
                  "unit_price": it.get("unit_price"), "amount": it.get("amount")}
                 for n, it in enumerate(result.get("line_items") or [], 1)]
        self.invoices.update_one({"id": invoice_id}, {"$set": {**row, "line_items": items}})
        self._queue_upload(invoice_id)  # nothing happens when the file is already stored

    def save_error(self, invoice_id, error):
        self.invoices.update_one({"id": invoice_id}, {"$set": {"status": "failed", "error": error, "needs_review": 1,
                                                               "processed_at": _now()}})
        self._queue_upload(invoice_id)

    def requeue(self, invoice_id=None, failed_only=False):
        """Put one invoice (or all failed ones) back in the queue, e.g. after improving the parser."""
        if invoice_id is not None:
            self.invoices.update_one({"id": invoice_id}, {"$set": {"status": "queued", "error": None}})
        elif failed_only:
            self.invoices.update_many({"status": "failed"}, {"$set": {"status": "queued", "error": None}})

    def reset_stuck(self, older_than_minutes=10):
        """After a crash, files left in 'processing' for too long go back to the queue."""
        self.invoices.update_many({"status": "processing", "$or": [
            {"started_at": None}, {"started_at": {"$lte": _minutes_ago(older_than_minutes)}}]},
            {"$set": {"status": "queued"}})

    # --- reading -----------------------------------------------------------------------
    @staticmethod
    def _match(q=None, status=None, review=None, batch_id=None, review_status=None):
        m = {}
        if review_status in REVIEW_STATUSES:
            m["review_status"] = {"$in": [None, "pending"]} if review_status == "pending" else review_status
        if q:
            like = {"$regex": re.escape(q.strip()), "$options": "i"}
            m["$or"] = [{f: like} for f in ("invoice_number", "seller_name", "buyer_name", "file_name",
                                             "seller_tax_id", "buyer_tax_id")]
        if status in STATUSES:
            m["status"] = status
        if review:
            m["needs_review"] = 1
        if batch_id:
            m["batch_ids"] = int(batch_id)
        return m

    @staticmethod
    def _projection(columns):
        return {"_id": 0, **{c: 1 for c in columns}}

    def search(self, q=None, status=None, review=None, batch_id=None, review_status=None, limit=50, offset=0):
        m = self._match(q, status, review, batch_id, review_status)
        total = self.invoices.count_documents(m)
        docs = self.invoices.find(m, self._projection(TABLE_COLUMNS)).sort("id", -1).skip(offset).limit(limit)
        return total, [self._row(d, TABLE_COLUMNS) for d in docs]

    def get(self, invoice_id):
        doc = self.invoices.find_one({"id": invoice_id}, {"_id": 0})
        if not doc:
            return None
        out = self._row(doc)
        out["line_items"] = [{"id": None, "invoice_id": invoice_id, **it}
                             for it in sorted(doc.get("line_items") or [], key=lambda it: it["line_no"])]
        return out

    def export_rows(self, q=None, status=None, review=None, batch_id=None, review_status=None):
        m = self._match(q, status, review, batch_id, review_status)
        return [self._row(d, EXPORT_COLUMNS)
                for d in self.invoices.find(m, self._projection(EXPORT_COLUMNS)).sort("id", 1)]

    def export_line_items(self, q=None, status=None, review=None, batch_id=None, review_status=None):
        m = self._match(q, status, review, batch_id, review_status)
        out = []
        fields = {"_id": 0, "id": 1, "invoice_number": 1, "seller_name": 1, "currency": 1, "line_items": 1}
        for d in self.invoices.find(m, fields).sort("id", 1):
            for it in sorted(d.get("line_items") or [], key=lambda it: it["line_no"]):
                out.append({"invoice_id": d["id"], "invoice_number": d.get("invoice_number"),
                            "seller_name": d.get("seller_name"), "currency": d.get("currency"),
                            "line_no": it["line_no"], "description": it.get("description"),
                            "quantity": it.get("quantity"), "unit_price": it.get("unit_price"),
                            "amount": it.get("amount")})
        return out

    def _status_counts(self, match):
        counts = dict.fromkeys(STATUSES, 0)
        for r in self.invoices.aggregate([{"$match": match}, {"$group": {"_id": "$status", "n": {"$sum": 1}}}]):
            counts[r["_id"]] = r["n"]
        return counts

    def batch_summary(self, batch_id):
        batch = self.batches.find_one({"id": batch_id}, {"_id": 0})
        if not batch:
            return None
        counts = self._status_counts({"batch_ids": batch_id})
        review = self.invoices.count_documents({"batch_ids": batch_id, "needs_review": 1})
        total = sum(counts.values())
        finished = counts["done"] + counts["failed"]
        return {**batch, **counts, "total": total, "finished": finished, "needs_review": review,
                "complete": finished == total}

    def list_batches(self, limit=10):
        ids = [b["id"] for b in self.batches.find({}, {"id": 1}).sort("id", -1).limit(limit)]
        return [self.batch_summary(i) for i in ids]

    def stats(self):
        counts = self._status_counts({})
        review = self.invoices.count_documents({"needs_review": 1})
        return {**counts, "total": sum(counts.values()), "needs_review": review}

    # --- review workflow --------------------------------------------------------------
    def update_fields(self, invoice_id, values):
        """A reviewer's corrections: updates the fields and the stored parser result."""
        row = self.get(invoice_id)
        if row is None:
            return None
        changes = corrections(row, values)
        if not changes:
            return row
        self.invoices.update_one({"id": invoice_id}, {"$set": changes})
        return self.get(invoice_id)

    def delete(self, invoice_ids):
        """Delete invoices with their line items and stored files. Invoices that pointed to a deleted
        one as 'possible duplicate' lose that pointer. Returns the number deleted."""
        ids = sorted({int(i) for i in invoice_ids})
        if not ids:
            return 0
        docs = list(self.invoices.find({"id": {"$in": ids}, "status": {"$ne": "processing"}},  # never mid-processing
                                       {"id": 1, "stored_path": 1, "file_sha256": 1}))
        if not docs:
            return 0
        found = [d["id"] for d in docs]
        self.invoices.delete_many({"id": {"$in": found}})
        self.invoices.update_many({"duplicate_of": {"$in": found}}, {"$set": {"duplicate_of": None}})
        for d in docs:
            if d.get("stored_path") and not self.invoices.find_one({"file_sha256": d["file_sha256"]}, {"_id": 1}):
                self._delete_file(d["file_sha256"], d["stored_path"])
        return len(found)

    def set_review(self, invoice_id, status, note=None):
        """Approve / reject (clears the needs-review flag) or put back to pending."""
        if status not in REVIEW_STATUSES:
            raise ValueError(f"Unknown review status '{status}'.")
        update = {"review_status": status, "review_note": (note or "").strip() or None,
                  "reviewed_at": None if status == "pending" else _now()}
        if status != "pending":
            update["needs_review"] = 0
        self.invoices.update_one({"id": invoice_id}, {"$set": update})

    # --- dashboards ---------------------------------------------------------------------
    def overview(self, recent=8):
        month = datetime.now().strftime("%Y-%m")
        stats = self.stats()
        review = {r["_id"]: r["n"] for r in self.invoices.aggregate([
            {"$match": {"status": "done"}},
            {"$group": {"_id": {"$ifNull": ["$review_status", "pending"]}, "n": {"$sum": 1}}}])}
        duplicates = self.invoices.count_documents({"duplicate_of": {"$ne": None}})
        total, total_n = _sum("$total")
        tax, tax_n = _sum("$tax")
        totals = [_none_if_empty({"currency": r.pop("_id"), **r}, "total", "tax") for r in self.invoices.aggregate([
            {"$match": _COUNTED},
            {"$group": {"_id": {"$ifNull": ["$currency", "unknown"]}, "invoices": {"$sum": 1},
                        "total": total, "total_n": total_n, "tax": tax, "tax_n": tax_n,
                        "month_total": {"$sum": {"$cond": [
                            {"$eq": [{"$substrCP": [{"$ifNull": ["$created_at", ""]}, 0, 7]}, month]},
                            {"$ifNull": ["$total", 0]}, 0]}}}}])]
        recent_cols = ["id", "file_name", "status", "needs_review", "review_status", "duplicate_of", "invoice_number",
                       "invoice_date", "seller_name", "total", "currency"]
        recent_rows = [self._row(d, recent_cols) for d in
                       self.invoices.find({}, self._projection(recent_cols)).sort("id", -1).limit(recent)]
        return {**stats, "review": review, "duplicates": duplicates, "totals": _by_total_desc(totals),
                "recent": recent_rows}

    def dashboard(self, currency=None, period="12m", today=None):
        """Figures for the Dashboard tab, for ONE currency (amounts in different currencies are never
        added together) and one period of invoice dates (see InvoiceStore.dashboard)."""
        today = today or datetime.now().date()
        currencies = [{"currency": r["_id"], "n": r["n"]} for r in self.invoices.aggregate([
            {"$match": _and(_COUNTED, {"currency": {"$ne": None}})},
            {"$group": {"_id": "$currency", "n": {"$sum": 1}, "last": {"$max": "$id"}}},
            {"$sort": {"n": -1, "last": -1}}])]  # tie: the latest invoice's currency
        if not currencies:
            return {"currencies": [], "currency": None, "period": period, "buckets": [], "vendors": [],
                    "summary": None, "undated": 0}
        if currency not in {c["currency"] for c in currencies}:
            currency = currencies[0]["currency"]

        fy_start = today.year if today.month >= 4 else today.year - 1  # Indian financial year
        month_keys = None
        if period == "fy":
            start, end = f"{fy_start}-04-01", f"{fy_start + 1}-03-31"
            month_keys = [f"{fy_start + (m > 12)}-{(m - 1) % 12 + 1:02d}" for m in range(4, 16)]
        elif period == "lfy":
            start, end = f"{fy_start - 1}-04-01", f"{fy_start}-03-31"
            month_keys = [f"{fy_start - 1 + (m > 12)}-{(m - 1) % 12 + 1:02d}" for m in range(4, 16)]
        elif period == "all":
            start, end = "0000-01-01", "9999-12-31"
        else:
            period = "12m"
            y, m = today.year, today.month
            month_keys = []
            for _ in range(12):
                month_keys.insert(0, f"{y}-{m:02d}")
                y, m = (y, m - 1) if m > 1 else (y - 1, 12)
            start, end = f"{month_keys[0]}-01", f"{today.year}-{today.month:02d}-31"

        where = _and(_COUNTED, {"currency": currency, "invoice_date": {"$gte": start, "$lte": end}})
        width = 7 if month_keys is not None else 4
        tax_sum = {"$sum": {"$ifNull": ["$tax", 0]}}
        rows = {r["_id"]: r for r in self.invoices.aggregate([
            {"$match": where},
            {"$group": {"_id": {"$substrCP": ["$invoice_date", 0, width]}, "invoices": {"$sum": 1},
                        "total": {"$sum": "$total"}, "tax": tax_sum}}])}
        keys = month_keys if month_keys is not None else sorted(rows)
        buckets = []
        for k in keys:
            r = rows.get(k, {"invoices": 0, "total": 0.0, "tax": 0.0})
            total, tax = r["total"] or 0.0, r["tax"] or 0.0
            buckets.append({"key": k, "invoices": r["invoices"], "total": round(total, 2),
                            "tax": round(tax, 2), "taxable": round(total - tax, 2)})
        total, total_n = _sum("$total")
        vendors = _by_total_desc([_none_if_empty({"vendor_key": r.pop("_id"), **r}, "total")
                                  for r in self.invoices.aggregate([
                                      {"$match": _and(where, {"seller_name": {"$ne": None}})},
                                      {"$group": {"_id": _VENDOR_KEY, "name": {"$max": "$seller_name"},
                                                  "gstin": {"$max": "$seller_tax_id"}, "invoices": {"$sum": 1},
                                                  "total": total, "total_n": total_n, "tax": tax_sum}}])])[:10]
        summary = next(self.invoices.aggregate([
            {"$match": where},
            {"$group": {"_id": None, "invoices": {"$sum": 1}, "total": total, "total_n": total_n, "tax": tax_sum,
                        "keys": {"$addToSet": _VENDOR_KEY}}}]), None)
        if summary:
            summary = _none_if_empty({"invoices": summary["invoices"], "total": summary["total"],
                                      "total_n": summary["total_n"], "tax": summary["tax"],
                                      "vendors": len(summary["keys"])}, "total")
        else:
            summary = {"invoices": 0, "total": None, "tax": None, "vendors": 0}
        undated = self.invoices.count_documents(_and(_COUNTED, {"currency": currency, "invoice_date": None}))
        return {"currencies": currencies, "currency": currency, "period": period, "buckets": buckets,
                "vendors": vendors, "summary": summary, "undated": undated, "by_year": month_keys is None}

    def vendors(self):
        """One row per seller (by GSTIN, else by name) and currency."""
        total, total_n = _sum({"$cond": [_COUNTED_EXPR, "$total", None]})
        tax, tax_n = _sum({"$cond": [_COUNTED_EXPR, "$tax", None]})
        rows = []
        for r in self.invoices.aggregate([
                {"$match": {"status": "done", "seller_name": {"$ne": None}}},
                {"$group": {"_id": {"key": _VENDOR_KEY, "currency": "$currency"}, "name": {"$max": "$seller_name"},
                            "gstin": {"$max": "$seller_tax_id"}, "invoices": {"$sum": 1},
                            "total": total, "total_n": total_n, "tax": tax, "tax_n": tax_n,
                            "last_invoice": {"$max": "$invoice_date"},
                            "needs_review": {"$sum": {"$cond": [{"$eq": ["$needs_review", 1]}, 1, 0]}},
                            "duplicates": {"$sum": {"$cond": [
                                {"$ne": [{"$ifNull": ["$duplicate_of", None]}, None]}, 1, 0]}}}}]):
            key = r.pop("_id")
            rows.append(_none_if_empty({"vendor_key": key["key"], "currency": key.get("currency"), **r},
                                       "total", "tax"))
        return _by_total_desc(rows)
