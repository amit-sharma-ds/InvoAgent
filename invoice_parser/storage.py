"""SQLite storage for parsed invoices.

* invoices      : one row per invoice with fixed columns (NULL when a field was not found)
* line_items    : one row per table row of an invoice
* batches       : uploads / folder imports; batch_members lists every file of a batch,
                  including files that were already in the database (duplicates)
Original files are kept under data/files/, named by their SHA-256, so the same file is
never stored or processed twice.
"""
import hashlib
import json
import re
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from . import config
from . import normalize as nz
from .extract import REQUIRED_FIELDS

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    source TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS invoices (
    id INTEGER PRIMARY KEY,
    batch_id INTEGER REFERENCES batches(id),     -- batch that first added the file
    file_name TEXT NOT NULL,
    file_sha256 TEXT NOT NULL UNIQUE,
    stored_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',       -- queued | processing | done | failed
    error TEXT,
    pages INTEGER,
    method TEXT,
    overall_confidence REAL,
    processing_ms INTEGER,
    invoice_number TEXT,
    invoice_date TEXT,
    due_date TEXT,
    po_number TEXT,
    currency TEXT,
    seller_name TEXT,
    seller_address TEXT,
    seller_tax_id TEXT,
    seller_email TEXT,
    seller_phone TEXT,
    buyer_name TEXT,
    buyer_address TEXT,
    buyer_tax_id TEXT,
    buyer_email TEXT,
    buyer_phone TEXT,
    subtotal REAL,
    tax REAL,
    discount REAL,
    shipping REAL,                               -- shipping / freight charges
    total REAL,
    totals_ok INTEGER,
    line_items_ok INTEGER,
    line_item_count INTEGER,
    needs_review INTEGER,
    warnings TEXT,
    result_json TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    processed_at TEXT,
    review_status TEXT DEFAULT 'pending',       -- pending | approved | rejected
    review_note TEXT,
    reviewed_at TEXT,
    edited INTEGER DEFAULT 0,                    -- 1 when a reviewer corrected fields
    duplicate_of INTEGER,                        -- earlier invoice with the same seller + number
    invoice_key TEXT                             -- invoice number without spaces / punctuation
);
CREATE INDEX IF NOT EXISTS idx_invoices_status ON invoices(status);
CREATE INDEX IF NOT EXISTS idx_invoices_number ON invoices(invoice_number);
CREATE INDEX IF NOT EXISTS idx_invoices_seller ON invoices(seller_name);
CREATE TABLE IF NOT EXISTS batch_members (
    batch_id INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    invoice_id INTEGER NOT NULL REFERENCES invoices(id) ON DELETE CASCADE,
    PRIMARY KEY (batch_id, invoice_id)
);
CREATE TABLE IF NOT EXISTS line_items (
    id INTEGER PRIMARY KEY,
    invoice_id INTEGER NOT NULL REFERENCES invoices(id) ON DELETE CASCADE,
    line_no INTEGER NOT NULL,
    description TEXT,
    quantity REAL,
    unit_price REAL,
    amount REAL
);
CREATE INDEX IF NOT EXISTS idx_items_invoice ON line_items(invoice_id);
"""

# Columns added after the first release: added automatically to older databases.
_ADDED_COLUMNS = {"review_status": "TEXT DEFAULT 'pending'", "review_note": "TEXT", "reviewed_at": "TEXT",
                  "edited": "INTEGER DEFAULT 0", "duplicate_of": "INTEGER", "invoice_key": "TEXT",
                  "shipping": "REAL"}

# Columns shown in the table view (same set for every invoice; missing values are NULL).
TABLE_COLUMNS = [
    "id", "file_name", "status", "needs_review", "review_status", "duplicate_of", "invoice_number",
    "invoice_date", "due_date", "po_number", "seller_name", "seller_tax_id", "buyer_name", "buyer_tax_id",
    "currency", "subtotal", "tax", "discount", "shipping", "total", "line_item_count", "overall_confidence",
]
# Columns written to CSV / Excel exports.
EXPORT_COLUMNS = [
    "id", "file_name", "status", "needs_review", "review_status", "review_note", "reviewed_at", "edited",
    "duplicate_of", "invoice_number", "invoice_date", "due_date", "po_number", "currency", "seller_name",
    "seller_address", "seller_tax_id", "seller_email", "seller_phone", "buyer_name", "buyer_address",
    "buyer_tax_id", "buyer_email", "buyer_phone", "subtotal", "tax", "discount", "shipping", "total", "totals_ok",
    "line_items_ok", "line_item_count", "overall_confidence", "pages", "method", "warnings", "error",
    "processing_ms", "created_at", "processed_at",
]
# Fields a reviewer can correct: column -> (kind, where the value lives in the parser result)
EDITABLE_FIELDS = {
    "invoice_number": ("text", ("invoice_number",)), "invoice_date": ("date", ("invoice_date",)),
    "due_date": ("date", ("due_date",)), "po_number": ("text", ("po_number",)),
    "currency": ("text", ("currency",)), "seller_name": ("text", ("seller", "name")),
    "seller_tax_id": ("text", ("seller", "tax_id")), "buyer_name": ("text", ("buyer", "name")),
    "buyer_tax_id": ("text", ("buyer", "tax_id")), "subtotal": ("amount", ("subtotal",)),
    "tax": ("amount", ("tax",)), "discount": ("amount", ("discount",)),
    "shipping": ("amount", ("shipping",)), "total": ("amount", ("total",)),
}
REVIEW_STATUSES = ("pending", "approved", "rejected")
# Every column of the invoices table (the MongoDB store keeps the same fields).
INVOICE_COLUMNS = [
    "id", "batch_id", "file_name", "file_sha256", "stored_path", "status", "error", "pages", "method",
    "overall_confidence", "processing_ms", "invoice_number", "invoice_date", "due_date", "po_number", "currency",
    "seller_name", "seller_address", "seller_tax_id", "seller_email", "seller_phone", "buyer_name",
    "buyer_address", "buyer_tax_id", "buyer_email", "buyer_phone", "subtotal", "tax", "discount", "shipping",
    "total", "totals_ok", "line_items_ok", "line_item_count", "needs_review", "warnings", "result_json",
    "created_at", "started_at", "processed_at", "review_status", "review_note", "reviewed_at", "edited",
    "duplicate_of", "invoice_key",
]


def open_store():
    """The invoice store of this installation: MongoDB when MONGODB_URI is set (data shared by every
    computer and the server, nothing lost on a restart), else SQLite in data/live."""
    if config.MONGODB_URI:
        from .mongo_store import MongoInvoiceStore

        return MongoInvoiceStore()
    return InvoiceStore.for_workspace("live")


def corrections(row, values):
    """A reviewer's corrections to an invoice row -> the columns that change (with edited, the
    updated parser result and invoice_key), or {} when nothing changes. Raises ValueError for an
    amount or date that cannot be read."""
    result = json.loads(row["result_json"]) if row["result_json"] else {}
    fields = result.setdefault("fields", {})
    changes = {}
    for column, (kind, path) in EDITABLE_FIELDS.items():
        if column not in values:
            continue
        raw = (values[column] or "").strip()
        if kind == "amount":
            new = nz.parse_amount(raw) if raw else None
            if raw and new is None:
                raise ValueError(f"'{raw}' is not an amount.")
        elif kind == "date":
            new = nz.parse_date(raw) if raw else None
            if raw and new is None:
                raise ValueError(f"Could not read the date '{raw}'. Use for example 2026-09-23.")
        else:
            new = raw or None
        if new == row[column]:
            continue
        changes[column] = new
        corrected = {"confidence": 1.0, "source_label": "corrected by reviewer", "source_text": "", "page": None}
        if len(path) == 1:
            fields[path[0]] = {**corrected, "value": new} if new is not None else None
        else:
            party = fields.get(path[0]) or {**corrected, "value": {}}
            party["value"] = {**(party.get("value") or {}), path[1]: new}
            party.update(confidence=1.0, source_label="corrected by reviewer")
            fields[path[0]] = party
    if changes:
        changes.update(edited=1, result_json=json.dumps(result, ensure_ascii=False),
                       invoice_key=invoice_key(changes.get("invoice_number", row["invoice_number"])))
    return changes


def invoice_key(number):
    """'DPNP/2026-27/007' and 'DPNP 2026 27 007' compare equal."""
    return re.sub(r"[^A-Z0-9]", "", (number or "").upper()) or None
ITEM_COLUMNS = ["invoice_id", "invoice_number", "seller_name", "currency", "line_no", "description",
                "quantity", "unit_price", "amount"]
STATUSES = ("queued", "processing", "done", "failed")


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _minutes_ago(minutes):
    return (datetime.now().astimezone() - timedelta(minutes=minutes)).isoformat(timespec="seconds")


def flatten(result: dict) -> dict:
    """Parser result -> fixed database columns (None for anything not found)."""
    fields = result.get("fields") or {}

    def value(name):
        f = fields.get(name)
        return f.get("value") if f else None

    def party(name, key):
        f = fields.get(name)
        return (f.get("value") or {}).get(key) if f else None

    checks = result.get("checks") or {}
    totals_ok = checks.get("subtotal_plus_tax_minus_discount_equals_total")
    items_ok = checks.get("line_items_sum_equals_subtotal")
    conf = result.get("overall_confidence")
    missing_required = any(fields.get(n) is None for n in REQUIRED_FIELDS)
    needs_review = ((conf or 0) < config.REVIEW_CONFIDENCE or totals_ok is False
                    or items_ok is False or missing_required)
    return {
        "pages": result.get("pages"),
        "method": (result.get("extraction") or {}).get("method"),
        "overall_confidence": conf,
        "invoice_number": value("invoice_number"),
        "invoice_date": value("invoice_date"),
        "due_date": value("due_date"),
        "po_number": value("po_number"),
        "currency": value("currency"),
        "seller_name": party("seller", "name"),
        "seller_address": party("seller", "address"),
        "seller_tax_id": party("seller", "tax_id"),
        "seller_email": party("seller", "email"),
        "seller_phone": party("seller", "phone"),
        "buyer_name": party("buyer", "name"),
        "buyer_address": party("buyer", "address"),
        "buyer_tax_id": party("buyer", "tax_id"),
        "buyer_email": party("buyer", "email"),
        "buyer_phone": party("buyer", "phone"),
        "subtotal": value("subtotal"),
        "tax": value("tax"),
        "discount": value("discount"),
        "shipping": value("shipping"),
        "total": value("total"),
        "totals_ok": None if totals_ok is None else int(totals_ok),
        "line_items_ok": None if items_ok is None else int(items_ok),
        "line_item_count": len(result.get("line_items") or []),
        "needs_review": int(needs_review),
        "warnings": "; ".join(result.get("warnings") or []) or None,
        "result_json": json.dumps(result, ensure_ascii=False),
        "invoice_key": invoice_key(value("invoice_number")),
    }


def move_legacy_data_to_demo():
    """Older versions kept everything in data/invoices.db. That data (samples, tests) becomes the
    Demo workspace, so the Live workspace starts empty for real company invoices."""
    old_db, demo = config.DATA_DIR / "invoices.db", config.workspace_dir("demo")
    if not old_db.exists() or (demo / "invoices.db").exists():
        return False
    demo.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        src = config.DATA_DIR / f"invoices.db{suffix}"
        if src.exists():
            shutil.move(str(src), str(demo / src.name))
    if (config.DATA_DIR / "files").exists():
        shutil.move(str(config.DATA_DIR / "files"), str(demo / "files"))
    return True


class InvoiceStore:
    def __init__(self, db_path=None, files_dir=None):
        self.db_path = Path(db_path or config.DB_PATH)
        self.files_dir = Path(files_dir or config.FILES_DIR)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        with self._db() as con:
            existing = {row["name"] for row in con.execute("PRAGMA table_info(invoices)")}
            if existing:  # older database: add the newer columns before the indexes refer to them
                for column, kind in _ADDED_COLUMNS.items():
                    if column not in existing:
                        con.execute(f"ALTER TABLE invoices ADD COLUMN {column} {kind}")
            con.executescript(SCHEMA)
            con.execute("CREATE INDEX IF NOT EXISTS idx_invoices_key ON invoices(invoice_key)")

    @classmethod
    def for_workspace(cls, name):
        folder = config.workspace_dir(name)
        return cls(db_path=folder / "invoices.db", files_dir=folder / "files")

    @contextmanager
    def _db(self):
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")  # readers never block the writer
        con.execute("PRAGMA foreign_keys=ON")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    # --- adding files ------------------------------------------------------------------
    def create_batch(self, name, source="upload"):
        with self._db() as con:
            return con.execute("INSERT INTO batches(name, source, created_at) VALUES (?, ?, ?)",
                               (name, source, _now())).lastrowid

    def add_file(self, file_name, data, batch_id=None, queue=True):
        """Store a file. Returns (invoice_id, is_new). A file whose content is already stored
        is not added again (is_new=False) but is still recorded as part of the batch.
        queue=False keeps a new file out of the background queue so the caller can
        process it right away."""
        sha = hashlib.sha256(data).hexdigest()
        with self._db() as con:
            row = con.execute("SELECT id FROM invoices WHERE file_sha256 = ?", (sha,)).fetchone()
            is_new = row is None
            if is_new:
                rel = f"{sha[:2]}/{sha}{Path(file_name).suffix.lower()}"
                path = self.files_dir / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_bytes(data)
                try:
                    invoice_id = con.execute(
                        "INSERT INTO invoices(batch_id, file_name, file_sha256, stored_path, status,"
                        " created_at, started_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (batch_id, Path(file_name).name, sha, rel, "queued" if queue else "processing",
                         _now(), None if queue else _now())).lastrowid
                except sqlite3.IntegrityError:  # same file added at the same moment elsewhere
                    is_new = False
                    row = con.execute("SELECT id FROM invoices WHERE file_sha256 = ?", (sha,)).fetchone()
            if not is_new:
                invoice_id = row["id"]
            if batch_id:
                con.execute("INSERT OR IGNORE INTO batch_members(batch_id, invoice_id) VALUES (?, ?)",
                            (batch_id, invoice_id))
            return invoice_id, is_new

    def file_path(self, row) -> Path:
        return self.files_dir / row["stored_path"]

    # --- queue -------------------------------------------------------------------------
    def claim_next(self):
        """Atomically take the oldest queued invoice and mark it 'processing'."""
        with self._db() as con:
            row = con.execute(
                "UPDATE invoices SET status = 'processing', started_at = ?"
                " WHERE id = (SELECT id FROM invoices WHERE status = 'queued' ORDER BY id LIMIT 1)"
                " RETURNING id, file_name, stored_path", (_now(),)).fetchone()
        return dict(row) if row else None

    def _find_duplicate(self, invoice_id, row):
        """Another invoice with the same number from the same seller (GSTIN or name)."""
        if not row["invoice_key"] or len(row["invoice_key"]) < 3:
            return None
        with self._db() as con:
            others = con.execute("SELECT id, seller_name, seller_tax_id FROM invoices WHERE invoice_key = ?"
                                 " AND id != ? AND status = 'done' ORDER BY id", (row["invoice_key"], invoice_id))
            for other in others:
                if row["seller_tax_id"] and other["seller_tax_id"]:
                    same = row["seller_tax_id"] == other["seller_tax_id"]
                else:
                    a, b = invoice_key(row["seller_name"]), invoice_key(other["seller_name"])
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
        cols = list(row)
        items = [(invoice_id, n, it.get("description"), it.get("quantity"), it.get("unit_price"),
                  it.get("amount")) for n, it in enumerate(result.get("line_items") or [], 1)]
        with self._db() as con:
            con.execute(f"UPDATE invoices SET {', '.join(c + ' = ?' for c in cols)} WHERE id = ?",
                        [row[c] for c in cols] + [invoice_id])
            con.execute("DELETE FROM line_items WHERE invoice_id = ?", (invoice_id,))
            con.executemany("INSERT INTO line_items(invoice_id, line_no, description, quantity, unit_price, amount)"
                            " VALUES (?, ?, ?, ?, ?, ?)", items)

    def save_error(self, invoice_id, error):
        with self._db() as con:
            con.execute("UPDATE invoices SET status = 'failed', error = ?, needs_review = 1, processed_at = ?"
                        " WHERE id = ?", (error, _now(), invoice_id))

    def requeue(self, invoice_id=None, failed_only=False):
        """Put one invoice (or all failed ones) back in the queue, e.g. after improving the parser."""
        with self._db() as con:
            if invoice_id is not None:
                con.execute("UPDATE invoices SET status = 'queued', error = NULL WHERE id = ?", (invoice_id,))
            elif failed_only:
                con.execute("UPDATE invoices SET status = 'queued', error = NULL WHERE status = 'failed'")

    def reset_stuck(self, older_than_minutes=10):
        """After a crash, files left in 'processing' for too long go back to the queue."""
        with self._db() as con:
            con.execute("UPDATE invoices SET status = 'queued' WHERE status = 'processing'"
                        " AND (started_at IS NULL OR started_at <= ?)", (_minutes_ago(older_than_minutes),))

    # --- reading -----------------------------------------------------------------------
    @staticmethod
    def _where(q=None, status=None, review=None, batch_id=None, review_status=None, alias=""):
        p = f"{alias}." if alias else ""
        conds, args = [], []
        if review_status in REVIEW_STATUSES:
            conds.append(f"COALESCE({p}review_status, 'pending') = ?")
            args.append(review_status)
        if q:
            like = f"%{q.strip()}%"
            conds.append(f"({p}invoice_number LIKE ? OR {p}seller_name LIKE ? OR {p}buyer_name LIKE ?"
                         f" OR {p}file_name LIKE ? OR {p}seller_tax_id LIKE ? OR {p}buyer_tax_id LIKE ?)")
            args += [like] * 6
        if status in STATUSES:
            conds.append(f"{p}status = ?")
            args.append(status)
        if review:
            conds.append(f"{p}needs_review = 1")
        if batch_id:
            conds.append(f"{p}id IN (SELECT invoice_id FROM batch_members WHERE batch_id = ?)")
            args.append(int(batch_id))
        return (" WHERE " + " AND ".join(conds)) if conds else "", args

    def search(self, q=None, status=None, review=None, batch_id=None, review_status=None, limit=50, offset=0):
        where, args = self._where(q, status, review, batch_id, review_status)
        with self._db() as con:
            total = con.execute(f"SELECT COUNT(*) FROM invoices{where}", args).fetchone()[0]
            rows = con.execute(f"SELECT {', '.join(TABLE_COLUMNS)} FROM invoices{where}"
                               " ORDER BY id DESC LIMIT ? OFFSET ?", args + [limit, offset]).fetchall()
        return total, [dict(r) for r in rows]

    def get(self, invoice_id):
        with self._db() as con:
            row = con.execute("SELECT * FROM invoices WHERE id = ?", (invoice_id,)).fetchone()
            if not row:
                return None
            items = con.execute("SELECT * FROM line_items WHERE invoice_id = ? ORDER BY line_no",
                                (invoice_id,)).fetchall()
        out = dict(row)
        out["line_items"] = [dict(i) for i in items]
        return out

    def export_rows(self, q=None, status=None, review=None, batch_id=None, review_status=None):
        where, args = self._where(q, status, review, batch_id, review_status)
        with self._db() as con:
            return [dict(r) for r in con.execute(
                f"SELECT {', '.join(EXPORT_COLUMNS)} FROM invoices{where} ORDER BY id", args)]

    def export_line_items(self, q=None, status=None, review=None, batch_id=None, review_status=None):
        where, args = self._where(q, status, review, batch_id, review_status, alias="i")
        with self._db() as con:
            return [dict(r) for r in con.execute(
                "SELECT li.invoice_id, i.invoice_number, i.seller_name, i.currency, li.line_no, li.description,"
                " li.quantity, li.unit_price, li.amount FROM line_items li JOIN invoices i ON i.id = li.invoice_id"
                f"{where} ORDER BY li.invoice_id, li.line_no", args)]

    def batch_summary(self, batch_id):
        members = "id IN (SELECT invoice_id FROM batch_members WHERE batch_id = ?)"
        with self._db() as con:
            batch = con.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
            if not batch:
                return None
            counts = {s: 0 for s in STATUSES}
            for row in con.execute(f"SELECT status, COUNT(*) AS n FROM invoices WHERE {members} GROUP BY status",
                                   (batch_id,)):
                counts[row["status"]] = row["n"]
            review = con.execute(f"SELECT COUNT(*) FROM invoices WHERE {members} AND needs_review = 1",
                                 (batch_id,)).fetchone()[0]
        total = sum(counts.values())
        finished = counts["done"] + counts["failed"]
        return {**dict(batch), **counts, "total": total, "finished": finished, "needs_review": review,
                "complete": finished == total}

    def list_batches(self, limit=10):
        with self._db() as con:
            ids = [r["id"] for r in con.execute("SELECT id FROM batches ORDER BY id DESC LIMIT ?", (limit,))]
        return [self.batch_summary(i) for i in ids]

    def stats(self):
        with self._db() as con:
            counts = {s: 0 for s in STATUSES}
            for row in con.execute("SELECT status, COUNT(*) AS n FROM invoices GROUP BY status"):
                counts[row["status"]] = row["n"]
            review = con.execute("SELECT COUNT(*) FROM invoices WHERE needs_review = 1").fetchone()[0]
        return {**counts, "total": sum(counts.values()), "needs_review": review}

    # --- review workflow --------------------------------------------------------------
    def update_fields(self, invoice_id, values):
        """A reviewer's corrections: updates the columns and the stored parser result."""
        row = self.get(invoice_id)
        if row is None:
            return None
        changes = corrections(row, values)
        if not changes:
            return row
        cols = list(changes)
        with self._db() as con:
            con.execute(f"UPDATE invoices SET {', '.join(c + ' = ?' for c in cols)} WHERE id = ?",
                        [changes[c] for c in cols] + [invoice_id])
        return self.get(invoice_id)

    def delete(self, invoice_ids):
        """Delete invoices: database rows, line items, batch links and the stored original files.
        Invoices that pointed to a deleted one as 'possible duplicate' lose that pointer.
        Returns the number deleted."""
        ids = sorted({int(i) for i in invoice_ids})
        if not ids:
            return 0
        marks = ",".join("?" * len(ids))
        with self._db() as con:
            rows = con.execute(f"SELECT id, stored_path FROM invoices WHERE id IN ({marks})"
                               " AND status != 'processing'", ids).fetchall()  # never mid-processing
            if not rows:
                return 0
            found = [r["id"] for r in rows]
            marks = ",".join("?" * len(found))
            con.execute(f"DELETE FROM line_items WHERE invoice_id IN ({marks})", found)
            con.execute(f"DELETE FROM batch_members WHERE invoice_id IN ({marks})", found)
            con.execute(f"UPDATE invoices SET duplicate_of = NULL WHERE duplicate_of IN ({marks})", found)
            con.execute(f"DELETE FROM invoices WHERE id IN ({marks})", found)
            paths = [r["stored_path"] for r in rows if r["stored_path"]]
            still_used = {r["stored_path"] for r in con.execute(
                f"SELECT stored_path FROM invoices WHERE stored_path IN ({','.join('?' * len(paths))})", paths)} \
                if paths else set()
        for rel in paths:
            if rel not in still_used:
                (self.files_dir / rel).unlink(missing_ok=True)
        return len(found)

    def set_review(self, invoice_id, status, note=None):
        """Approve / reject (clears the needs-review flag) or put back to pending."""
        if status not in REVIEW_STATUSES:
            raise ValueError(f"Unknown review status '{status}'.")
        with self._db() as con:
            con.execute("UPDATE invoices SET review_status = ?, review_note = ?, reviewed_at = ?,"
                        " needs_review = CASE WHEN ? = 'pending' THEN needs_review ELSE 0 END WHERE id = ?",
                        (status, (note or "").strip() or None, None if status == "pending" else _now(),
                         status, invoice_id))

    # --- dashboards ---------------------------------------------------------------------
    def overview(self, recent=8):
        month = datetime.now().strftime("%Y-%m")
        stats = self.stats()
        with self._db() as con:
            review = {r["s"]: r["n"] for r in con.execute(
                "SELECT COALESCE(review_status, 'pending') AS s, COUNT(*) AS n FROM invoices"
                " WHERE status = 'done' GROUP BY s")}
            duplicates = con.execute("SELECT COUNT(*) FROM invoices WHERE duplicate_of IS NOT NULL").fetchone()[0]
            totals = [dict(r) for r in con.execute(
                "SELECT COALESCE(currency, 'unknown') AS currency, COUNT(*) AS invoices, SUM(total) AS total,"
                " SUM(tax) AS tax, SUM(CASE WHEN substr(created_at, 1, 7) = ? THEN total ELSE 0 END) AS month_total"
                " FROM invoices WHERE status = 'done' AND COALESCE(review_status, 'pending') != 'rejected'"
                " AND (duplicate_of IS NULL OR review_status = 'approved')"  # a copy is not counted twice
                " GROUP BY currency ORDER BY total DESC", (month,))]
            recent_rows = [dict(r) for r in con.execute(
                "SELECT id, file_name, status, needs_review, review_status, duplicate_of, invoice_number,"
                " invoice_date, seller_name, total, currency FROM invoices ORDER BY id DESC LIMIT ?", (recent,))]
        return {**stats, "review": review, "duplicates": duplicates, "totals": totals, "recent": recent_rows}

    # rejected invoices and duplicate copies never count in money totals (an approved "duplicate" does)
    _COUNTED = ("status = 'done' AND COALESCE(review_status, 'pending') != 'rejected'"
                " AND (duplicate_of IS NULL OR review_status = 'approved')")

    def dashboard(self, currency=None, period="12m", today=None):
        """Figures for the Dashboard tab, for ONE currency (amounts in different currencies are never
        added together) and one period of invoice dates.
        period: '12m' last 12 months, 'fy' this financial year (Apr-Mar), 'lfy' last financial year,
        'all' everything (shown per year)."""
        today = today or datetime.now().date()
        with self._db() as con:
            currencies = [dict(r) for r in con.execute(
                f"SELECT currency, COUNT(*) AS n FROM invoices WHERE {self._COUNTED} AND currency IS NOT NULL"
                " GROUP BY currency ORDER BY n DESC, MAX(id) DESC")]  # tie: the latest invoice's currency
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

            where = f"{self._COUNTED} AND currency = ? AND invoice_date BETWEEN ? AND ?"
            args = (currency, start, end)
            bucket = "substr(invoice_date, 1, 7)" if month_keys is not None else "substr(invoice_date, 1, 4)"
            rows = {r["k"]: dict(r) for r in con.execute(
                f"SELECT {bucket} AS k, COUNT(*) AS invoices, SUM(total) AS total, SUM(COALESCE(tax, 0)) AS tax"
                f" FROM invoices WHERE {where} GROUP BY k ORDER BY k", args)}
            keys = month_keys if month_keys is not None else sorted(rows)
            buckets = []
            for k in keys:
                r = rows.get(k, {"invoices": 0, "total": 0.0, "tax": 0.0})
                total, tax = r["total"] or 0.0, r["tax"] or 0.0
                buckets.append({"key": k, "invoices": r["invoices"], "total": round(total, 2),
                                "tax": round(tax, 2), "taxable": round(total - tax, 2)})
            vendors = [dict(r) for r in con.execute(
                "SELECT COALESCE(NULLIF(seller_tax_id, ''), UPPER(TRIM(seller_name))) AS vendor_key,"
                " MAX(seller_name) AS name, MAX(seller_tax_id) AS gstin, COUNT(*) AS invoices,"
                " SUM(total) AS total, SUM(COALESCE(tax, 0)) AS tax"
                f" FROM invoices WHERE {where} AND seller_name IS NOT NULL"
                " GROUP BY vendor_key ORDER BY total DESC LIMIT 10", args)]
            summary = dict(con.execute(
                "SELECT COUNT(*) AS invoices, SUM(total) AS total, SUM(COALESCE(tax, 0)) AS tax,"
                " COUNT(DISTINCT COALESCE(NULLIF(seller_tax_id, ''), UPPER(TRIM(seller_name)))) AS vendors"
                f" FROM invoices WHERE {where}", args).fetchone())
            undated = con.execute(f"SELECT COUNT(*) FROM invoices WHERE {self._COUNTED} AND currency = ?"
                                  " AND invoice_date IS NULL", (currency,)).fetchone()[0]
        return {"currencies": currencies, "currency": currency, "period": period, "buckets": buckets,
                "vendors": vendors, "summary": summary, "undated": undated, "by_year": month_keys is None}

    def vendors(self):
        """One row per seller (by GSTIN, else by name) and currency."""
        with self._db() as con:
            return [dict(r) for r in con.execute(
                "SELECT COALESCE(NULLIF(seller_tax_id, ''), UPPER(TRIM(seller_name))) AS vendor_key,"
                " MAX(seller_name) AS name, MAX(seller_tax_id) AS gstin, currency, COUNT(*) AS invoices,"
                " SUM(CASE WHEN (duplicate_of IS NULL OR review_status = 'approved') AND COALESCE(review_status, 'pending') != 'rejected' THEN total END) AS total,"
                " SUM(CASE WHEN (duplicate_of IS NULL OR review_status = 'approved') AND COALESCE(review_status, 'pending') != 'rejected' THEN tax END) AS tax, MAX(invoice_date) AS last_invoice,"
                " SUM(CASE WHEN needs_review = 1 THEN 1 ELSE 0 END) AS needs_review,"
                " SUM(CASE WHEN duplicate_of IS NOT NULL THEN 1 ELSE 0 END) AS duplicates"
                " FROM invoices WHERE status = 'done' AND seller_name IS NOT NULL"
                " GROUP BY vendor_key, currency ORDER BY total DESC")]
