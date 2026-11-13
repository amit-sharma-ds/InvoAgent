"""Copy this computer's invoices (SQLite in data/live) into MongoDB (MONGODB_URI or data/mongodb_uri.txt).

Invoice numbers (#ids), approvals, corrections, line items and batches are kept, and the original
files go to MongoDB as their smaller copies. Invoices already in MongoDB (the same file) are skipped,
so the script can be run again safely. The SQLite database is not changed: it stays as a backup.

    python scripts/move_to_mongodb.py --dry-run   # only show what would be copied
    python scripts/move_to_mongodb.py
"""
import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from invoice_parser import config  # noqa: E402
from invoice_parser.storage import INVOICE_COLUMNS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="only show what would be copied")
    ap.add_argument("--sqlite", default=str(config.workspace_dir("live") / "invoices.db"))
    args = ap.parse_args()
    if not config.MONGODB_URI:
        sys.exit("No MongoDB connection string: set MONGODB_URI or put it in data/mongodb_uri.txt.")
    if not Path(args.sqlite).exists():
        sys.exit(f"No SQLite database at {args.sqlite}")

    from invoice_parser.mongo_store import MongoInvoiceStore

    con = sqlite3.connect(args.sqlite)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute("SELECT * FROM invoices ORDER BY id")]
    members = {}
    for r in con.execute("SELECT batch_id, invoice_id FROM batch_members"):
        members.setdefault(r["invoice_id"], []).append(r["batch_id"])
    batches = [dict(r) for r in con.execute("SELECT * FROM batches ORDER BY id")]
    items = {}
    for r in con.execute("SELECT * FROM line_items ORDER BY invoice_id, line_no"):
        items.setdefault(r["invoice_id"], []).append(
            {k: r[k] for k in ("line_no", "description", "quantity", "unit_price", "amount")})

    store = MongoInvoiceStore()
    print(f"MongoDB database '{store.db.name}': {store.invoices.count_documents({})} invoices before.")
    new_ids, copied, skipped = {}, [], 0
    batch_ids = {}
    for b in batches:  # batches keep their number unless MongoDB already uses it
        same = store.batches.find_one({"name": b["name"], "created_at": b["created_at"]}, {"id": 1})
        if same:
            batch_ids[b["id"]] = same["id"]
            continue
        bid = b["id"] if not store.batches.find_one({"id": b["id"]}) else store._next_id("batches")
        batch_ids[b["id"]] = bid
        if not args.dry_run:
            store.batches.insert_one({"id": bid, "name": b["name"], "source": b["source"], "created_at": b["created_at"]})
    for row in rows:
        existing = store.invoices.find_one({"file_sha256": row["file_sha256"]}, {"id": 1})
        if existing:
            new_ids[row["id"]] = existing["id"]
            skipped += 1
            continue
        invoice_id = row["id"] if not store.invoices.find_one({"id": row["id"]}) else store._next_id("invoices")
        new_ids[row["id"]] = invoice_id
        doc = {c: row.get(c) for c in INVOICE_COLUMNS}
        doc.update(id=invoice_id, line_items=items.get(row["id"], []), file_stored=False,
                   batch_id=batch_ids.get(row["batch_id"]),
                   batch_ids=sorted({batch_ids[b] for b in members.get(row["id"], []) if b in batch_ids}))
        copied.append(doc)
    for doc in copied:  # a duplicate points at the invoice's number in MongoDB
        if doc["duplicate_of"] is not None:
            doc["duplicate_of"] = new_ids.get(doc["duplicate_of"])
    print(f"{len(copied)} invoices to copy, {skipped} already in MongoDB.")
    for doc in copied:
        print(f"  #{doc['id']:<4} {doc['file_name'][:40]:40} {doc['status']:8} {doc['review_status'] or 'pending':9}"
              f" {doc['invoice_number'] or ''}")
    if args.dry_run or not copied:
        return
    store.invoices.insert_many(copied)
    for name, ids in (("invoices", [d["id"] for d in store.invoices.find({}, {"id": 1})]),
                      ("batches", [b["id"] for b in store.batches.find({}, {"id": 1})])):
        if ids:  # new invoices / batches continue after the highest number
            store.counters.update_one({"_id": name}, {"$max": {"value": max(ids)}}, upsert=True)
    stored = sum(store.upload_file(doc["id"]) for doc in copied)
    print(f"Copied {len(copied)} invoices; {stored} original files stored in MongoDB "
          f"({len(copied) - stored} kept on this computer only).")


if __name__ == "__main__":
    main()
