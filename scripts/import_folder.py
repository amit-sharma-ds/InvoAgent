#!/usr/bin/env python
"""Import a folder of invoices into the database and process them in parallel.

Built for large volumes (hundreds to thousands of files):
  * PDFs, images and ZIP archives are found recursively.
  * Files already in the database (same content) are not processed again.
  * N worker threads process the rest; one bad file never stops the run.
  * Safe to stop with Ctrl+C and run again: finished files are kept, the rest continue.

Usage (from the project folder, with the venv active):
    python scripts/import_folder.py "D:\\Invoices\\2024"
    python scripts/import_folder.py "D:\\Invoices\\2024" --workers 6 --export "D:\\Invoices\\2024.xlsx"
"""
import argparse
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from invoice_parser import config  # noqa: E402
from invoice_parser.exporting import build_xlsx, to_csv  # noqa: E402
from invoice_parser.pipeline import get_parser  # noqa: E402
from invoice_parser.storage import EXPORT_COLUMNS, ITEM_COLUMNS, open_store  # noqa: E402
from invoice_parser.worker import BatchWorker  # noqa: E402

ALLOWED = config.PDF_EXTENSIONS | config.IMAGE_EXTENSIONS


def iter_files(folder: Path):
    """Yield (file_name, bytes) for every supported file, including files inside ZIPs."""
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in ALLOWED:
            yield path.name, path.read_bytes()
        elif ext == ".zip":
            try:
                with zipfile.ZipFile(path) as zf:
                    for info in zf.infolist():
                        name = Path(info.filename).name
                        if (not info.is_dir() and name and not name.startswith("._")
                                and Path(name).suffix.lower() in ALLOWED):
                            yield name, zf.read(info)
            except zipfile.BadZipFile:
                print(f"  skipped damaged ZIP: {path}")


def export(store, batch_id, target: Path):
    rows = store.export_rows(batch_id=batch_id)
    if target.suffix.lower() == ".csv":
        target.write_bytes(to_csv(rows, EXPORT_COLUMNS))
        items = target.with_name(target.stem + "_line_items.csv")
        items.write_bytes(to_csv(store.export_line_items(batch_id=batch_id), ITEM_COLUMNS))
        print(f"Exported {len(rows)} invoices to {target} and line items to {items}")
    else:
        target.write_bytes(build_xlsx(rows, EXPORT_COLUMNS, store.export_line_items(batch_id=batch_id), ITEM_COLUMNS))
        print(f"Exported {len(rows)} invoices (+ line items sheet) to {target}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="folder with invoices (searched recursively)")
    ap.add_argument("--workers", type=int, default=config.WORKERS,
                    help=f"parallel worker threads (default {config.WORKERS})")
    ap.add_argument("--batch-name", help="name shown in the web app (default: folder name)")
    ap.add_argument("--export", help="write the results to this .xlsx or .csv file when finished")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"Folder not found: {folder}")

    store = open_store()  # MongoDB when configured, else SQLite in data/live
    batch_id = store.create_batch(args.batch_name or folder.name, source=str(folder.resolve()))
    new = old = 0
    for name, data in iter_files(folder):
        _, is_new = store.add_file(name, data, batch_id=batch_id)
        new += is_new
        old += not is_new
    print(f"Batch #{batch_id}: {new} new files queued, {old} already in the database.")

    summary = store.batch_summary(batch_id)
    if not summary["complete"]:
        store.reset_stuck(older_than_minutes=0)  # files left 'processing' by an interrupted run
        worker = BatchWorker(store, workers=args.workers).start()
        print(f"Processing with {worker.workers} worker threads (Ctrl+C to stop, run again to resume)...")
        started = time.time()
        first_done = summary["finished"]
        try:
            while True:
                s = store.batch_summary(batch_id)
                elapsed = max(time.time() - started, 1e-6)
                rate = (s["finished"] - first_done) / elapsed
                eta = (s["total"] - s["finished"]) / rate if rate > 0 else 0
                print(f"\r  {s['finished']}/{s['total']} processed | {s['failed']} failed | "
                      f"{rate:.1f} files/s | about {eta / 60:.1f} min left   ", end="", flush=True)
                if s["complete"]:
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopped. Run the same command again to continue.")
        finally:
            worker.stop()
            get_parser().matcher.flush()
        print()

    s = store.batch_summary(batch_id)
    print(f"Batch #{batch_id}: {s['done']} parsed, {s['failed']} failed, {s['needs_review']} flagged for review, "
          f"{s['queued'] + s['processing']} still waiting.")
    if args.export:
        export(store, batch_id, Path(args.export))
    if hasattr(store, "wait_for_uploads"):  # MongoDB: the original files finish uploading first
        print("Storing the original files in MongoDB...")
        store.wait_for_uploads()


if __name__ == "__main__":
    main()
