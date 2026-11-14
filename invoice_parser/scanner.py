"""Scan to folder: invoices from an office scanner (or any device or program that saves files) go into
the app by themselves.

Most office scanners and printers can save each scan as a PDF or JPG in a folder ("scan to folder" /
"scan to network folder"). The app checks that folder every few seconds, adds every new PDF or image to
the processing queue (the same file is never added twice) and moves it into the folder's 'imported'
subfolder, so the scanner folder only holds what is still waiting. Files that cannot be read stay
where they are. This works where the app runs on the office PC (a cloud server cannot see local folders);
other systems can always send files to POST /api/parse.
"""
import logging
import shutil
import threading
import time
from pathlib import Path

from . import config

log = logging.getLogger(__name__)

POLL_SECONDS = 5
SETTLE_SECONDS = 3  # a file still being written by the scanner is left for the next round
ALLOWED = config.PDF_EXTENSIONS | config.IMAGE_EXTENSIONS


def import_folder_once(folder, store, notify=None, settle=SETTLE_SECONDS):
    """Queue every finished PDF / image in folder and move it to folder/imported. Returns how many
    files were queued (files already in the database are moved too, but not counted)."""
    folder = Path(folder)
    if not folder.is_dir():
        return 0
    done = folder / "imported"
    added = 0
    now = time.time()
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in ALLOWED or now - path.stat().st_mtime < settle:
            continue
        try:
            data = path.read_bytes()
            _, is_new = store.add_file(path.name, data, queue=True)
            done.mkdir(exist_ok=True)
            target = done / path.name
            if target.exists():
                target = done / f"{path.stem}-{int(now)}{path.suffix}"
            shutil.move(str(path), str(target))
            added += is_new
        except OSError:
            log.exception("Could not import %s from the scanner folder", path)
    if added and notify:
        notify()
    return added


class FolderWatcher:
    """Checks the scanner folder chosen on the Scanner page (settings.json) in a background thread."""

    def __init__(self, store, notify, settings):
        self.store, self.notify, self.settings = store, notify, settings
        self.last_import = None  # (time, count) of the last round that queued files
        self._thread = None

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="scanner-folder", daemon=True)
            self._thread.start()
        return self

    def _run(self):
        while True:
            settings = self.settings()
            if settings.get("scan_folder_on") and settings.get("scan_folder"):
                try:
                    count = import_folder_once(settings["scan_folder"], self.store, self.notify)
                    if count:
                        self.last_import = (time.time(), count)
                except Exception:
                    log.exception("Scanner folder check failed")
            time.sleep(POLL_SECONDS)
