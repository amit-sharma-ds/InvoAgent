"""Background processing of queued invoices.

N worker threads take queued invoices from the database and parse them. Threads are enough
for real parallelism here: Tesseract runs as a separate program and PyMuPDF / OpenCV release
Python's GIL, so several files are OCR'd at the same time, while the label model is loaded once.
A file that fails is marked 'failed' with its error; the rest of the batch continues.
"""
import logging
import os
import threading
import time

from . import config
from .pipeline import get_parser
from .storage import known_party_names

log = logging.getLogger(__name__)


class BatchWorker:
    def __init__(self, store, workers=None, parser=None):
        self.store = store
        self.workers = workers or config.WORKERS
        self._parser = parser
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._threads = []
        if self.workers > 1:
            os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # one core per Tesseract run
            os.environ.setdefault("KMP_WARNINGS", "0")      # torch's OpenMP would warn about that limit

    @property
    def parser(self):
        return self._parser or get_parser()

    def start(self):
        if self._threads:
            return self
        self.store.reset_stuck()
        for i in range(self.workers):
            t = threading.Thread(target=self._run, name=f"invoice-worker-{i + 1}", daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def notify(self):
        """Wake the workers right away (new files were queued)."""
        self._wake.set()

    def stop(self, wait=True):
        self._stop.set()
        self._wake.set()
        if wait:
            for t in self._threads:
                t.join(timeout=60)
        self._threads = []

    def _run(self):
        while not self._stop.is_set():
            job = self.store.claim_next()
            if job is None:
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue
            self.process(job)

    def process(self, job):
        """Parse one stored file and save the result (or the error)."""
        started = time.perf_counter()
        try:
            data = self.store.file_path(job).read_bytes()
            result = self.parser.parse_bytes(job["file_name"], data)
            result["invoice_id"] = job["id"]
            if hasattr(self.store, "approved_party_names"):
                try:  # a misread party name -> the name approved before for its GSTIN
                    known_party_names(result, self.store)
                except Exception:
                    log.exception("Could not look up known party names")
            self.store.save_result(job["id"], result, int((time.perf_counter() - started) * 1000))
            return result
        except Exception as exc:
            log.exception("Failed to process invoice %s (%s)", job["id"], job["file_name"])
            self.store.save_error(job["id"], f"{type(exc).__name__}: {exc}")
            return None

    def process_now(self, invoice_id):
        """Process one invoice in the calling thread (used for single uploads and the API)."""
        row = self.store.get(invoice_id)
        if row is None:
            return None
        return self.process({"id": row["id"], "file_name": row["file_name"], "stored_path": row["stored_path"]})
