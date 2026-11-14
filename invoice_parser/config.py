"""Settings. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "1" if default else "0")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- Storage -----------------------------------------------------------------
DATA_DIR = Path(os.environ.get("INVOICE_DATA_DIR", str(PROJECT_ROOT / "data")))
# One workspace, "live": the company's invoices. (An earlier "demo" workspace with sample invoices
# was removed from the app; its folder data/demo is left untouched on disk.)
WORKSPACES = ("live",)


def workspace_dir(name: str) -> Path:
    return DATA_DIR / name


DB_PATH = Path(os.environ.get("INVOICE_DB_PATH", str(workspace_dir("live") / "invoices.db")))
FILES_DIR = workspace_dir("live") / "files"       # original invoices, stored by SHA-256
LABEL_CACHE_PATH = DATA_DIR / "label_cache.json"  # remembered label -> field decisions


def _mongodb_uri() -> str:
    uri = os.environ.get("MONGODB_URI", "").strip()
    if uri:
        return uri
    try:  # this PC: the connection string in a file that is never committed (data/*.txt)
        return (DATA_DIR / "mongodb_uri.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


# --- MongoDB (shared database) -----------------------------------------------
# With a MongoDB connection string, invoices, approvals, corrections and the original files are kept
# in MongoDB: every computer and the server show the same data and a server restart loses nothing.
# Set MONGODB_URI (Render: Environment) or put the string in data/mongodb_uri.txt on a PC.
# Without it the app keeps SQLite in data/live.
MONGODB_URI = _mongodb_uri()
MONGODB_DB = os.environ.get("MONGODB_DB", "invoagent")
MONGODB_ERROR = ""  # set when MONGODB_URI is given but MongoDB could not be reached at start-up
# Original files are not stored in MongoDB once the whole cluster would pass this size (MB), so the
# other databases on a free 512 MB Atlas cluster never run out of space. Invoice data is still saved.
MONGODB_FILES_LIMIT_MB = int(os.environ.get("MONGODB_FILES_LIMIT_MB", "450"))

# --- Processing --------------------------------------------------------------
# Parallel worker threads for bulk jobs. Default: CPU cores - 1 (max 8).
WORKERS = int(os.environ.get("INVOICE_WORKERS", "0")) or max(1, min(8, (os.cpu_count() or 2) - 1))
# Invoices below this overall confidence (or with failed checks / missing fields) are flagged for review.
REVIEW_CONFIDENCE = float(os.environ.get("REVIEW_CONFIDENCE", "0.8"))

# --- OCR (Tesseract program) -------------------------------------------------
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
OCR_LANG = os.environ.get("OCR_LANG", "eng")
OCR_CONFIG = os.environ.get("OCR_CONFIG", "--oem 3 --psm 6")
# Images and photos (box-style forms, ruled lines removed) read best as columns of text (psm 4).
OCR_PHOTO_CONFIG = os.environ.get("OCR_PHOTO_CONFIG", "--oem 3 --psm 4")
RENDER_DPI = int(os.environ.get("RENDER_DPI", "300"))
# A PDF page with fewer embedded words than this is treated as a scan and OCR'd.
MIN_DIGITAL_WORDS = 8
# Pages are scaled so the median letter is about this many pixels tall before OCR.
# 18 px reads as well as larger sizes on the test sets and halves OCR time for 300-DPI scans.
OCR_TEXT_PX = int(os.environ.get("OCR_TEXT_PX", "18"))
# Phone photos are softer than scans; reading them at 24 px recovered values that 18 px missed
# (tested on real phone photos of a Tally GST invoice).
OCR_PHOTO_TEXT_PX = int(os.environ.get("OCR_PHOTO_TEXT_PX", "24"))
# Never OCR an image with a longest side above this (very large photos).
MAX_OCR_SIDE = int(os.environ.get("MAX_OCR_SIDE", "4000"))

# --- Semantic label matching (seller ~ vendor ~ supplier) --------------------
USE_EMBEDDINGS = _env_bool("USE_EMBEDDINGS", True)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Ambiguous dates such as 05/03/2024: True = day first (India/EU), False = month first (US)
DAY_FIRST = _env_bool("DAY_FIRST", True)

# --- Uploads -----------------------------------------------------------------
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "300"))  # whole request (many files or a ZIP)
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
