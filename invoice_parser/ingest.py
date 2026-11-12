"""Load a PDF or image into pages of positioned words.

* PDF with a text layer  -> words read directly (exact, no OCR needed)
* PDF without text (scan) -> page rendered at RENDER_DPI, then OCR
* Image (jpg/png/...)     -> preprocessed (photo mode), then OCR
* Camera photo             -> page cut out, then the fast neural OCR (fast_ocr.py); Tesseract if
                              that is not installed
* Multi-page TIFF          -> every page OCR'd
"""
import io
import logging
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np

from . import config, fast_ocr
from .models import Page, Word
from .ocr import best_quarter_turn, ocr_image
from .preprocess import prepare_camera_photo, preprocess_scan

log = logging.getLogger(__name__)


class UnsupportedFileError(ValueError):
    pass


def load_pages(filename: str, data: bytes):
    ext = Path(filename).suffix.lower()
    if ext in config.PDF_EXTENSIONS:
        return _load_pdf(data)
    if ext in config.IMAGE_EXTENSIONS:
        return _load_images(data, ext, filename)
    raise UnsupportedFileError(f"Unsupported file type '{ext}'. Use PDF or an image.")


def overall_method(pages):
    methods = {p.method for p in pages}
    if not methods:
        return "none"
    return methods.pop() if len(methods) == 1 else "mixed"


def _load_pdf(data: bytes):
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        raw = page.get_text("words")
        w_pt, h_pt = page.rect.width, page.rect.height
        if len(raw) >= config.MIN_DIGITAL_WORDS:
            words = [Word(t, x0 / w_pt, y0 / h_pt, x1 / w_pt, y1 / h_pt, 1.0, i)
                     for x0, y0, x1, y1, t, *_ in raw if t.strip()]
            pages.append(Page(i, words, "digital", 1.0))
        else:
            pix = page.get_pixmap(dpi=config.RENDER_DPI, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            words, conf = ocr_image(preprocess_scan(bgr, photo=False), i)
            pages.append(Page(i, words, "ocr", conf))
    doc.close()
    return pages


def _load_images(data: bytes, ext: str, filename: str = ""):
    frames = _tiff_frames(data) if ext in (".tif", ".tiff") else []
    if not frames:
        arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise UnsupportedFileError("Could not read this image file.")
        frames = [arr]
    if (_is_camera_photo(data) or _from_app_camera(filename)) and fast_ocr.available():
        try:
            return [_read_camera_photo(arr, i) for i, arr in enumerate(frames)]
        except Exception:  # never lose a photo to the fast reader: Tesseract below reads it instead
            log.exception("Fast photo OCR failed for %s; using Tesseract", filename or "an image")
    # Camera photos are softer than scans and read best with larger text (24 px instead of 18 px).
    text_px = config.OCR_PHOTO_TEXT_PX if _is_camera_photo(data) else config.OCR_TEXT_PX
    pages = []
    for i, arr in enumerate(frames):
        img = preprocess_scan(arr, photo=True, text_px=text_px)
        words, conf = ocr_image(img, i, config.OCR_PHOTO_CONFIG)
        if conf < UPSIDE_DOWN_CHECK_BELOW and best_quarter_turn(img, (0, 2)) == 2:
            # poor reading: the page may be upside down (a quick sample check decides; upright pages
            # that are merely blurry are not read twice)
            flipped = np.ascontiguousarray(np.rot90(img, 2))
            words2, conf2 = ocr_image(flipped, i, config.OCR_PHOTO_CONFIG)
            if conf2 > conf:
                words, conf = words2, conf2
        pages.append(Page(i, words, "ocr", conf))
    return pages


# Mean OCR confidence under which a photo is checked for being upside down. Upright phone photos
# read at 0.6-0.85; an upside-down page reads well below 0.5.
UPSIDE_DOWN_CHECK_BELOW = 0.55


def _read_camera_photo(arr, index):
    """A phone photo through the fast reader (an upside-down page is turned inside read_photo)."""
    img, turned = prepare_camera_photo(arr)
    words, conf = fast_ocr.read_photo(img, index, check_upside_down=turned)
    return Page(index, words, "ocr", conf)


def _from_app_camera(filename: str) -> bool:
    """Pictures taken with the app's own camera page (IP Webcam / DroidCam / webcam): their frames
    carry no camera EXIF, so they are recognised by the names the app gives them."""
    return Path(filename).name.lower().startswith(("phone-", "webcam-"))


def _is_camera_photo(data: bytes) -> bool:
    """True for pictures taken by a phone / camera (EXIF camera make or model is present)."""
    from PIL import Image

    try:
        exif = Image.open(io.BytesIO(data)).getexif()
        return bool(exif.get(271) or exif.get(272))
    except Exception:
        return False


def _tiff_frames(data: bytes):
    """All pages of a (multi-page) TIFF as BGR arrays; [] if Pillow cannot read it."""
    from PIL import Image, ImageSequence

    try:
        with Image.open(io.BytesIO(data)) as im:
            return [cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
                    for frame in ImageSequence.Iterator(im)]
    except Exception:
        return []
