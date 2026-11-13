"""Load a PDF or image into pages of positioned words.

* PDF with a text layer  -> words read directly (exact, no OCR needed)
* PDF without text (scan) -> page rendered, then the fast neural OCR (fast_ocr.py)
* Image (jpg/png/...), camera photo, scan, multi-page TIFF -> page cut out, then the fast neural OCR,
                              which also reads forms filled in by hand
Tesseract reads a page only when the fast OCR is not installed (or switched off with FAST_PHOTO_OCR=0)
or fails on it. The fast OCR read the synthetic set (scans, scanned PDFs, photos) and the public
katanaml invoice images exactly as well as Tesseract (100% and 99.3% of fields) in a fraction of the
time, and Tesseract cannot read handwriting at all.
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
from .preprocess import CAMERA_WORK_SIDE, prepare_camera_photo, preprocess_scan

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
            if fast_ocr.available():
                try:  # 200 DPI is plenty: the fast reader works on a 2600 px copy of the page
                    pix = page.get_pixmap(dpi=FAST_RENDER_DPI, alpha=False)
                    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    pages.append(_read_camera_photo(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), i))
                    continue
                except Exception:
                    log.exception("Fast OCR failed for page %d of a scanned PDF; using Tesseract", i + 1)
            pix = page.get_pixmap(dpi=config.RENDER_DPI, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            words, conf = ocr_image(preprocess_scan(bgr, photo=False), i)
            pages.append(Page(i, words, "ocr", conf))
    doc.close()
    return pages


FAST_RENDER_DPI = 200


def _load_images(data: bytes, ext: str, filename: str = ""):
    fast = fast_ocr.available()  # photos, scans and screenshots alike (see the module docstring)
    frames = _tiff_frames(data) if ext in (".tif", ".tiff") else []
    reduced = False
    if not frames:
        flag = _photo_decode_flag(data) if fast else cv2.IMREAD_COLOR
        reduced = flag != cv2.IMREAD_COLOR
        arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flag)
        if arr is None:
            raise UnsupportedFileError("Could not read this image file.")
        frames = [arr]
    if fast:
        try:
            return [_read_camera_photo(arr, i) for i, arr in enumerate(frames)]
        except Exception:  # never lose a photo to the fast reader: Tesseract below reads it instead
            log.exception("Fast photo OCR failed for %s; using Tesseract", filename or "an image")
            if reduced:
                frames = [cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)]
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


def _photo_decode_flag(data: bytes):
    """How to decode a phone photo for the fast reader, which works on a copy of at most
    CAMERA_WORK_SIDE px. A JPEG at least twice (four times) that size is decoded at half (a quarter)
    size by the JPEG decoder itself: a 50 MP photo then needs 37 MB instead of 150 MB of memory
    (Render's free instance has 512 MB in all) and decodes several times faster."""
    from PIL import Image

    if not data.startswith(b"\xff\xd8"):  # not a JPEG
        return cv2.IMREAD_COLOR
    try:
        longest = max(Image.open(io.BytesIO(data)).size)  # reads the header only
    except Exception:
        return cv2.IMREAD_COLOR
    if longest >= 4 * CAMERA_WORK_SIDE:
        return cv2.IMREAD_REDUCED_COLOR_4
    if longest >= 2 * CAMERA_WORK_SIDE:
        return cv2.IMREAD_REDUCED_COLOR_2
    return cv2.IMREAD_COLOR


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
