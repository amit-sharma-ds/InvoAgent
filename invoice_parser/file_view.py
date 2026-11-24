"""The original file of an invoice shown in the app, page by page as JPEG pictures.

Every browser shows a picture, while a phone's browser has no PDF viewer and downloads a PDF instead
(and no browser shows a TIFF), so the View page shows pictures and downloading stays a separate button.
"""
import io
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageOps

from . import config

PAGE_ZOOM = 2.0      # PDF pages at 144 dpi: an A4 page is 1190 px wide, sharp on a phone and on a laptop
PAGE_MAX_SIDE = 2400  # pictures larger than this are made smaller for the screen
JPEG_QUALITY = 85


def _is_pdf(file_name):
    return Path(file_name or "").suffix.lower() in config.PDF_EXTENSIONS


def page_count(path, file_name):
    """Pages of the file (frames of a TIFF); 0 when it cannot be shown."""
    try:
        if _is_pdf(file_name):
            with fitz.open(path) as doc:
                return doc.page_count
        with Image.open(path) as im:
            return getattr(im, "n_frames", 1)
    except Exception:
        return 0


def page_jpeg(path, file_name, page):
    """Page `page` (from 1) of the file as JPEG bytes; None when there is no such page."""
    if page < 1:
        return None
    if _is_pdf(file_name):
        with fitz.open(path) as doc:
            if page > doc.page_count:
                return None
            pix = doc[page - 1].get_pixmap(matrix=fitz.Matrix(PAGE_ZOOM, PAGE_ZOOM), alpha=False)
            im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    else:
        with Image.open(path) as src:
            try:
                src.seek(page - 1)
            except EOFError:
                return None
            im = ImageOps.exif_transpose(src)  # a phone photo the right way up
            if im.mode in ("RGBA", "LA", "P"):
                rgba = im.convert("RGBA")
                im = Image.new("RGB", rgba.size, "white")
                im.paste(rgba, mask=rgba.split()[-1])
            else:
                im = im.convert("RGB")
    if max(im.size) > PAGE_MAX_SIDE:
        im.thumbnail((PAGE_MAX_SIDE, PAGE_MAX_SIDE), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return out.getvalue()
