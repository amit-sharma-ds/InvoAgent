"""Phone photos are read by the fast neural OCR (invoice_parser/fast_ocr.py); PDFs, scans and other
images keep Tesseract."""
import io

import numpy as np
import pytest
from PIL import Image
from test_pipeline import _tally_layout, _to_pdf

from invoice_parser import fast_ocr, ingest
from invoice_parser.extract import _looks_like_name
from invoice_parser.matcher import LabelMatcher
from invoice_parser.pipeline import InvoiceParser

needs_fast_ocr = pytest.mark.skipif(not fast_ocr.available(), reason="rapidocr / onnxruntime not installed")


@pytest.fixture(scope="module")
def parser():
    return InvoiceParser(matcher=LabelMatcher(use_embeddings=False))


def _camera_jpeg(turns=0):
    """The Tally invoice as a phone camera JPEG (camera make and model in EXIF), turned by `turns`
    quarter turns the way a phone stores a page photographed flat on a desk without a rotation tag."""
    import fitz

    page = fitz.open(stream=_to_pdf(_tally_layout), filetype="pdf")[0]
    pix = page.get_pixmap(dpi=200)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]
    exif = Image.Exif()
    exif[271], exif[272] = "vivo", "vivo Y100A"
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(np.rot90(img, turns))).save(buf, "JPEG", quality=90, exif=exif)
    return buf.getvalue()


@needs_fast_ocr
@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_phone_photo_is_read_by_the_fast_reader(parser, monkeypatch, turns):
    def no_tesseract(*args, **kwargs):
        raise AssertionError("a phone photo should not need Tesseract")

    monkeypatch.setattr(ingest, "ocr_image", no_tesseract)
    result = parser.parse_bytes("IMG_20261112_101010.jpg", _camera_jpeg(turns))
    f = result["fields"]
    assert f["invoice_number"]["value"] == "DPNP/2026-27/007"
    assert f["invoice_date"]["value"] == "2026-09-23"
    assert f["seller"]["value"]["name"] == "Deepnap Softech"
    assert f["seller"]["value"]["tax_id"] == "06BMXPS9096Q3ZB"
    assert f["buyer"]["value"]["name"] == "Jay Fe Cylinders Limited"
    assert f["buyer"]["value"]["tax_id"] == "06AABCL3501H1Z0"
    assert f["tax"]["value"] == pytest.approx(45000)
    assert f["total"]["value"] == pytest.approx(295000)


def test_images_without_camera_details_keep_tesseract(monkeypatch):
    def no_fast_reader(*args, **kwargs):
        raise AssertionError("scans and other images stay on Tesseract")

    monkeypatch.setattr(fast_ocr, "read_photo", no_fast_reader)
    monkeypatch.setattr(ingest, "ocr_image", lambda img, page, cfg=None: ([], 0.9))
    png = io.BytesIO()
    Image.new("RGB", (400, 300), "white").save(png, "PNG")
    pages = ingest.load_pages("scan.png", png.getvalue())
    assert len(pages) == 1 and pages[0].method == "ocr"


def test_a_failing_fast_reader_falls_back_to_tesseract(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("model file damaged")

    monkeypatch.setattr(fast_ocr, "available", lambda: True)
    monkeypatch.setattr(fast_ocr, "read_photo", broken)
    monkeypatch.setattr(ingest, "ocr_image", lambda img, page, cfg=None: ([], 0.9))
    pages = ingest.load_pages("phone-20261112-101010.jpg", _camera_jpeg())
    assert len(pages) == 1 and pages[0].ocr_conf == 0.9  # read by the (stubbed) Tesseract path


def test_pictures_from_the_apps_own_camera_page_count_as_photos():
    assert ingest._from_app_camera("phone-20261112-101010.jpg")
    assert ingest._from_app_camera("phone-camera-20261112-101010.jpg")
    assert ingest._from_app_camera("webcam-20261112-101010.png")
    assert not ingest._from_app_camera("scan.png")


def _boxes(text, x0=0.0, w=10.0):
    """(text, score, quad) for every character of text, side by side, 10 px wide each."""
    out = []
    for i, ch in enumerate(text):
        left = x0 + i * w
        out.append((ch, 0.95, [[left, 0], [left + w, 0], [left + w, 20], [left, 20]]))
    return out


def test_boxes_given_per_character_become_whole_words():
    """A full-width colon makes the recogniser box a GSTIN letter by letter; the words come from the
    line's own spaces, so it stays one word."""
    line = "GSTIN ：06AABCL3501H1Z0"
    parts = _boxes("GSTIN") + _boxes("：06AABCL3501H1Z0", x0=60)
    words = fast_ocr._line_words(line, 0.95, [[0, 0], [220, 0], [220, 20], [0, 20]], parts)
    assert [w for w, _, _ in words] == ["GSTIN", "：06AABCL3501H1Z0"]
    pts = words[1][2]
    assert pts[:, 0].min() == pytest.approx(60) and pts[:, 0].max() == pytest.approx(220)
    assert fast_ocr._clean(words[1][0]) == ":06AABCL3501H1Z0"


def test_a_date_with_the_space_before_the_year_missing_is_split():
    line = "Date: Jan 172012"
    parts = [("Date:", 0.99, [[0, 0], [50, 0], [50, 20], [0, 20]]),
             ("Jan", 0.99, [[60, 0], [90, 0], [90, 20], [60, 20]]),
             ("172012", 0.99, [[100, 0], [160, 0], [160, 20], [100, 20]])]
    words = fast_ocr._line_words(line, 0.99, [[0, 0], [160, 0], [160, 20], [0, 20]], parts)
    assert [w for w, _, _ in words] == ["Date:", "Jan", "17", "2012"]


def test_rows_are_levelled_with_the_slope_at_their_own_height():
    """A bent photo: text lines slope 4 degrees at the top of the page and 1 degree at the bottom.
    A word at the bottom right is levelled with the bottom slope, not the page's average."""
    def line(y, degrees):
        s = np.tan(np.radians(degrees))
        return [[100, y], [900, y + 800 * s], [900, y + 800 * s + 20], [100, y + 20]]

    boxes = [line(100 + 30 * i, 4.0) for i in range(8)] + [line(1500 + 30 * i, 1.0) for i in range(8)]
    field = fast_ocr._slope_field(boxes)
    word = np.array([[900, 1600], [960, 1600], [960, 1620], [900, 1620]], dtype=np.float32)
    levelled = fast_ocr._level(word, field, 1000)
    expected = 1600 - (word[:, 0] - 500) * np.tan(np.radians(1.0))
    assert levelled[:, 1][:2] == pytest.approx(expected[:2], abs=0.5)


def test_a_label_and_a_gstin_is_not_a_name():
    assert not _looks_like_name("STIN:09DFIPA7756B1ZH")
    assert not _looks_like_name("O6AABCL3501H1Z0")
    assert _looks_like_name("Yash Traders GSTIN 09DFIPA7756B1ZH")
    assert _looks_like_name("Deepnap Softech")


def test_a_phone_number_beside_a_label_is_not_an_amount(parser):
    def layout(c):
        c.setFont("Helvetica", 10)
        c.drawString(40, 800, "GSTIN: 09DFIPA7756B1ZH")
        c.drawString(250, 800, "Tax Invoice")
        c.drawString(420, 800, "Mob.: 8535000042")
        c.drawString(40, 760, "M/S YASH TRADERS")

    f = parser.parse_bytes("handwritten.pdf", _to_pdf(layout))["fields"]
    assert not f.get("tax") or f["tax"]["value"] != pytest.approx(8535000042)
