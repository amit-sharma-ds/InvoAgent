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
    assert f["invoice_number"]["value"] == "KVIN/2026-27/007"
    assert f["invoice_date"]["value"] == "2026-09-23"
    assert f["seller"]["value"]["name"] == "Kavya Infotech"
    assert f["seller"]["value"]["tax_id"] == "06BKXPT4822Q3ZY"
    assert f["buyer"]["value"]["name"] == "Shakti Steel Tubes Limited"
    assert f["buyer"]["value"]["tax_id"] == "06AABCS7788K1ZP"
    assert f["tax"]["value"] == pytest.approx(45000)
    assert f["total"]["value"] == pytest.approx(295000)


def test_scans_are_read_by_the_fast_reader_too(monkeypatch):
    def no_tesseract(*args, **kwargs):
        raise AssertionError("a scan should not need Tesseract")

    monkeypatch.setattr(fast_ocr, "available", lambda: True)
    monkeypatch.setattr(fast_ocr, "read_photo", lambda img, page, check_upside_down=False, keep_lines=False:
                        ([], 0.8, None) if keep_lines else ([], 0.8))
    monkeypatch.setattr(ingest, "ocr_image", no_tesseract)
    png = io.BytesIO()
    Image.new("RGB", (400, 300), "white").save(png, "PNG")
    pages = ingest.load_pages("scan.png", png.getvalue())
    assert len(pages) == 1 and pages[0].method == "ocr" and pages[0].ocr_conf == 0.8


def test_tesseract_reads_images_when_the_fast_reader_is_off(monkeypatch):
    monkeypatch.setattr(fast_ocr, "available", lambda: False)
    monkeypatch.setattr(ingest, "ocr_image", lambda img, page, cfg=None: ([], 0.9))
    png = io.BytesIO()
    Image.new("RGB", (400, 300), "white").save(png, "PNG")
    pages = ingest.load_pages("scan.png", png.getvalue())
    assert len(pages) == 1 and pages[0].ocr_conf == 0.9


def test_a_failing_fast_reader_falls_back_to_tesseract(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("model file damaged")

    monkeypatch.setattr(fast_ocr, "available", lambda: True)
    monkeypatch.setattr(fast_ocr, "read_photo", broken)
    monkeypatch.setattr(ingest, "ocr_image", lambda img, page, cfg=None: ([], 0.9))
    pages = ingest.load_pages("phone-20261112-101010.jpg", _camera_jpeg())
    assert len(pages) == 1 and pages[0].ocr_conf == 0.9  # read by the (stubbed) Tesseract path


def _jpeg(width, height):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, "JPEG")
    return buf.getvalue()


def test_big_phone_photos_are_decoded_at_a_smaller_size():
    import cv2

    assert ingest._photo_decode_flag(_jpeg(4608, 2592)) == cv2.IMREAD_COLOR       # 12 MP: as it is
    assert ingest._photo_decode_flag(_jpeg(8160, 6144)) == cv2.IMREAD_REDUCED_COLOR_2  # 50 MP: half
    assert ingest._photo_decode_flag(_jpeg(10400, 7000)) == cv2.IMREAD_REDUCED_COLOR_4
    png = io.BytesIO()
    Image.new("RGB", (8160, 6144), "white").save(png, "PNG")
    assert ingest._photo_decode_flag(png.getvalue()) == cv2.IMREAD_COLOR  # only JPEGs are reduced


def test_a_big_photo_still_falls_back_to_a_full_size_tesseract_page(monkeypatch):
    seen = []
    monkeypatch.setattr(fast_ocr, "available", lambda: True)
    monkeypatch.setattr(fast_ocr, "read_photo", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("broken")))
    monkeypatch.setattr(ingest, "preprocess_scan", lambda img, **k: seen.append(img.shape) or img[:8, :8])
    monkeypatch.setattr(ingest, "ocr_image", lambda img, page, cfg=None: ([], 0.9))
    ingest.load_pages("phone-big.jpg", _jpeg(5400, 3000))
    assert seen == [(3000, 5400, 3)]


@needs_fast_ocr
def test_the_models_give_their_working_memory_back_after_each_run():
    session = fast_ocr._get_engine().text_rec.session.session
    assert isinstance(session, fast_ocr._Session)
    assert session._run_options.get_run_config_entry("memory.enable_memory_arena_shrinkage") == "cpu:0"


@needs_fast_ocr
def test_batches_read_at_the_same_time_give_the_same_text(monkeypatch):
    import cv2

    lines = []
    for k, text in enumerate(["Invoice No. 7", "Kavya Infotech", "GSTIN 06BKXPT4822Q3ZY", "Total 2,95,000.00",
                              "23/09/2026", "CGST @ 9%", "Plot 12, Sector 5, Gurugram", "Qty 1"] * 4):
        img = np.full((48, 40 + 22 * len(text) + 7 * (k % 5), 3), 255, np.uint8)
        cv2.putText(img, text, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        lines.append(img)
    rec = fast_ocr._get_engine().text_rec
    monkeypatch.setattr(fast_ocr, "PARALLEL", 1)
    one_by_one = fast_ocr._recognize(rec, lines, batch_size=3)
    monkeypatch.setattr(fast_ocr, "PARALLEL", 4)
    monkeypatch.setattr(fast_ocr, "_pool", None)
    together = fast_ocr._recognize(rec, lines, batch_size=3)
    assert [(t, round(s, 6)) for t, s, _ in together] == [(t, round(s, 6)) for t, s, _ in one_by_one]
    assert together[0][0].replace(" ", "") == "InvoiceNo.7"


def _line_image(text, width=None):
    import cv2

    img = np.full((48, width or 30 + 20 * len(text), 3), 255, np.uint8)
    cv2.putText(img, text, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return img


def _page_lines(readings):
    """fast_ocr.PageLines of a page whose lines (rendered text) the small model read as `readings`."""
    crops = [_line_image(truth) for truth, _, _ in readings]
    boxes = np.array([[[10, 60 * k], [10 + c.shape[1], 60 * k], [10 + c.shape[1], 60 * k + 48], [10, 60 * k + 48]]
                      for k, c in enumerate(crops)], dtype=np.float32)
    return fast_ocr.PageLines(0, (60 * len(crops) + 40, 1000), boxes, crops,
                              [(read, score, None) for _, read, score in readings], set())


@needs_fast_ocr
def test_a_second_look_keeps_the_surer_reading_of_a_line():
    lines = _page_lines([("Skirting boards", "Skirtng boards", 0.80),   # misread, not sure
                         ("Fit 3 kitchen cupboards", "Fit 3 kitchen cupboards", 0.99)])  # sure: left alone
    words = fast_ocr.reread_lines(lines, [0, 1])
    assert lines.read[0][0] == "Skirting boards" and lines.read[0][1] > 0.8
    assert lines.read[1] == ("Fit 3 kitchen cupboards", 0.99, None) and 1 not in lines.hand_read
    assert "Skirting" in [w.text for w in words]
    assert fast_ocr.reread_lines(lines, [0]) is None  # a line is read again only once


def test_the_second_look_reads_the_item_and_party_lines(monkeypatch):
    from invoice_parser import pipeline
    from invoice_parser.models import Page

    lines = fast_ocr.PageLines(0, (800, 600), None, [None] * 5, [
        ("OAKWOOD JOINERY LTD", 1.0, None), ("To: Mr K. Verma", 0.9, None), ("Skirting boards", 0.9, None),
        ("95.00", 0.9, None), ("Thank you", 0.9, None)], set())
    asked = []
    monkeypatch.setattr(fast_ocr, "reread_lines", lambda page_lines, indexes: asked.append(indexes))
    result = {"line_items": [{"description": "Skirting boards"}],
              "fields": {"buyer": {"value": {"name": "Mr K. Verma"}}, "seller": None}}
    assert pipeline._second_look([Page(0, [], "ocr", 0.9, lines)], result) is False
    assert asked == [[1, 2]]


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
    line = "GSTIN ：06AABCS7788K1ZP"
    parts = _boxes("GSTIN") + _boxes("：06AABCS7788K1ZP", x0=60)
    words = fast_ocr._line_words(line, 0.95, [[0, 0], [220, 0], [220, 20], [0, 20]], parts)
    assert [w for w, _, _ in words] == ["GSTIN", "：06AABCS7788K1ZP"]
    pts = words[1][2]
    assert pts[:, 0].min() == pytest.approx(60) and pts[:, 0].max() == pytest.approx(220)
    assert fast_ocr._clean(words[1][0]) == ":06AABCS7788K1ZP"


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
    assert not _looks_like_name("STIN:09AGHPR3344C1ZG")
    assert not _looks_like_name("O6AABCS7788K1ZP")
    assert _looks_like_name("Om Traders GSTIN 09AGHPR3344C1ZG")
    assert _looks_like_name("Kavya Infotech")


def test_a_phone_number_beside_a_label_is_not_an_amount(parser):
    def layout(c):
        c.setFont("Helvetica", 10)
        c.drawString(40, 800, "GSTIN: 09AGHPR3344C1ZG")
        c.drawString(250, 800, "Tax Invoice")
        c.drawString(420, 800, "Mob.: 9000012345")
        c.drawString(40, 760, "M/S OM TRADERS")

    f = parser.parse_bytes("handwritten.pdf", _to_pdf(layout))["fields"]
    assert not f.get("tax") or f["tax"]["value"] != pytest.approx(9000012345)
