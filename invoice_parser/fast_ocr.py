"""Fast OCR for phone photos: PaddleOCR's PP-OCRv6 tiny text detector and recogniser, run on the
CPU with ONNX Runtime (through RapidOCR). Both models ship with the app (invoice_parser/models,
6 MB, Apache-2.0, SHA-256 below), so nothing is downloaded at run time.

Why: Tesseract needed 25-35 s for a phone photo on the 2-core dev PC (minutes on a small server),
mostly for reading a 2000x3000 page at large letter size, deskewing it and re-reading doubtful
words. These neural models find the text lines on a small copy of the page and read each line
from a 2000 px copy: about 4 s for a full Tally invoice on the same busy PC (whole parse), with
cleaner text (GSTINs and invoice numbers came out exactly, tilted photos need no deskew pass).

Only camera photos use it (ingest.py decides); PDFs, scans and other images keep Tesseract.
"""
import os
import re
import threading
import unicodedata
from pathlib import Path

import cv2
import numpy as np

from .models import Word

MODEL_DIR = Path(__file__).with_name("models")
DET_MODEL = MODEL_DIR / "PP-OCRv6_det_tiny.onnx"  # sha256 f42c0fbd294d95eac1a550e131b277dac97462c8025fa4b6c3cec1b7894bd3d5
REC_MODEL = MODEL_DIR / "PP-OCRv6_rec_tiny.onnx"  # sha256 e16e242de5937ad92609223f19bc2aff3727ee40b095f996907c24749bad251b

READ_SIDE = 2000  # longest side of the page the text lines are cut from (recognition quality)
FIND_SIDE = 1280  # longest side of the copy the text lines are found on (detection time grows with pixels)
# Text lines are read in batches of lines of about the same width, because every line of a batch is
# padded to the widest one. The library pads every line to at least 320 px (6.7 line heights) and
# batches lines in order of width, 16 at a time: most invoice lines are 2-5 heights wide, so about
# half the reading work went into padding.
BATCH = 24             # most text lines read per model call
LINE_MIN_WIDTH = 2.0   # shorter lines are padded to this width (in line heights)
GROUP_SPREAD = 1.3     # a batch takes lines up to this many times as wide as its narrowest
# Share of lines read with 0.9+ confidence under which the page is tried upside down. Upright
# photos of the test sets: 0.9-1.0; the same photos upside down: 0.1-0.2.
UPSIDE_DOWN_BELOW = 0.5
THREADS = int(os.environ.get("PHOTO_OCR_THREADS", "0"))  # 0 = ONNX Runtime's default (all cores)

_engine = None
_engine_lock = threading.Lock()
_run_lock = threading.Lock()  # one photo at a time: a single read already uses every core


def available() -> bool:
    """True when the models and the runtime are installed (otherwise photos use Tesseract)."""
    if os.environ.get("FAST_PHOTO_OCR", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    if not (DET_MODEL.is_file() and REC_MODEL.is_file()):
        return False
    try:
        import onnxruntime  # noqa: F401
        import rapidocr  # noqa: F401
    except ImportError:
        return False
    return True


def _get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            from rapidocr import RapidOCR

            engine = RapidOCR(params={
                "Global.log_level": "error",
                "Global.use_cls": False,  # page orientation is handled for the whole page (below, preprocess)
                "Global.max_side_len": 4000,
                "Det.model_path": str(DET_MODEL),
                "Det.limit_type": "max",
                "Det.limit_side_len": FIND_SIDE,
                "Rec.model_path": str(REC_MODEL),
                "EngineConfig.onnxruntime.intra_op_num_threads": THREADS or -1,
            })
            # The library's model wrappers are used (image preparation, decoding, word boxes), with
            # ONNX Runtime sessions of our own, whose idle threads sleep instead of spinning: on the
            # dev PC, sharing the CPU with another busy program, that took 20-25% less CPU per photo.
            engine._load_det_model().session.session = _session(DET_MODEL)
            engine._load_rec_model().session.session = _session(REC_MODEL)
            _engine = engine
        return _engine


def _session(model_path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_cpu_mem_arena = True  # 5-10% faster on the dev PC
    if THREADS:
        opts.intra_op_num_threads = THREADS
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])


def warm_up():
    """Load the models (about a second) so the first photo does not wait for it."""
    if available():
        img = np.full((64, 320, 3), 255, np.uint8)
        cv2.putText(img, "Invoice 123", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        read_photo(img, 0)


def read_photo(img, page_index: int, check_upside_down: bool = False):
    """OCR a phone photo of a page (BGR or gray, already cut out of the photo; text running left to
    right, possibly upside down). Returns (words, mean_confidence) like ocr.ocr_image.

    check_upside_down: the page was given a quarter turn (photographed sideways), so it is upside
    down half the time; its widest lines are read both ways round to decide before the full reading.
    Otherwise the page is read as it is, and read again turned round only when it reads poorly.
    The text lines are found once: a turned page reuses them, turned with it."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    longest = max(img.shape[:2])
    if longest > READ_SIDE:
        f = READ_SIDE / longest
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    with _run_lock:
        engine = _get_engine()
        boxes = _find_lines(engine, img)
        if boxes is None:
            return [], 0.0
        if check_upside_down and _reads_better_turned(engine, img, boxes):
            img, boxes = _turn_round(engine, img, boxes)
        words, conf, sure = _read(engine, img, boxes, page_index)
        if sure < UPSIDE_DOWN_BELOW:  # poor reading: the page may be upside down
            words2, conf2, sure2 = _read(engine, *_turn_round(engine, img, boxes), page_index)
            if sure2 > sure:
                return words2, conf2
    return words, conf


def _find_lines(engine, img):
    """Corner points (n x 4 x 2: top-left, top-right, bottom-right, bottom-left) of the page's text
    lines, top to bottom; None when there is no text."""
    found = engine.text_det(img)
    if found.boxes is None or not len(found.boxes):
        return None
    return np.asarray(found.boxes, dtype=np.float32)


def _turn_round(engine, img, boxes):
    """The page and its text lines turned 180 degrees."""
    h, w = img.shape[:2]
    turned = boxes.copy()
    turned[..., 0] = w - turned[..., 0]
    turned[..., 1] = h - turned[..., 1]
    turned = np.roll(turned, 2, axis=1)  # the old bottom-right corner is the new top-left one
    return np.ascontiguousarray(img[::-1, ::-1]), engine.text_det.sorted_boxes(turned)


PROBE_LINES = 8  # widest text lines the upside-down check reads both ways round


def _reads_better_turned(engine, img, boxes):
    """True when the page's widest text lines read with more confidence turned 180 degrees."""
    width = np.linalg.norm(boxes[:, 1] - boxes[:, 0], axis=1)
    height = np.linalg.norm(boxes[:, 3] - boxes[:, 0], axis=1)
    lines = [k for k in np.argsort(-width) if width[k] >= 3 * height[k] and height[k] >= 8][:PROBE_LINES]
    if not lines:
        return False
    crops = engine.crop_text_regions(img, boxes[lines])
    read = _recognize(engine.text_rec, crops + [np.ascontiguousarray(c[::-1, ::-1]) for c in crops])
    as_is = sum(score for _, score, _ in read[:len(crops)])
    turned = sum(score for _, score, _ in read[len(crops):])
    return turned > as_is


def _recognize(rec, crops, word_boxes=False):
    """(text, confidence, character positions or None) for each cropped text line. Lines are sorted
    by width and read in batches of about equally wide lines, so little of the work is padding."""
    ratios = [c.shape[1] / c.shape[0] for c in crops]
    order = np.argsort(ratios, kind="stable")
    out = [None] * len(crops)
    start = 0
    while start < len(order):
        end = start + 1
        limit = GROUP_SPREAD * max(ratios[order[start]], LINE_MIN_WIDTH)
        while end < len(order) and end - start < BATCH and ratios[order[end]] <= limit:
            end += 1
        batch = order[start:end]
        widest = max(LINE_MIN_WIDTH, ratios[order[end - 1]])
        pixels = np.stack([rec.resize_norm_img(crops[k], widest) for k in batch]).astype(np.float32)
        lines, positions = rec.postprocess_op(rec.session(pixels), word_boxes,
                                              wh_ratio_list=[ratios[k] for k in batch], max_wh_ratio=widest)
        for m, k in enumerate(batch):
            out[k] = (lines[m][0], float(lines[m][1]), positions[m] if word_boxes else None)
        start = end
    return out


def _read(engine, img, boxes, page_index):
    """(words, mean confidence, share of lines read with 0.9+ confidence)."""
    from rapidocr.ch_ppocr_rec import TextRecOutput

    crops = engine.crop_text_regions(img, boxes)
    read = _recognize(engine.text_rec, crops, word_boxes=True)
    h, w = img.shape[:2]
    boxes = boxes.copy()
    boxes[..., 0] = np.clip(boxes[..., 0], 0, w)
    boxes[..., 1] = np.clip(boxes[..., 1], 0, h)
    # word / character boxes from the character positions, in page coordinates
    located = engine.cal_rec_boxes(crops, boxes, TextRecOutput(txts=tuple(t for t, _, _ in read),
                                                               word_results=tuple(p for _, _, p in read)))
    parts_of = located.word_results if len(located.word_results) == len(read) else [()] * len(read)
    keep = [k for k, (text, score, _) in enumerate(read) if text.strip() and score >= engine.text_score]
    if not keep:
        return [], 0.0, 0.0
    scores = np.array([read[k][1] for k in keep])
    slopes = _slope_field(boxes[keep])
    words = []
    for k in keep:
        line, line_score = read[k][0], read[k][1]
        parts = [(t, s, b) for t, s, b in parts_of[k] if b is not None]
        for text, score, pts in _line_words(line, line_score, boxes[k], parts):
            text = _clean(text)
            if not text:
                continue
            pts = _level(pts, slopes, w)
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            words.append(Word(text, _unit(x0 / w), _unit(y0 / h), _unit(x1 / w), _unit(y1 / h),
                              float(score), page_index))
    return words, float(scores.mean()), float((scores >= 0.9).mean())


MAX_SLOPE = np.tan(np.radians(10))  # tilts beyond this are not text lines (or not straightened)
SLOPE_NEIGHBOURS = 7  # long text lines whose median slope gives the slope at a height of the page


def _slope_field(boxes):
    """(height, slope) of every long text line, sorted by height; None with too few lines. Each
    detected line box follows its text, so these slopes describe how the photo tilts, also when it
    tilts differently at the top and the bottom (a bent page, or a photo taken at an angle): 4
    degrees at the top and 1 degree at the totals on a real photo, where one rotation for the whole
    page put each tax amount on the row above its label."""
    field = []
    for quad in boxes:
        (x0, y0), (x1, y1), _, (x3, y3) = np.asarray(quad, dtype=np.float64)[:4]
        width, height = np.hypot(x1 - x0, y1 - y0), np.hypot(x3 - x0, y3 - y0)
        if width >= 4 * height and x1 > x0:
            slope = (y1 - y0) / (x1 - x0)
            if abs(slope) <= MAX_SLOPE:
                field.append(((y0 + y1 + y3) / 3, slope))
    if len(field) < 3:
        return None
    return np.array(sorted(field))


def _level(pts, slopes, w):
    """Move a word's corner points so that rows are level: each point drops or rises by the local
    slope times its distance from the middle of the page. The image itself is never rotated."""
    if slopes is None:
        return pts
    yc = float(pts[:, 1].mean())
    near = np.argsort(np.abs(slopes[:, 0] - yc))[:SLOPE_NEIGHBOURS]
    slope = float(np.median(slopes[near, 1]))
    if abs(slope) < np.tan(np.radians(0.2)):
        return pts
    out = pts.copy()
    out[:, 1] -= (pts[:, 0] - w / 2) * slope
    return out


# The recogniser sometimes drops the narrow space between a day and the year ('Jan 172012' for
# 'Jan 17 2012' on two of twelve SuperStore photos); the year is put back on its own.
_GLUED_DATE = re.compile(r"(?i)\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?)\s*(\d{1,2})((?:19|20)\d\d)\b")


def _line_words(line, line_score, line_quad, parts):
    """The words of one text line, split at its spaces, each as (text, score, corner points).

    The recogniser's own word boxes (parts) are used for the positions, character by character: it
    sometimes returns one box per character instead of per word (a line with a full-width colon,
    ': 06AABCL3501H1Z0', is treated as Chinese), which spaced a GSTIN out letter by letter."""
    line = _GLUED_DATE.sub(r"\1 \2 \3", line)
    chars = []  # (character, score, its share of its box) for every non-space character
    for text, score, quad in parts or ():
        t = [c for c in text if not c.isspace()]
        tl, tr, br, bl = np.asarray(quad, dtype=np.float32)[:4]
        for i in range(len(t)):
            a, b = i / len(t), (i + 1) / len(t)
            chars.append((t[i], score, np.array([tl + (tr - tl) * a, tl + (tr - tl) * b,
                                                 bl + (br - bl) * b, bl + (br - bl) * a])))
    if [c for c, _, _ in chars] != [c for c in line if not c.isspace()]:
        return _split_line(line, line_score, line_quad)  # boxes do not match the text: spread the words
    out, pos = [], 0
    for token in line.split():
        piece = chars[pos: pos + len(token)]
        pos += len(token)
        out.append((token, min(s for _, s, _ in piece), np.vstack([p for _, _, p in piece])))
    return out


def _split_line(text, score, quad):
    """Word boxes for a line without usable ones: each space-separated word gets a share of the
    line's width in proportion to its length."""
    tokens = text.split()
    if not tokens:
        return []
    tl, tr, br, bl = np.asarray(quad, dtype=np.float32)[:4]
    total = sum(len(t) for t in tokens) + len(tokens) - 1
    out, pos = [], 0
    for t in tokens:
        a, b = pos / total, (pos + len(t)) / total
        out.append((t, score, np.array([tl + (tr - tl) * a, tl + (tr - tl) * b,
                                        bl + (br - bl) * b, bl + (br - bl) * a])))
        pos += len(t) + 1
    return out


def _clean(text):
    """Plain text: full-width punctuation ('：' from the multilingual model) becomes ASCII, and letters
    of other scripts (a stray Chinese character read from a speck) are dropped."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith(("Lo", "Lm", "C")))
    return text.strip()


def _unit(v):
    return float(min(1.0, max(0.0, v)))
