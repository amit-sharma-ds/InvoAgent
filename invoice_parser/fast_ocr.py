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
# A larger recogniser of the same family (21 MB) that reads handwriting much better: used only on pages
# that read poorly (handwritten forms), and only for their doubtful lines.
HAND_MODEL = MODEL_DIR / "PP-OCRv6_rec_small.onnx"  # sha256 6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884

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
# Handwriting: a page with less than this share of lines read with 0.9+ confidence (printed photos:
# 0.9-1.0; a printed form filled in by hand: about 0.6) has its lines below REREAD_BELOW read again
# by the larger model, which keeps the more confident reading of each line.
HANDWRITING_BELOW = 0.85
REREAD_BELOW = 0.95
REREAD_LONG_BELOW = 0.85  # lines longer than REREAD_MAX_RATIO heights are read again only below this
REREAD_MAX_RATIO = 12
HAND_TRUST = 0.6  # the larger model's reading wins from this confidence on
HAND_BATCH = 8  # the larger model's working memory grows with the batch (512 MB servers)
THREADS = int(os.environ.get("PHOTO_OCR_THREADS", "0"))  # 0 = ONNX Runtime's default (all cores)

_engine = None
_hand = None
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
            # ONNX Runtime sessions of our own (_Session).
            engine._load_det_model().session.session = _Session(DET_MODEL)
            engine._load_rec_model().session.session = _Session(REC_MODEL)
            _engine = engine
        return _engine


class _Session:
    """An ONNX Runtime session tuned for a small shared server:

    * its idle threads sleep instead of spinning: on the dev PC, sharing the CPU with another busy
      program, that took 20-25% less CPU per photo;
    * it gives its unused working memory back after every run. ONNX Runtime keeps the memory a
      model needed in a pool for the next run, and with text lines of every size the pool only
      grew: 140 MB after loading, 500 MB after five photos (peak 570 MB), more than the 512 MB of
      Render's free instance, which restarted the app. Emptied after each run, the process stays
      at about 190 MB between photos (peak about 375 MB) at the same speed; switching the pool off
      instead cost 25-30% more CPU."""

    def __init__(self, model_path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_cpu_mem_arena = True  # 5-10% faster on the dev PC
        opts.enable_mem_pattern = False   # input sizes change with every photo
        if THREADS:
            opts.intra_op_num_threads = THREADS
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self._session = ort.InferenceSession(str(model_path), sess_options=opts,
                                             providers=["CPUExecutionProvider"])
        self._run_options = ort.RunOptions()
        self._run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0")

    def run(self, output_names, input_feed, run_options=None):
        return self._session.run(output_names, input_feed, self._run_options)

    def __getattr__(self, name):  # inputs, outputs and metadata of the model
        return getattr(self._session, name)


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
        crops, read = _read(engine, img, boxes)
        conf, sure = _confidence(engine, read)
        if sure < UPSIDE_DOWN_BELOW:  # poor reading: the page may be upside down
            t_img, t_boxes = _turn_round(engine, img, boxes)
            t_crops, t_read = _read(engine, t_img, t_boxes)
            t_conf, t_sure = _confidence(engine, t_read)
            if t_sure > sure:
                img, boxes, crops, read, conf, sure = t_img, t_boxes, t_crops, t_read, t_conf, t_sure
        if sure < HANDWRITING_BELOW:  # still poor: details written by hand
            read = [_with_gaps(text, score, info) for text, score, info in _reread(crops, read)]
            boxes, crops, read = _fill_gaps(engine, img, boxes, crops, read)
            conf, _ = _confidence(engine, read)
        return _words(engine, img, boxes, crops, read, page_index), conf


def _hand_reader():
    """The larger recogniser for handwriting, loaded on first use (None when it is not installed or
    switched off with HANDWRITING_OCR=0)."""
    global _hand
    if _hand is None and HAND_MODEL.is_file() and \
            os.environ.get("HANDWRITING_OCR", "1").strip().lower() not in ("0", "false", "no", "off"):
        import types

        from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode

        session = _Session(HAND_MODEL)
        name = session.get_inputs()[0].name
        keys = session.get_modelmeta().custom_metadata_map["character"].splitlines()
        _hand = types.SimpleNamespace(session=lambda pixels: session.run(None, {name: pixels})[0],
                                      postprocess_op=CTCLabelDecode(character=keys),
                                      resize_norm_img=_get_engine().text_rec.resize_norm_img)  # same 48 px lines
    return _hand


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


def _recognize(rec, crops, word_boxes=False, batch_size=BATCH):
    """(text, confidence, character positions or None) for each cropped text line. Lines are sorted
    by width and read in batches of about equally wide lines, so little of the work is padding."""
    ratios = [c.shape[1] / c.shape[0] for c in crops]
    order = np.argsort(ratios, kind="stable")
    out = [None] * len(crops)
    start = 0
    while start < len(order):
        end = start + 1
        limit = GROUP_SPREAD * max(ratios[order[start]], LINE_MIN_WIDTH)
        while end < len(order) and end - start < batch_size and ratios[order[end]] <= limit:
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


def _read(engine, img, boxes):
    """The page's text lines: (crops, [(text, confidence, character positions)])."""
    crops = engine.crop_text_regions(img, boxes)
    return crops, _recognize(engine.text_rec, crops, word_boxes=True)


def _confidence(engine, read):
    """(mean confidence, share of lines read with 0.9+ confidence) of the lines that are kept."""
    scores = np.array([s for t, s, _ in read if t.strip() and s >= engine.text_score])
    if not len(scores):
        return 0.0, 0.0
    return float(scores.mean()), float((scores >= 0.9).mean())


def _reread(crops, read):
    """Doubtful lines read again by the larger (handwriting) model. Its reading is kept when it is
    fairly sure (HAND_TRUST) or surer than the small model: on a real handwritten form it read
    '3805' for '380S', '42288' for '4R88' and the circled quantity '①' for '0', often with a
    slightly lower confidence than the small model's wrong reading."""
    hand = _hand_reader()
    # fairly sure long lines are printed sentences (address, terms): reading them again costs the most
    # (the larger model's time grows with a line's width) and changes nothing that matters
    doubtful = [k for k, (_, score, _) in enumerate(read)
                if score < REREAD_LONG_BELOW or (score < REREAD_BELOW
                                                 and crops[k].shape[1] <= REREAD_MAX_RATIO * crops[k].shape[0])]
    if hand is None or not doubtful:
        return read
    again = _recognize(hand, [crops[k] for k in doubtful], word_boxes=True, batch_size=HAND_BATCH)
    read = list(read)
    for k, new in zip(doubtful, again):
        if _clean(new[0]) and (new[1] >= HAND_TRUST or new[1] > read[k][1]):
            read[k] = new
    return read


GAP_MIN = 0.5     # a gap between two words of a row, in line heights, that may hide a missed character ...
GAP_MAX = 3.0     # ... up to this wide
GAP_INK = 0.25    # ink across at least this share of a line height (a ruled line is narrower)
GAP_TRUST = 0.9   # the larger model's confidence needed to keep what it read in a gap
GAPS_PER_PAGE = 40
_RULE_LIKE = set("1lIi|!/\\[](){}.,:;-_'\"`~")  # what a ruled line or a dot leader reads as


def _fill_gaps(engine, img, boxes, crops, read):
    """Handwriting the line finder missed between two words of a row is read on its own: a date
    written in the boxes of a form, '12 | 9 | 26', came back as '12' and '26' without the '9'.
    A gap is read only when it holds ink wider than a ruled line, and the larger model's reading is
    kept only when it is sure of a letter or digit that a ruled line does not look like ('1', 'l')."""
    hand = _hand_reader()
    if hand is None or len(boxes) < 2:
        return boxes, crops, read
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x0, y0 = boxes[..., 0].min(axis=1), boxes[..., 1].min(axis=1)
    x1, y1 = boxes[..., 0].max(axis=1), boxes[..., 1].max(axis=1)
    tall = y1 - y0
    kept = [k for k, (t, s, _) in enumerate(read) if t.strip() and s >= engine.text_score]
    spots = []
    for i in kept:
        right = [j for j in kept if x0[j] > x1[i]
                 and min(y1[i], y1[j]) - max(y0[i], y0[j]) > 0.5 * min(tall[i], tall[j])]
        if not right:
            continue
        j = min(right, key=lambda k: x0[k])
        h = float(min(tall[i], tall[j]))
        if not GAP_MIN * h <= x0[j] - x1[i] <= GAP_MAX * h or h < 8:
            continue
        top, bottom = int(min(y0[i], y0[j])), int(np.ceil(max(y1[i], y1[j])))
        left, end = int(np.ceil(x1[i])) + 2, int(x0[j]) - 2
        patch = gray[top:bottom, left:end]
        if patch.size == 0 or int(patch.max()) - int(patch.min()) < 60:
            continue  # blank paper
        cut, _ = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        core = patch[int(0.15 * patch.shape[0]): int(np.ceil(0.85 * patch.shape[0]))] < cut
        cols = np.flatnonzero(core.sum(axis=0) >= 2)  # the band's middle: not the row's ruled lines
        if len(cols) < GAP_INK * h:
            continue
        a, b = left + max(0, cols[0] - 3), left + min(patch.shape[1], cols[-1] + 4)
        crop = img[top:bottom, a:b].copy()
        # the form's box lines in the gap ('12 | 9 | 26') are whited out: they read as '1'
        rule = (gray[top:bottom, a:b] < cut).mean(axis=0) >= 0.8
        crop[:, rule] = 255
        spots.append((np.array([[a, top], [b, top], [b, bottom], [a, bottom]], dtype=np.float32), crop))
    if not spots:
        return boxes, crops, read
    spots = spots[:GAPS_PER_PAGE]
    gap_crops = [c for _, c in spots]
    found = _recognize(hand, gap_crops, word_boxes=True, batch_size=HAND_BATCH)
    small = _recognize(engine.text_rec, gap_crops)  # both models must read the same
    new = [k for k, (text, score, _) in enumerate(found)
           if score >= GAP_TRUST and re.search(r"[A-Za-z0-9]", text) and set(text.replace(" ", "")) - _RULE_LIKE
           and text.replace(" ", "") == small[k][0].replace(" ", "")]
    if not new:
        return boxes, crops, read
    return (np.concatenate([boxes, np.stack([spots[k][0] for k in new])]), list(crops) + [gap_crops[k] for k in new],
            list(read) + [found[k] for k in new])


SPLIT_GAP_MIN = 5    # model columns between two letters (the library itself splits words above 5)
SPLIT_GAP_TIMES = 1.6  # ... and this many times the line's usual gap between letters


def _with_gaps(text, score, info):
    """Handwriting often comes back without the space between words ('SURESHMITTAL', 'SAIENTERPRISES')
    or with rupees and paise run together across the column line ('4228814'). A gap between two
    letters much wider than the line's usual one starts a new word: a space is put in the text and
    the character positions are grouped again, so each word gets its own box."""
    from rapidocr.ch_ppocr_rec.typings import WordType

    # printed lines read with confidence keep their words: a wide printed letter ('m' in a bold
    # 'SunElectricals') leaves a gap as wide as a handwritten word space
    if score >= REREAD_BELOW and not any(len(t) >= 8 and re.fullmatch(r"[A-Z0-9]+", t) for t in text.split()):
        return text, score, info
    cols = [c for wc in (getattr(info, "word_cols", None) or []) for c in wc]
    chars = [ch for ch in text if not ch.isspace()]
    if len(cols) < 4 or len(cols) != len(chars):
        return text, score, info
    usual = float(np.median(np.diff(cols)))
    cuts = {j for j in range(1, len(cols))
            if cols[j] - cols[j - 1] >= max(SPLIT_GAP_MIN, SPLIT_GAP_TIMES * usual)}
    if not cuts:
        return text, score, info
    out, j = [], 0  # the text with a space before every cut that has none yet
    for ch in text:
        if ch.isspace():
            out.append(ch)
            continue
        if j in cuts and out and not out[-1].isspace():
            out.append(" ")
        out.append(ch)
        j += 1
    new_text = "".join(out)
    words, word_cols, k = [], [], 0
    for token in new_text.split():
        words.append(list(token))
        word_cols.append(cols[k: k + len(token)])
        k += len(token)
    info.words, info.word_cols = words, word_cols
    info.word_types = [WordType.EN_NUM] * len(words)
    return new_text, score, info


def _words(engine, img, boxes, crops, read, page_index):
    """Positioned words of the page from its read lines (rows levelled, see _level)."""
    from rapidocr.ch_ppocr_rec import TextRecOutput

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
        return []
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
            # below 1.0: elsewhere a confidence of 1.0 marks the exact text of a PDF text layer
            words.append(Word(text, _unit(x0 / w), _unit(y0 / h), _unit(x1 / w), _unit(y1 / h),
                              min(float(score), 0.999), page_index))
    return words


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
    ': 06AABCS7788K1ZP', is treated as Chinese), which spaced a GSTIN out letter by letter."""
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
    text = unicodedata.normalize("NFKC", text).replace("×", "x")  # a handwritten x reads as a times sign
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith(("Lo", "Lm", "C")))
    return text.strip()


def _unit(v):
    return float(min(1.0, max(0.0, v)))
