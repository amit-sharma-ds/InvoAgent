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
import types
import unicodedata
from pathlib import Path

import cv2
import numpy as np

from .models import Word

MODEL_DIR = Path(__file__).with_name("models")
DET_MODEL = MODEL_DIR / "PP-OCRv6_det_tiny.onnx"  # sha256 f42c0fbd294d95eac1a550e131b277dac97462c8025fa4b6c3cec1b7894bd3d5
REC_MODEL = MODEL_DIR / "PP-OCRv6_rec_tiny.onnx"  # sha256 e16e242de5937ad92609223f19bc2aff3727ee40b095f996907c24749bad251b
# A larger recogniser of the same family that reads handwriting much better: used only on pages that
# read poorly (handwritten forms), and only for their doubtful lines. Its matrix multiplications are
# stored as 8-bit integers (ONNX Runtime dynamic quantization of PP-OCRv6_rec_small, 21 -> 14 MB): 27%
# faster on the dev PC and as accurate (hand-written digits 64/69 against 63/69; the same 86/96 fields on
# the real handwritten bills). Quantizing its convolutions too made it 7x slower and wrong half the time.
HAND_MODEL = MODEL_DIR / "PP-OCRv6_rec_small_int8.onnx"  # sha256 107c36589b690810f5be9703526ce9f419e61e1be9e96e16b33c5f2669bc6aa4

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
HAND_BATCH = 4  # the larger model's working memory grows with the batch (512 MB servers: 8 used 80 MB more)
THREADS = int(os.environ.get("PHOTO_OCR_THREADS", "0"))  # 0 = ONNX Runtime's default (all cores)
# Batches of text lines read at the same time (1 = one after another). A model call splits every layer
# over the cores and waits for the slowest part, so the cores often wait for each other, most of all on
# a busy CPU. Handwritten bills on the dev PC (2 cores / 4 threads, another program using about 1.6):
# 19.9 s one batch at a time, 17.6 s two, 14.3 s four (with this program given priority: 6.3, 5.9 and
# 5.4 s), the same text and fields, the same peak memory (about 340 MB).
PARALLEL = max(1, int(os.environ.get("PHOTO_OCR_PARALLEL", "0")) or min(4, os.cpu_count() or 1))

_engine = None
_hand = None
_pool = None
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
    """Load the models (a second or two) so the first photo does not wait for it: the handwriting
    model too, so the first handwritten bill is not a second slower than the next ones."""
    if available():
        img = np.full((64, 320, 3), 255, np.uint8)
        cv2.putText(img, "Invoice 123", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        read_photo(img, 0)
        with _run_lock:
            hand = _hand_reader()
            if hand is not None:
                _recognize(hand, [img], batch_size=HAND_BATCH)


def read_photo(img, page_index: int, check_upside_down: bool = False, keep_lines: bool = False):
    """OCR a phone photo of a page (BGR or gray, already cut out of the photo; text running left to
    right, possibly upside down). Returns (words, mean_confidence) like ocr.ocr_image, and with
    keep_lines the page's text lines as a third value (PageLines), for a second look at the lines the
    extraction used (reread_lines).

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
        boxes, text_map = _find_lines(engine, img)
        if boxes is None:
            return ([], 0.0, None) if keep_lines else ([], 0.0)
        if check_upside_down and _reads_better_turned(engine, img, boxes):
            img, boxes = _turn_round(engine, img, boxes)
            text_map = text_map[..., ::-1, ::-1]
        crops, read = _read(engine, img, boxes)
        conf, sure = _confidence(engine, read)
        if sure < UPSIDE_DOWN_BELOW:  # poor reading: the page may be upside down
            t_img, t_boxes = _turn_round(engine, img, boxes)
            t_crops, t_read = _read(engine, t_img, t_boxes)
            t_conf, t_sure = _confidence(engine, t_read)
            if t_sure > sure:
                img, boxes, crops, read, conf, sure = t_img, t_boxes, t_crops, t_read, t_conf, t_sure
                text_map = text_map[..., ::-1, ::-1]
        hand_read, own = set(), len(read)  # lines the handwriting model has read already; the page's own lines
        if sure < HANDWRITING_BELOW:  # still poor: details written by hand
            hand_read = set(_doubtful(crops, read))
            read = [_with_gaps(text, score, info) for text, score, info in _reread(crops, read)]
            boxes, crops, read = _fill_gaps(engine, img, boxes, crops, read)
            boxes, crops, read = _more_date_boxes(engine, img, boxes, crops, read, text_map)
        boxes, crops, read, dated = _reread_dates(engine, img, boxes, crops, read)
        if sure < HANDWRITING_BELOW or dated:
            conf, _ = _confidence(engine, read)
        words = _words(engine, img, boxes, crops, read, page_index)
    if not keep_lines:
        return words, conf
    # gap and date readings (added after the page's own lines, or without character positions) are
    # the larger model's already
    hand_read |= {k for k, (_, _, info) in enumerate(read) if info is None or k >= own}
    return words, conf, PageLines(page_index, img.shape[:2], boxes, list(crops), list(read), hand_read, sure)


class PageLines:
    """A read page's text lines, kept for a second look (reread_lines) until its fields are found."""

    def __init__(self, page_index, shape, boxes, crops, read, hand_read, sure=0.0):
        self.page_index, self.shape, self.boxes = page_index, shape, boxes
        self.crops, self.read, self.hand_read = crops, read, hand_read
        self.sure = sure  # share of its lines the small model read with 0.9+ confidence


SECOND_LOOK_BELOW = 0.98  # lines the extraction used are read again by the larger model below this ...
SECOND_LOOK_LINES = 16    # ... at most this many per page ...
# ... on pages with some doubtful lines. Printed photos read 99-100% of their lines at 0.9+ (no second
# look, no extra time); a printed form filled in by hand read 83-97% (photo variants of one such bill).
SECOND_LOOK_PAGE_BELOW = 0.98
# On a page that reads well as a whole, a line is still read again when one of its letters was read
# below this: a WhatsApp copy of a printed form with two handwritten words read 100% of its lines at 0.9+,
# while 'internal' came back as 'intermal' (its 'n' at 0.55) and 'fitting' as 'fiing' (an 'i' at 0.47).
WEAK_LETTER = 0.6


def _weakest_letter(text, info):
    """The lowest confidence of a letter of a read line (1.0 when the letters' confidences are not known)."""
    sure = getattr(info, "confs", None) or ()
    if len(sure) != len(text):
        return 1.0
    return min((p for ch, p in zip(text, sure) if ch.isalpha()), default=1.0)


def reread_lines(lines, indexes):
    """A second look at chosen lines of a read page (the item and party lines the extraction used) with
    the handwriting model. A page that reads well as a whole (a printed form with a few words written
    in) is not read again line by line, and the small model can misread those words: a handwritten
    'fitting' came back as 'fiing', 'internal' as 'intermal' (the larger model read both right). On such
    a page only the lines with a doubtful letter (WEAK_LETTER) are read again. The larger model's reading
    is kept only when it is surer and changes no number. Returns the page's words when a line changed,
    otherwise None."""
    def doubtful(k):
        text, score, info = lines.read[k]
        return score < SECOND_LOOK_BELOW and (lines.sure < SECOND_LOOK_PAGE_BELOW
                                              or _weakest_letter(text, info) < WEAK_LETTER)

    hand = _hand_reader()
    todo = [k for k in dict.fromkeys(indexes)
            if 0 <= k < len(lines.read) and k not in lines.hand_read and doubtful(k)][:SECOND_LOOK_LINES]
    if hand is None or not todo:
        return None
    with _run_lock:
        again = _recognize(hand, [lines.crops[k] for k in todo], word_boxes=True, batch_size=HAND_BATCH)
        changed = False
        for k, new in zip(todo, again):
            lines.hand_read.add(k)
            old, score, _ = lines.read[k]
            if (_clean(new[0]) and new[1] > score and not _spoils_number(old, new[0])
                    and not _adds_digits(old, new[0])):
                lines.read[k] = _with_gaps(*new)
                changed = True
        if not changed:
            return None
        shape_only = np.broadcast_to(np.zeros((), np.uint8), tuple(lines.shape) + (3,))  # _words needs the size
        return _words(_get_engine(), shape_only, lines.boxes, lines.crops, lines.read, lines.page_index)


def _hand_reader():
    """The larger recogniser for handwriting, loaded on first use (None when it is not installed or
    switched off with HANDWRITING_OCR=0)."""
    global _hand
    if _hand is None and HAND_MODEL.is_file() and \
            os.environ.get("HANDWRITING_OCR", "1").strip().lower() not in ("0", "false", "no", "off"):
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
    lines, top to bottom (None when there is no text), and the line finder's text map, kept for a
    second, looser look on a handwritten page (_more_date_boxes)."""
    det = engine.text_det
    pixels = det.get_preprocess()(img)
    if pixels is None:
        return None, None
    text_map = det.session(pixels)
    boxes, _ = det.postprocess_op(text_map, img.shape[:2])
    if not len(boxes):
        return None, text_map
    return np.asarray(det.sorted_boxes(np.asarray(boxes)), dtype=np.float32), text_map


def _turn_round(engine, img, boxes):
    """The page and its text lines turned 180 degrees."""
    h, w = img.shape[:2]
    turned = boxes.copy()
    turned[..., 0] = w - turned[..., 0]
    turned[..., 1] = h - turned[..., 1]
    turned = np.roll(turned, 2, axis=1)  # the old bottom-right corner is the new top-left one
    return np.ascontiguousarray(img[::-1, ::-1]), engine.text_det.sorted_boxes(turned)


PROBE_LINES = 8  # widest text lines the upside-down check reads both ways round
PROBE_CLEAR = 0.3  # ... in two rounds of four: a mean difference this large after the first round decides


def _reads_better_turned(engine, img, boxes):
    """True when the page's widest text lines read with more confidence turned 180 degrees. Usually
    the first four lines settle it (one way round they read at 0.9, the other at 0.3); the other four
    are read only when it is close, which saves about 0.4 s on a photo taken sideways."""
    width = np.linalg.norm(boxes[:, 1] - boxes[:, 0], axis=1)
    height = np.linalg.norm(boxes[:, 3] - boxes[:, 0], axis=1)
    lines = [k for k in np.argsort(-width) if width[k] >= 3 * height[k] and height[k] >= 8][:PROBE_LINES]
    as_is = turned = 0.0
    for start in range(0, len(lines), PROBE_LINES // 2):
        part = lines[start: start + PROBE_LINES // 2]
        crops = engine.crop_text_regions(img, boxes[part])
        read = _recognize(engine.text_rec, crops + [np.ascontiguousarray(c[::-1, ::-1]) for c in crops])
        as_is += sum(score for _, score, _ in read[:len(crops)])
        turned += sum(score for _, score, _ in read[len(crops):])
        if abs(turned - as_is) >= PROBE_CLEAR * (start + len(part)):
            break
    return turned > as_is


def _batch_pool():
    """The threads that read batches of text lines at the same time (PARALLEL)."""
    global _pool
    with _engine_lock:
        if _pool is None:
            from concurrent.futures import ThreadPoolExecutor

            _pool = ThreadPoolExecutor(max_workers=PARALLEL, thread_name_prefix="photo-ocr")
        return _pool


def _recognize(rec, crops, word_boxes=False, batch_size=BATCH):
    """(text, confidence, character positions or None) for each cropped text line. Lines are sorted
    by width and read in batches of about equally wide lines, so little of the work is padding.
    PARALLEL batches are read at the same time; each batch's reading is the same either way."""
    ratios = [c.shape[1] / c.shape[0] for c in crops]
    order = np.argsort(ratios, kind="stable")
    batches, start = [], 0
    while start < len(order):
        end = start + 1
        limit = GROUP_SPREAD * max(ratios[order[start]], LINE_MIN_WIDTH)
        while end < len(order) and end - start < batch_size and ratios[order[end]] <= limit:
            end += 1
        batches.append(order[start:end])
        start = end

    def read(batch):
        widest = max(LINE_MIN_WIDTH, ratios[batch[-1]])
        pixels = np.stack([rec.resize_norm_img(crops[k], widest) for k in batch]).astype(np.float32)
        return rec.postprocess_op(rec.session(pixels), word_boxes,
                                  wh_ratio_list=[ratios[k] for k in batch], max_wh_ratio=widest)

    done = _batch_pool().map(read, batches) if PARALLEL > 1 and len(batches) > 1 else map(read, batches)
    out = [None] * len(crops)
    for batch, (lines, positions) in zip(batches, done):
        for m, k in enumerate(batch):
            out[k] = (lines[m][0], float(lines[m][1]), positions[m] if word_boxes else None)
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
    doubtful = _doubtful(crops, read)
    if hand is None or not doubtful:
        return read
    again = _recognize(hand, [crops[k] for k in doubtful], word_boxes=True, batch_size=HAND_BATCH)
    read = list(read)
    for k, new in zip(doubtful, again):
        old = read[k][0]
        if (_clean(new[0]) and not _spoils_number(old, new[0])
                and (new[1] > read[k][1] or (new[1] >= HAND_TRUST and not _adds_digits(old, new[0])))):
            read[k] = new
    return read


def _doubtful(crops, read):
    """The lines _reread reads again. Fairly sure long lines are printed sentences (address, terms):
    reading them again costs the most (the larger model's time grows with a line's width) and changes
    nothing that matters."""
    return [k for k, (_, score, _) in enumerate(read)
            if score < REREAD_LONG_BELOW or (score < REREAD_BELOW
                                             and crops[k].shape[1] <= REREAD_MAX_RATIO * crops[k].shape[0])]


def _spoils_number(old, new):
    """True when `new` turns a digit of a number in `old` (a run of digits only, such as '12' or the '288'
    of '42,288.14') into a letter or drops it: the quantized larger model once read the date 'Date:12' as
    'Date:1R'. Letters among digits (a GSTIN '068KXPT…' -> '06BKXPT…') and in words may still change."""
    from difflib import SequenceMatcher

    for tag, i1, i2, j1, j2 in SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag not in ("replace", "delete"):
            continue
        lost = sum(ch.isdigit() for ch in old[i1:i2]) - sum(ch.isdigit() for ch in new[j1:j2])
        if lost <= 0:
            continue
        for i in range(i1, i2):
            if not old[i].isdigit():
                continue
            a = b = i
            while a > 0 and old[a - 1].isalnum():
                a -= 1
            while b + 1 < len(old) and old[b + 1].isalnum():
                b += 1
            if b > a and old[a: b + 1].isdigit():
                return True
    return False


def _adds_digits(old, new):
    """True when `new` has digits put in between the characters of `old` rather than in place of some of
    them. A less sure re-reading may turn letters into the digits they hide ('380S' -> '3805', '4R88' ->
    '42288'), but not add digits to a number read surely: the larger model, quantized, once read a total
    of '49900' as '499900'."""
    from difflib import SequenceMatcher

    return any(tag == "insert" and any(ch.isdigit() for ch in new[j1:j2])
               for tag, _, _, j1, j2 in SequenceMatcher(None, old, new, autojunk=False).get_opcodes())


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


_DATE_WORD = re.compile(r"(?i)\bdat(?:e|ed)\b\s*[:.;]?")
_WRITTEN_DATE = re.compile(r"\b(\d{1,2})\s*[|/.\- ]\s*(\d{1,2})\s*[|/.\- ]\s*((?:19|20)?\d{2})\b")
_MONTH_NAME = re.compile(r"(?i)\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b")
_WORD = re.compile(r"[A-Za-z]{3,}")  # a word, not a misread figure ('1R', 'O9')
_YEAR_FIRST = re.compile(r"\b(?:19|20)\d\d\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{1,2}\b")  # 2026-09-12
DATE_SURE = 0.5         # mean confidence of a date's digits needed to keep a date read figure by figure
DATE_BOX_THRESH = 0.4   # a second, looser look for date boxes on a handwritten page (the finder's own: 0.5)


def _more_date_boxes(engine, img, boxes, crops, read, text_map):
    """A date label the line finder was not quite sure of is looked for again: on a slightly turned
    photo of a bill book it dropped the whole 'Date: 12|8|25' box (score just under 0.5) and only '25'
    was read. Boxes found at a looser threshold that the first look did not have are read, and those
    holding a date label are added."""
    if text_map is None or len(boxes) == 0:
        return boxes, crops, read
    import copy

    looser = copy.copy(engine.text_det.postprocess_op)
    looser.box_thresh = DATE_BOX_THRESH
    extra, _ = looser(text_map, img.shape[:2])
    if not len(extra):
        return boxes, crops, read
    extra = np.asarray(extra, dtype=np.float32)
    ex0, ey0, ex1, ey1 = extra[..., 0].min(1), extra[..., 1].min(1), extra[..., 0].max(1), extra[..., 1].max(1)
    bx0, by0, bx1, by1 = boxes[..., 0].min(1), boxes[..., 1].min(1), boxes[..., 0].max(1), boxes[..., 1].max(1)
    inter = (np.clip(np.minimum(ex1[:, None], bx1) - np.maximum(ex0[:, None], bx0), 0, None)
             * np.clip(np.minimum(ey1[:, None], by1) - np.maximum(ey0[:, None], by0), 0, None)).sum(1)
    new = extra[inter < 0.3 * np.maximum((ex1 - ex0) * (ey1 - ey0), 1)]
    if not len(new):
        return boxes, crops, read
    new_crops = engine.crop_text_regions(img, new)
    found = _recognize(engine.text_rec, new_crops, word_boxes=True)
    keep = [k for k, (t, s, _) in enumerate(found) if s >= engine.text_score and _DATE_WORD.search(t)]
    if not keep:
        return boxes, crops, read
    return (np.concatenate([boxes, new[keep]]), list(crops) + [new_crops[k] for k in keep],
            list(read) + [found[k] for k in keep])


def _plain_date(text):
    """True when the text holds one written date and no other figures ('12 8 25', '12/08/2025'); a
    date with a month name ('12 Sep 2026') or the year first ('2026-09-12') counts as plain too: it
    is left as read."""
    if _MONTH_NAME.search(text) or _YEAR_FIRST.search(text):
        return True
    m = _WRITTEN_DATE.search(text)
    return bool(m) and not re.search(r"\d", text[:m.start()] + text[m.end():])


def _reread_dates(engine, img, boxes, crops, read):
    """A handwritten date is read again figure by figure. On bill books it came back as 'Date:1R 8 25.',
    as '10 1 7 1 25' (the form's box lines read as '1') or in pieces, and was then not found or found
    wrong. The label and what is written right of it (also a little lower: handwriting sits under
    the printed label) are cut out as one strip and read by both models, and the strip's best reading
    as a date (day / month / year: see _read_date) replaces the pieces when its figures are read with
    some confidence (DATE_SURE). A date that already reads plainly is left as it is, so a printed page
    costs nothing here. Returns (boxes, crops, read, whether a date was read again)."""
    hand = _hand_reader()
    dated = False
    x0, y0 = boxes[..., 0].min(axis=1), boxes[..., 1].min(axis=1)
    x1, y1 = boxes[..., 0].max(axis=1), boxes[..., 1].max(axis=1)
    read = list(read)
    for i, (text, score, _) in enumerate(read):
        label = _DATE_WORD.search(text)
        if not label or score < engine.text_score or _WORD.search(text[label.end():]):
            continue  # not a date label, or other words after it in the same box ('Date: ... Due Date')
        h = y1[i] - y0[i]
        row, end = [i], x1[i]
        for j in sorted((j for j in range(len(read)) if j != i and x0[j] >= x1[i] - 2 and read[j][0].strip()
                         and read[j][1] >= engine.text_score and y0[i] - 0.3 * h <= (y0[j] + y1[j]) / 2 <= y1[i] + 0.6 * h),
                        key=lambda j: x0[j]):
            if x0[j] - end > 3 * h or _WORD.search(read[j][0]):
                break  # the next label or words: not part of the date
            row.append(j)
            end = max(end, x1[j])
        after = " ".join([text[label.end():]] + [read[j][0] for j in row[1:]])
        if not re.search(r"\d", after) or _plain_date(after):
            continue
        in_line = bool(text[label.end():].strip())  # the date was read in the label's own box
        top = max(0, int(min(y0[j] for j in row)) - 2)
        bottom = min(img.shape[0], int(np.ceil(max(y1[j] for j in row))) + 2)
        left = int(x0[i]) if in_line else int(np.ceil(x1[i]))
        right = min(img.shape[1], int(np.ceil(end)) + 2)
        if right - left < 8 or bottom - top < 8:
            continue
        strip = img[top:bottom, left:right].copy()
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        cut, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        rule = (gray < cut).mean(axis=0) >= 0.8  # the form's box lines across the strip: they read as '1'
        if in_line:
            rule[: int(x1[i] - left) // 2] = False  # not in the printed label
        strip[:, rule] = 255
        best = None
        for rec in (hand, engine.text_rec):
            if rec is None:
                continue
            got = _read_date(rec, strip, after_label=in_line)
            if got and (best is None or got[1] > best[1]):
                best = got
        if best is None or best[1] < DATE_SURE:
            continue
        boxes = boxes.copy()
        boxes[i] = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)
        crops = list(crops)
        crops[i] = strip
        read[i] = (text[:label.end()].rstrip() + " " + best[0], round(best[1], 3), None)
        for j in row[1:]:
            read[j] = ("", 0.0, None)  # read as part of the date
        dated = True
    return boxes, crops, read, dated


_DATE_LEAD = set(":;.,-|/ ")
_DATE_SEPS = set("/-.|\\, ")
_DATE_TAIL = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ")
_machines = {}


def _date_machine(keys):
    """The written-date grammar for a recogniser's character list, as a table of moves: what may come
    before the date (':', spaces), day (1-31), a separator, month (1-12), a separator, year (two
    figures, or 20xx), and anything but a figure after it. Day first, as on Indian bills: with the
    month free to be 1-31 a box line next to a '9' made the month '19'. States: 0 before; 1-5 day
    ('0', '1'/'2', '3', '4'-'9', two figures); 6 separator; 7-11 month ('0', '1', -, '2'-'9', two
    figures); 12 separator; 13 '2', 14 '0'/'1'/'3', 15 '20', 16 two-figure year, 17 '20x', 18
    four-figure year; 19 after the date. Returns (model columns, moves, part of each state (0 before,
    1 date, 2 after), accepting states)."""
    if id(keys) in _machines:
        return _machines[id(keys)]
    allowed = [k for k, ch in enumerate(keys) if k and len(ch) == 1 and (ch in "0123456789" or ch in _DATE_TAIL)]

    def move(s, ch):
        digit = ch in "0123456789"
        if s == 0:
            if digit:
                return 1 if ch == "0" else 2 if ch in "12" else 3 if ch == "3" else 4
            return 0 if ch in _DATE_LEAD else -1
        if 1 <= s <= 5:  # day
            k = s - 1
            if digit:
                return 5 if (k == 0 and ch != "0") or k == 1 or (k == 2 and ch in "01") else -1
            return 6 if ch in _DATE_SEPS and k != 0 else -1
        if 7 <= s <= 11:  # month
            k = s - 7
            if digit:
                return 11 if (k == 0 and ch != "0") or (k == 1 and ch in "012") else -1
            return 12 if ch in _DATE_SEPS and k != 0 else -1
        if s == 6:
            if digit:
                return 7 if ch == "0" else 8 if ch == "1" else 10
            return 6 if ch in _DATE_SEPS else -1
        if s == 12:
            if digit:
                return 13 if ch == "2" else 14 if ch in "013" else -1
            return 12 if ch in _DATE_SEPS else -1
        if s == 13:
            return (15 if ch == "0" else 16) if digit else -1
        if s == 14:
            return 16 if digit else -1
        if s == 15:
            return (17 if ch in "0123" else -1) if digit else (19 if ch in _DATE_TAIL else -1)
        if s == 17:
            return 18 if digit else -1
        return 19 if not digit and ch in _DATE_TAIL else -1  # 16, 18, 19

    moves = np.array([[move(s, keys[k]) for k in allowed] for s in range(20)], dtype=np.int32)
    part = np.array([0] + [1] * 18 + [2])
    machine = (np.array([0] + allowed), moves, part, np.array([15, 16, 18, 19]))
    _machines[id(keys)] = machine
    return machine


def _read_date(rec, strip, after_label=False):
    """(date as 'd/m/y', mean confidence of its figures) read from a strip of the page by one of the
    recognisers, or None. The model's character scores are decoded with the date grammar
    (_date_machine): the most likely reading that is a date, rather than the most likely characters
    ('1R 8 25.' -> '12/8/25', '10 1 7 1 25' -> '10/7/25'). after_label: the strip starts with the
    printed label, which is skipped (everything up to the label's end as the model reads it)."""
    keys = rec.postprocess_op.character
    ratio = max(LINE_MIN_WIDTH, strip.shape[1] / strip.shape[0])
    pixels = rec.resize_norm_img(strip, ratio)[None].astype(np.float32)
    probs = np.asarray(rec.session(pixels))[0]
    if after_label:
        best = probs.argmax(axis=1)
        frames = [t for t in range(len(best)) if best[t] and (t == 0 or best[t] != best[t - 1])]
        text = "".join(keys[best[t]] for t in frames)
        label = _DATE_WORD.search(text) or re.search(r"[:;]", text[:8])
        if not label:
            return None
        start = frames[label.end() - 1] + 1
        while start < len(best) and best[start] == best[start - 1]:
            start += 1
        probs = probs[start:]
    if len(probs) < 3:
        return None
    cols, moves, part, ends = _date_machine(keys)
    q, a = moves.shape
    width = a + 1  # a state is (machine state, last column: 0 blank, 1.. a character)
    lp = np.log(np.maximum(probs[:, cols], 1e-12))
    score = np.full((q, width), -1e18)
    score[0, 0] = 0.0
    back = np.zeros((len(lp), q * width), dtype=np.int64)
    src_q, src_c = np.nonzero(moves >= 0)
    dst = moves[src_q, src_c] * width + src_c + 1
    stay = np.arange(q * width).reshape(q, width)
    for t in range(len(lp)):
        top = np.argsort(score, axis=1)[:, -2:]  # the best two columns of every state
        new = np.empty_like(score)
        new[:, 0] = score[np.arange(q), top[:, 1]] + lp[t, 0]  # blank
        new[:, 1:] = score[:, 1:] + lp[t, 1:]  # the same character again
        prev = stay.copy()
        prev[:, 0] = np.arange(q) * width + top[:, 1]
        # a new character: from the state's best column other than this character
        from_c = np.where(top[src_q, 1] == src_c + 1, top[src_q, 0], top[src_q, 1])
        cand = score[src_q, from_c] + lp[t, src_c + 1]
        flat, prev = new.ravel(), prev.ravel()
        np.maximum.at(flat, dst, cand)
        won = cand >= flat[dst]
        prev[dst[won]] = src_q[won] * width + from_c[won]
        score = flat.reshape(q, width)
        back[t] = prev
    final = score[ends].max()
    if final < -1e17:
        return None
    s = int(np.argmax(np.where(np.isin(np.arange(q * width) // width, ends), score.ravel(), -np.inf)))
    path = np.empty(len(lp), dtype=np.int64)
    for t in range(len(lp) - 1, -1, -1):
        path[t] = s
        s = back[t][s]
    chars, confs = [], []
    for t in range(len(lp)):
        state, c = divmod(int(path[t]), width)
        if not c:
            continue
        p = float(probs[t, cols[c]])
        if t and path[t - 1] == path[t]:
            confs[-1] = max(confs[-1], p)  # the same character, one frame further
        elif part[state] == 1:
            chars.append(keys[cols[c]])
            confs.append(p)
        else:
            chars.append(None)
            confs.append(p)
    figures = [(ch, p) for ch, p in zip(chars, confs) if ch is not None and ch.isdigit()]
    groups = re.findall(r"\d+", "".join(ch if ch and ch.isdigit() else " " for ch in chars))
    if len(groups) != 3 or not figures:
        return None
    return "/".join(groups), float(np.mean([p for _, p in figures]))


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
    # the text with a space before every cut that has none yet; the letters' confidences (info.confs,
    # one per character of the text) keep in step, a space put in counting as sure
    letters = list(getattr(info, "confs", None) or [])
    if len(letters) != len(text):
        letters = None
    out, sure, j = [], [], 0
    for i, ch in enumerate(text):
        if not ch.isspace():
            if j in cuts and out and not out[-1].isspace():
                out.append(" ")
                sure.append(1.0)
            j += 1
        out.append(ch)
        sure.append(letters[i] if letters else 1.0)
    new_text = "".join(out)
    if letters:
        info.confs = sure
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
    unplaced = types.SimpleNamespace(line_txt_len=0)  # a date read again: its words are spread evenly
    located = engine.cal_rec_boxes(crops, boxes, TextRecOutput(
        txts=tuple(t for t, _, _ in read), word_results=tuple(unplaced if p is None else p for _, _, p in read)))
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
