"""Tesseract OCR wrapper. Returns words with normalised boxes and confidence."""
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from . import config
from .models import Word


class OCRUnavailableError(RuntimeError):
    pass


def _configure_tesseract():
    cmd = config.TESSERACT_CMD if Path(config.TESSERACT_CMD).exists() else shutil.which("tesseract")
    if not cmd:
        raise OCRUnavailableError(
            "Tesseract OCR was not found. Install it (see README) or set TESSERACT_CMD to tesseract.exe."
        )
    pytesseract.pytesseract.tesseract_cmd = cmd


def ocr_image(img, page_index: int, tess_config=None):
    """Run OCR on a numpy image. Returns (words, mean_confidence)."""
    _configure_tesseract()
    h, w = img.shape[:2]
    data = pytesseract.image_to_data(img, lang=config.OCR_LANG,
                                     config=tess_config or config.OCR_CONFIG, output_type=Output.DICT)
    found, doubtful = [], []
    for i, raw in enumerate(data["text"]):
        text = raw.strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if not text or conf < 0:
            continue
        box = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
        if (conf < REREAD_BELOW and box[3] >= 10 and box[2] >= box[3] and any(c.isalnum() for c in text)
                and len(doubtful) < MAX_REREADS):
            doubtful.append(len(found))
        found.append([text, conf, box])
    def as_words():
        return [Word(text, left / w, top / h, (left + width) / w, (top + height) / h, conf / 100.0, page_index)
                for text, conf, (left, top, width, height) in found]

    # The two follow-up readings (doubtful words, missed letterhead) are separate Tesseract runs. They
    # run at the same time: the letterhead strip does not depend on the re-read words, and re-reading
    # only raises confidences, so a strip that is not needed before it is not needed after it either.
    header = None
    with ThreadPoolExecutor(2) as pool:
        if _missed_header_boxes(img, as_words()):
            header = pool.submit(_read_header_strip, img)
        rereads = _reread_all(img, [found[k] for k in doubtful])
        for k, (text, conf) in zip(doubtful, rereads):
            found[k][0], found[k][1] = text, conf
        words = _add_missed_header_words(img, as_words(), page_index, header)
    mean_conf = float(np.mean([conf for _, conf, _ in found])) / 100.0 if found else 0.0
    return words, mean_conf


def best_quarter_turn(gray, turns):
    """Of the given turns (np.rot90 k), the one under which the page's text reads best. A dense
    sample of the page is turned each way and the copies are read in a single OCR call."""
    from .preprocess import text_sample

    _configure_tesseract()
    sample = text_sample(gray)
    parts = [np.rot90(sample, k) for k in turns]
    width = max(p.shape[1] for p in parts)
    blocks, spans, y = [], [], 0
    for p in parts:
        block = cv2.copyMakeBorder(p, 30, 30, 20, 20 + width - p.shape[1], cv2.BORDER_CONSTANT, value=255)
        blocks.append(block)
        spans.append((y, y + block.shape[0]))
        y += block.shape[0]
    try:
        data = pytesseract.image_to_data(np.vstack(blocks), lang=config.OCR_LANG, config="--oem 3 --psm 6",
                                         output_type=Output.DICT)
    except pytesseract.TesseractError:
        return turns[0]
    scores = [0.0] * len(parts)
    for t, c, top, hh in zip(data["text"], data["conf"], data["top"], data["height"]):
        c = float(c)
        if c < 0 or sum(ch.isalnum() for ch in t) < 2:
            continue
        mid = top + hh / 2
        for i, (a, b) in enumerate(spans):
            if a <= mid < b:
                scores[i] += max(0.0, c - 50)  # confident words count; garbage from upside-down text does not
    return turns[int(np.argmax(scores))]


HEADER_PART = 0.15  # top part of the page where the company name / letterhead is printed


def _missed_header_boxes(img, words):
    """Tesseract sometimes returns nothing at all for a small letterhead next to a big title
    ('SuperStore' beside 'INVOICE' on phone photos). If the top of the page has printed text that no
    confident OCR word covers, that strip is read again as sparse text and only the missing words are
    added. Costs nothing when nothing was missed (one quick ink check).

    This part returns the boxes of printed marks in the top strip that no confident word covers."""
    h, w = img.shape[:2]
    top = int(HEADER_PART * h)
    heights = sorted((wd.y1 - wd.y0) * h for wd in words if wd.conf >= 0.6)
    if top < 20 or not heights:
        return []
    text_h = heights[len(heights) // 2]
    gray = img[:top] if img.ndim == 2 else cv2.cvtColor(img[:top], cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(0.6 * text_h)), 1)))
    good = [(wd.x0 * w, wd.y0 * h, wd.x1 * w, wd.y1 * h) for wd in words if wd.conf >= 0.5 and wd.y0 * h < top]
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    missed = []
    for i in range(1, n):
        x, y, bw, bh = (int(v) for v in stats[i, :4])
        if not (0.5 * text_h <= bh <= 3 * text_h) or bw < 1.5 * bh or y + bh >= top - 2:
            continue  # not a word-shaped mark, or cut off by the strip's edge
        if not any(min(x + bw, x1) > max(x, x0) and min(y + bh, y1) > max(y, y0) for x0, y0, x1, y1 in good):
            missed.append((x, y, x + bw, y + bh))  # no confident word touches this printed text
    return missed


def _read_header_strip(img):
    """Tesseract's sparse-text reading of the top strip, or None."""
    try:
        return pytesseract.image_to_data(img[: int(HEADER_PART * img.shape[0])], lang=config.OCR_LANG,
                                         config="--oem 3 --psm 11", output_type=Output.DICT)
    except pytesseract.TesseractError:
        return None


def _add_missed_header_words(img, words, page_index, strip=None):
    """Add the letterhead words that only the strip reading found (strip: a Future of
    _read_header_strip started earlier, or None to read the strip now when needed)."""
    h, w = img.shape[:2]
    missed = _missed_header_boxes(img, words)
    if not missed:
        return words
    data = strip.result() if strip is not None else _read_header_strip(img)
    if data is None:
        return words
    added = []
    for t, c, l, tp, ww, hh in zip(data["text"], data["conf"], data["left"], data["top"], data["width"], data["height"]):
        t, c = t.strip(), float(c)
        if not t or c < 60 or not any(ch.isalnum() for ch in t):
            continue
        cx, cy = l + ww / 2, tp + hh / 2
        if any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in missed):
            added.append(Word(t, l / w, tp / h, (l + ww) / w, (tp + hh) / h, c / 100.0, page_index))
    if not added:
        return words

    def overlaps(a, b):
        return min(a.x1, b.x1) > max(a.x0, b.x0) and min(a.y1, b.y1) > max(a.y0, b.y0)

    confident = [wd for wd in words if wd.conf >= 0.5]
    added = [a for a in added if not any(overlaps(a, wd) for wd in confident)]  # never a second copy
    if not added:
        return words
    # a doubtful word at the same spot ("once" where 'SuperStore' is printed) is replaced, not kept twice
    kept = [wd for wd in words if wd.conf >= 0.5 or not any(overlaps(wd, a) for a in added)]
    return kept + added


REREAD_BELOW = 35  # Tesseract confidence (0-100) under which a word is read again on its own
MAX_REREADS = 8    # per page


def _reread_all(img, doubtful):
    """Read doubtful words again, each cut out on its own. Tesseract reads a line at a time; a small
    word on the same line as a big title ('SuperStore' next to 'INVOICE') is then misread ('once',
    15%), while the same word cut out and read alone is read correctly. The cut-outs are stacked
    one under another and read in one OCR call: starting Tesseract once per word cost up to 8 s on
    a busy PC. Returns (text, conf) for each word, the original when the new reading is no better."""
    if not doubtful:
        return []
    pad, gap = 3, 24  # a tight crop: more margin would pull in neighbouring text ("Bill To:" above a speck)
    crops = []
    for _, _, (left, top, width, height) in doubtful:
        y0, x0 = max(0, top - pad), max(0, left - pad)
        crops.append(img[y0: top + height + pad, x0: left + width + pad])
    canvas_w = max(c.shape[1] for c in crops) + 2 * gap
    rows, spans, y = [], [], 0
    for c in crops:
        row = cv2.copyMakeBorder(c, gap, gap, gap, canvas_w - gap - c.shape[1], cv2.BORDER_CONSTANT, value=255)
        rows.append(row)
        spans.append((y, y + row.shape[0]))
        y += row.shape[0]
    try:
        data = pytesseract.image_to_data(np.vstack(rows), lang=config.OCR_LANG, config="--oem 3 --psm 6",
                                         output_type=Output.DICT)
    except pytesseract.TesseractError:
        return [(text, conf) for text, conf, _ in doubtful]
    parts = [[] for _ in doubtful]
    for t, c, lf, tp, hh in zip(data["text"], data["conf"], data["left"], data["top"], data["height"]):
        t, c = t.strip(), float(c)
        if not t or c < 0:
            continue
        mid = tp + hh / 2
        for i, (a, b) in enumerate(spans):
            if a <= mid < b:
                parts[i].append((lf, t, c))
    out = []
    for (text, conf, (_, _, width, height)), got in zip(doubtful, parts):
        if not got:
            out.append((text, conf))
            continue
        got.sort()
        new_text = " ".join(t for _, t, _ in got)
        new_conf = sum(c for _, _, c in got) / len(got)
        fits = len(new_text) <= max(4, 1.6 * width / (0.5 * height))  # about as many letters as the box holds
        if new_conf >= 60 and new_conf > conf + 20 and fits and any(ch.isalnum() for ch in new_text):
            out.append((new_text, new_conf))
        else:
            out.append((text, conf))
    return out
