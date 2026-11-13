"""Image clean-up before OCR.

* Phone photos: find the page's four corners and warp it flat (perspective correction), and turn
  a page photographed sideways upright (upside-down pages are handled after OCR, in ingest.py).
* Everything: bring the page to a good OCR size, even out shadows and uneven light,
  straighten small skew, keep it grayscale.

Measured on the synthetic test sets (scripts/make_sample_invoices.py --scans --photos):
* Binarisation is left to Tesseract. Our own thresholding (median blur + adaptive threshold)
  dropped field accuracy on scans from 100% to 34%.
* Light flattening took phone photos with uneven lighting from 20% to 100%.
"""
import cv2
import numpy as np

from . import config

MIN_OCR_SIDE = 2000  # small images are enlarged: Tesseract wants text roughly 20+ px tall


def preprocess_scan(img, photo: bool = False, text_px=None):
    """Return a grayscale image ready for Tesseract.

    img   : BGR (or gray) numpy array
    photo : True for camera photos and uploaded images (enables page-edge detection)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    if photo:
        quad = _find_document_quad(gray)
        if quad is not None:
            gray = _four_point_warp(gray, quad)

    gray = _flatten_light(gray)
    if photo:
        gray = _upright(gray)  # a page photographed sideways (phone held flat, no rotation tag)
    gray = _invert_dark_regions(gray)  # after flattening: shadows are gone, header bars are still dark
    gray = _deskew(_resize_for_ocr(gray, text_px or config.OCR_TEXT_PX))
    return _remove_ruled_lines(gray) if photo else gray


CAMERA_WORK_SIDE = 2600  # phone photos are cut out on a copy this size (the fast reader reads 2000 px)


def prepare_camera_photo(img):
    """A phone photo for the fast neural reader (fast_ocr.py): the page cut out of the photo and
    warped flat, and a page photographed sideways given a quarter turn. Returns (image, turned):
    after a quarter turn the text may still be upside down, which the reader checks.

    Nothing else is done: the neural models read shadows, small tilts, ruled lines and white-on-dark
    text as they are, so the steps Tesseract needs (light flattening, deskewing, line removal,
    enlarging) would only cost time."""
    color = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    longest = max(color.shape[:2])
    if longest > CAMERA_WORK_SIDE:
        f = CAMERA_WORK_SIDE / longest
        color = cv2.resize(color, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    quad = _find_document_quad(gray)
    if quad is not None and not _paper_around(gray, quad):
        color = _four_point_warp(color, quad)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    if _text_runs_vertically(gray):  # photographed sideways (phone held flat, no rotation tag)
        return np.ascontiguousarray(np.rot90(color, 1)), True
    return color, False


def _paper_around(gray, quad):
    """True when the outline found is a box printed on the page (a bill book's border, on a scan or
    a photo filling the frame) rather than the page's edge: the image around it is as bright as the
    paper inside. Cutting the page out along such a box would drop the header printed above it
    (the seller's name and GSTIN)."""
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    inside, outside = gray[mask > 0], gray[mask == 0]
    if outside.size < 0.02 * gray.size:
        return False
    return float(np.median(outside)) >= 0.9 * float(np.median(inside))


def _remove_ruled_lines(gray):
    """Erase long horizontal / vertical table lines. Box borders touching the text confuse OCR on
    form-style invoices (Tally etc.): on real phone photos this raised correct key values from 7-8
    to 9 of 11 and made OCR faster."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = bw.shape
    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 25), 1)))
    vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 35))))
    mask = cv2.dilate(cv2.bitwise_or(horizontal, vertical), np.ones((3, 3), np.uint8))
    out = gray.copy()
    out[mask > 0] = 255
    return out


def _resize_for_ocr(gray, target_px=None):
    """Scale the page so the median letter is about target_px tall: large enough for Tesseract
    to read reliably, small enough to stay fast (OCR time grows with the number of pixels)."""
    target_px = target_px or config.OCR_TEXT_PX
    longest = max(gray.shape[:2])
    letter = _median_letter_height(gray)
    if letter:
        scale = min(max(target_px / letter, 0.4), 2.5)
    else:  # no clear text found: fall back to the page size
        scale = MIN_OCR_SIDE / longest if longest < MIN_OCR_SIDE else 1.0
    scale = min(scale, config.MAX_OCR_SIDE / longest)
    if 0.92 <= scale <= 1.08:
        return gray
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interpolation)


def _median_letter_height(gray):
    """Median height (px) of letter-like ink blobs, measured on a small copy; None if unclear."""
    h, w = gray.shape[:2]
    s = min(1.0, 1600 / max(h, w))
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    widths = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    letters = (heights >= 3) & (heights <= 60) & (widths <= 2.5 * heights + 3)
    if letters.sum() < 30:
        return None
    return float(np.median(heights[letters])) / s


def _invert_dark_regions(gray):
    """White text on a dark bar (a common table-header design: 'Item  Quantity  Rate  Amount') is not
    read by Tesseract. Areas whose surroundings are mostly dark are inverted to dark text on white."""
    h, w = gray.shape[:2]
    scale = min(1.0, 1000 / max(h, w))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    local = cv2.medianBlur(small, 15)  # text strokes are thin: the median is the background of the area
    dark = (local < 100).astype(np.uint8)
    if dark.mean() < 0.002:
        return gray  # nothing dark: most invoices
    # only solid dark bands at least ~8% of the page wide (header bars, dark boxes), not single marks
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark)
    out = gray
    margin = int(10 / scale)
    for i in range(1, n):
        x, y, bw, bh = (stats[i, k] for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH,
                                               cv2.CC_STAT_HEIGHT))
        if bw < 0.08 * small.shape[1] or bh < 6:
            continue
        if bh > 0.2 * small.shape[0] or bw * bh > 0.25 * small.size:
            continue  # too big for a header bar: a shadow or a dark photo background
        x0, y0 = max(0, int(x / scale) - margin), max(0, int(y / scale) - margin)
        x1, y1 = min(w, int((x + bw) / scale) + margin), min(h, int((y + bh) / scale) + margin)
        region = gray[y0:y1, x0:x1]
        _, dark_px = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # the bar is the biggest dark shape; the white letters are holes inside it (tilted or rounded bars too)
        m, lab, st, _ = cv2.connectedComponentsWithStats(dark_px)
        if m < 2:
            continue
        biggest = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        bar = np.where(lab == biggest, 255, 0).astype(np.uint8)
        outside = cv2.copyMakeBorder(bar, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        cv2.floodFill(outside, None, (0, 0), 255)  # everything reachable from the edge is not a hole
        mask = (bar > 0) | (outside[1:-1, 1:-1] == 0)
        if out is gray:
            out = gray.copy()
        sub = out[y0:y1, x0:x1]
        flipped = 255 - sub[mask].astype(np.float32)
        # the bar's background becomes paper-white (not grey), so the table-line remover never takes the
        # whole bar for a thick line and wipes the header text with it
        background = max(1.0, float(np.median(flipped)))
        sub[mask] = np.clip(flipped * (255.0 / background), 0, 255).astype(np.uint8)
    return out


def _upright(gray):
    """Turn a page that was photographed sideways. Phones often store a photo of a document lying
    flat on a desk without a rotation tag, so the text runs top-to-bottom. The direction of the text
    lines is measured from the letters (cheap); only for a sideways page is a small sample read twice
    to choose between a quarter turn left or right."""
    if not _text_runs_vertically(gray):
        return gray
    from .ocr import best_quarter_turn  # OCR helper; imported here to keep preprocessing OCR-free

    return np.ascontiguousarray(np.rot90(gray, best_quarter_turn(gray, (1, 3))))


def _letter_blobs(gray, longest=1500):
    """Centres of letter-sized ink blobs on a small copy, and that copy's scale."""
    h, w = gray.shape[:2]
    s = min(1.0, longest / max(h, w))
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, _, stats, centres = cv2.connectedComponentsWithStats(bw, connectivity=8)
    hgt, wid, area = stats[1:, cv2.CC_STAT_HEIGHT], stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_AREA]
    keep = (hgt >= 3) & (wid >= 2) & (hgt <= 30) & (wid <= 30) & (area >= 6)
    return centres[1:][keep], s


def _text_runs_vertically(gray):
    """True when the text lines run up/down the image. Each letter's nearest neighbour is the next
    letter of its word: beside it in upright text, above or below it in sideways text (letters
    are closer together than lines are). Tolerates tilts far beyond what deskewing handles."""
    pts, _ = _letter_blobs(gray)
    if len(pts) < 80:
        return False  # too little text to judge
    # OpenCV's own k-d tree (FLANN): importing scipy for this took up to 13 s on a busy PC
    pts = np.ascontiguousarray(pts, dtype=np.float32)
    idx, _ = cv2.flann_Index(pts, dict(algorithm=1, trees=4)).knnSearch(pts, 2, params=dict(checks=64))
    v = np.abs(pts[idx[:, 1]] - pts)
    angle = np.degrees(np.arctan2(v[:, 1], v[:, 0]))  # 0 = neighbour beside, 90 = above / below
    beside, above = int((angle < 30).sum()), int((angle > 60).sum())
    return above > 1.5 * beside


def text_sample(gray, size=900, letter_px=22):
    """The most text-dense square of the page, scaled so letters are about letter_px tall: a quick
    sample for checking which way up the page is."""
    pts, s = _letter_blobs(gray)
    letter = _median_letter_height(gray) or 25.0
    k = letter_px / letter
    side = int(size / k)  # sample side in original pixels
    h, w = gray.shape[:2]
    if len(pts):
        grid = np.zeros((int(h * s) + 1, int(w * s) + 1), np.float32)
        np.add.at(grid, (pts[:, 1].astype(int), pts[:, 0].astype(int)), 1)
        win = max(3, int(side * s))
        density = cv2.boxFilter(grid, -1, (win, win), normalize=False)
        cy, cx = np.unravel_index(int(np.argmax(density)), density.shape)
        cx, cy = cx / s, cy / s
    else:
        cx, cy = w / 2, h / 2
    x0 = int(min(max(0, cx - side / 2), max(0, w - side)))
    y0 = int(min(max(0, cy - side / 2), max(0, h - side)))
    crop = gray[y0: y0 + side, x0: x0 + side]
    return cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)


def _flatten_light(gray):
    """Even out shadows and uneven lighting. A large morphological closing removes the text and
    leaves only the paper's brightness; dividing by it makes the paper evenly white."""
    k = max(31, (min(gray.shape[:2]) // 25) | 1)
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return cv2.divide(gray, background, scale=255)


def _find_document_quad(gray):
    h, w = gray.shape[:2]
    scale = 1000 / max(h, w)  # edge detection on a small copy is fast and just as reliable
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    scale = min(scale, 1.0)
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(blur, 50, 150), np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = small.shape[0] * small.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(contour) < 0.25 * area:
            break  # the document should fill a good part of the photo
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            if cv2.contourArea(approx) > 0.97 * area:
                return None  # the "page" is the whole image: nothing to flatten
            return _order_points(approx.reshape(4, 2).astype("float32") / scale)
    return None


def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]  # top-right
    rect[3] = pts[np.argmax(d)]  # bottom-left
    return rect


def _four_point_warp(gray, rect):
    tl, tr, br, bl = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(gray, matrix, (width, height), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def _line_angle(gray):
    """Tilt (degrees) measured from long, nearly horizontal straight lines (table borders, rules).
    Photos of invoices usually have them; None when there are too few to trust."""
    h, w = gray.shape
    scale = min(1.0, 1200 / max(h, w))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    edges = cv2.Canny(small, 50, 150)
    found = cv2.HoughLinesP(edges, 1, np.pi / 1800, threshold=100, minLineLength=small.shape[1] // 4, maxLineGap=10)
    if found is None:
        return None
    angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in found.reshape(-1, 4)]
    angles = [a for a in angles if abs(a) < 6]
    if len(angles) < 5:
        return None
    return float(np.median(angles))


def _deskew(gray):
    """Straighten small tilts (within +-5 degrees). Uses the angle of long straight lines when the
    page has them (precise, and not fooled by a desk or keyboard around the paper); otherwise
    searches for the angle where text rows line up (coarse 1 degree, then 0.25 degree)."""
    h, w = gray.shape
    angle = _line_angle(gray)
    if angle is not None:
        if abs(angle) < 0.2:
            return gray
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    scale = 800 / max(h, w)
    small = cv2.resize(bw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bw
    sh, sw = small.shape

    def row_variance(angle):
        matrix = cv2.getRotationMatrix2D((sw / 2, sh / 2), float(angle), 1.0)
        rotated = cv2.warpAffine(small, matrix, (sw, sh))
        return float(np.var(rotated.sum(axis=1, dtype=np.float64)))

    best = max(np.arange(-5.0, 5.01, 1.0), key=row_variance)
    best = float(max(np.arange(best - 0.75, best + 0.76, 0.25), key=row_variance))
    if abs(best) < 0.25:
        return gray
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), best, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
