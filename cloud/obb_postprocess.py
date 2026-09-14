"""
obb_postprocess.py — everything Ultralytics does AFTER the network, in NumPy.

WHY THIS FILE EXISTS
    The exported ONNX graph has metadata `end2end: False`. It gives you a raw
    tensor of shape (1, 6, 8400) and nothing else. Ultralytics does confidence
    filtering, rotated non-maximum suppression, and the conversion to corner
    points in PYTHON, after the model runs. None of that ships in the .onnx.

    We cannot just `pip install ultralytics` in the container, because it pulls
    in torch — the 2 GB dependency this whole exercise exists to escape. So the
    postprocessing gets reimplemented here against numpy only.

    Dependencies: numpy, cv2. That is the entire list.

WHAT THE 8400 NUMBERS ARE
    YOLO divides a 640x640 image into three grids, one per detection scale:
        80x80 = 6400   (stride 8,  small objects)
        40x40 = 1600   (stride 16, medium)
        20x20 =  400   (stride 32, large)
        ---------------
                8400 candidate positions

    Every one of those positions predicts a box, whether or not anything is
    there. So the model always returns 8400 candidates and it is postprocessing's
    job to throw away the ~8399 that are noise.

THE 6 CHANNELS
    index 0..3 : cx, cy, w, h   — already decoded and in 640x640 pixel space
    index 4    : class score    — sigmoid already applied inside the graph
    index 5    : angle          — radians, range [-pi/4, 3pi/4]

    The class count here is 1 ('test_device'), which is why 4 + 1 + 1 = 6.
"""

from __future__ import annotations

import math

import numpy as np

# Must match python/detect.py's model.predict(imgsz=640, conf=0.4, iou=0.5).
IMGSZ = 640
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
PAD_COLOR = (114, 114, 114)   # Ultralytics' letterbox grey


# ==========================================================================
# 1. PREPROCESSING — letterbox
# ==========================================================================
def letterbox(img_bgr: np.ndarray, size: int = IMGSZ):
    """Resize preserving aspect ratio, then pad to a `size` x `size` square.

    Why not just cv2.resize to 640x640? Because that stretches the image.
    A cassette photographed in portrait would be squashed horizontally, and
    the detector was trained on correctly-proportioned objects. Letterboxing
    scales by a single factor and fills the leftover space with grey.

    NOTE ON A REAL DIFFERENCE FROM THE DESKTOP APP
        Ultralytics' predict() on a single PyTorch model uses `auto=True`,
        which pads only to the next multiple of 32 — so a 4:3 photo becomes
        640x480, not 640x640. Our ONNX graph is frozen at 640x640, so we must
        pad to the full square. The extra grey padding is a genuine difference
        between the two paths. verify_obb.py measures whether it matters.

    Returns (padded_image, scale_ratio, (pad_x, pad_y)).
    """
    import cv2

    h0, w0 = img_bgr.shape[:2]
    r = min(size / h0, size / w0)

    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Centre the image; split the leftover space evenly on both sides.
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=PAD_COLOR)

    # RETURN THE PADDING THAT WAS ACTUALLY APPLIED, not the ideal float.
    #
    # Bug this fixes: we pad by an INTEGER number of pixels (you cannot add
    # half a pixel to an image), but the first version returned the float
    # dw/dh. Un-padding by 80.5 when 80 was added leaves a half-pixel offset,
    # which gets multiplied by 1/ratio when scaling back to the original
    # image — a 1-2px shift on a downscaled photo.
    #
    # It was invisible in the confidence scores (identical to 5 decimal
    # places) and only showed up as polygon IoU ~0.96 on the images where
    # the rounding happened to land badly. A good reminder that "the numbers
    # look close" is not the same as "the code is right".
    return padded, r, (float(left), float(top))


def preprocess(img_bgr: np.ndarray, size: int = IMGSZ):
    """Full detector input prep: letterbox -> RGB -> 0..1 -> NCHW float32.

    YOLO does NOT use ImageNet mean/std normalization — just a divide by 255.
    Applying mean/std here (a natural assumption if you have come from
    torchvision) would quietly wreck the detections.
    """
    import cv2

    padded, r, pad = letterbox(img_bgr, size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None, ...], r, pad


# ==========================================================================
# 2. ROTATED IoU — probiou
# ==========================================================================
def _covariance(boxes: np.ndarray):
    """Treat each rotated box as a 2-D Gaussian and return its covariance.

    Exact polygon intersection for rotated rectangles is fiddly and slow.
    Ultralytics instead approximates each box by a Gaussian with the same
    centre, spread and orientation, then measures similarity between the two
    distributions. A w x h rectangle has variance w^2/12 along its own axes
    (the variance of a uniform distribution), which is where the /12 comes from.

    boxes: (N, 5) as cx, cy, w, h, angle
    returns a, b, c — the entries of [[a, c], [c, b]]
    """
    w = boxes[:, 2]
    h = boxes[:, 3]
    r = boxes[:, 4]

    a_ = (w ** 2) / 12.0
    b_ = (h ** 2) / 12.0

    cos = np.cos(r)
    sin = np.sin(r)
    cos2 = cos ** 2
    sin2 = sin ** 2

    # Rotate the axis-aligned covariance into the box's own frame.
    a = a_ * cos2 + b_ * sin2
    b = a_ * sin2 + b_ * cos2
    c = (a_ - b_) * cos * sin
    return a, b, c


def probiou(obb1: np.ndarray, obb2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Pairwise rotated-IoU approximation via the Bhattacharyya distance.

    Returns an (N, M) matrix of similarities in [0, 1]. This is a port of
    Ultralytics' probiou so that our NMS keeps exactly the boxes theirs would.

    The three terms below are the standard Bhattacharyya distance between two
    2-D Gaussians: t1 and t2 measure how far apart the centres are relative to
    the spread, t3 measures how differently shaped they are. Distance is then
    mapped to a similarity via the Hellinger distance.
    """
    x1 = obb1[:, 0][:, None]
    y1 = obb1[:, 1][:, None]
    x2 = obb2[:, 0][None, :]
    y2 = obb2[:, 1][None, :]

    a1, b1, c1 = (v[:, None] for v in _covariance(obb1))
    a2, b2, c2 = (v[None, :] for v in _covariance(obb2))

    dx = x2 - x1
    dy = y1 - y2

    denom = (a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps

    t1 = (((a1 + a2) * dy ** 2 + (b1 + b2) * dx ** 2) / denom) * 0.25
    t2 = (((c1 + c2) * dx * dy) / denom) * 0.5

    det1 = np.clip(a1 * b1 - c1 ** 2, 0.0, None)
    det2 = np.clip(a2 * b2 - c2 ** 2, 0.0, None)
    t3 = 0.5 * np.log(denom / (4.0 * np.sqrt(det1 * det2) + eps) + eps)

    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1.0 - hd


def nms_rotated(boxes: np.ndarray, scores: np.ndarray,
                iou_threshold: float = IOU_THRESHOLD) -> list[int]:
    """Greedy NMS using rotated IoU.

    The model fires on the same cassette from several neighbouring grid cells,
    so without this you get five overlapping boxes around one object. Take the
    highest-scoring box, discard everything overlapping it beyond the threshold,
    repeat.
    """
    order = np.argsort(-scores)
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = probiou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious < iou_threshold]

    return keep


# ==========================================================================
# 3. GEOMETRY
# ==========================================================================
def xywhr_to_polygon(boxes: np.ndarray) -> np.ndarray:
    """(N,5) cx,cy,w,h,angle -> (N,4,2) corner points.

    Build two edge vectors from the centre — one along the box's width, one
    along its height — then add and subtract them to reach the four corners.
    Produces the same ordering as Ultralytics' xywhr2xyxyxyxy, which matters
    because detect.py's _order_polygon and rotated_crop consume these points.
    """
    ctr = boxes[:, :2]
    w = boxes[:, 2:3]
    h = boxes[:, 3:4]
    r = boxes[:, 4:5]

    cos = np.cos(r)
    sin = np.sin(r)

    vec1 = np.concatenate([w / 2 * cos, w / 2 * sin], axis=-1)
    vec2 = np.concatenate([-h / 2 * sin, h / 2 * cos], axis=-1)

    return np.stack([ctr + vec1 + vec2,
                     ctr + vec1 - vec2,
                     ctr - vec1 - vec2,
                     ctr - vec1 + vec2], axis=-2)


def scale_polygons(polys: np.ndarray, ratio: float, pad: tuple[float, float],
                   orig_shape: tuple[int, int]) -> np.ndarray:
    """Undo the letterbox: map 640x640 coordinates back to the original image.

    Subtract the padding offset, then divide by the scale factor. Angles need
    no correction because letterboxing scales both axes equally.

    DELIBERATELY NOT CLIPPED TO THE IMAGE BOUNDS.

    The first version clamped every corner into [0, w] x [0, h], which looks
    like sensible defensive coding and is actively wrong for ORIENTED boxes.

    Clamp the corners of an axis-aligned box and you get a smaller
    axis-aligned box — fine. Clamp one corner of a ROTATED box and the shape
    stops being a rectangle: three corners stay put, one slides along the
    edge, and you are left with an arbitrary quadrilateral. On demo_examples
    /invalid/invalid_05.png (a 146x541 strip where the cassette runs past the
    frame edge) this pulled one corner from x=153.04 to x=146.00, shrinking
    the reported height by 7px and the area by 625px^2 — the sole reason that
    image scored IoU 0.967 instead of 1.0 against Ultralytics.

    Clipping is also unnecessary: rotated_crop warps with BORDER_REPLICATE,
    which handles corners outside the frame without complaint. Ultralytics
    does not clip oriented boxes either.
    """
    out = polys.copy()
    out[..., 0] -= pad[0]
    out[..., 1] -= pad[1]
    out /= ratio
    return out


# ==========================================================================
# 4. THE WHOLE THING
# ==========================================================================
def postprocess(raw: np.ndarray, ratio: float, pad: tuple[float, float],
                orig_shape: tuple[int, int],
                conf_threshold: float = CONF_THRESHOLD,
                iou_threshold: float = IOU_THRESHOLD) -> list[dict]:
    """Raw ONNX output -> list of detections in ORIGINAL image coordinates.

    Each detection: {confidence, polygon (4x2), xywhr, class_id}
    Sorted by confidence, highest first.
    """
    # (1, 6, 8400) -> (8400, 6). The transpose matters: the raw layout is
    # channels-first, and getting this backwards produces 6 "detections"
    # with 8400 attributes each, which fails in a confusing way later.
    pred = raw[0].T

    boxes = pred[:, 0:4]        # cx, cy, w, h
    scores = pred[:, 4]         # single class, sigmoid already applied
    angles = pred[:, 5:6]       # radians

    mask = scores >= conf_threshold
    if not np.any(mask):
        return []

    boxes = boxes[mask]
    scores = scores[mask]
    angles = angles[mask]

    xywhr = np.concatenate([boxes, angles], axis=1)

    keep = nms_rotated(xywhr, scores, iou_threshold)
    xywhr = xywhr[keep]
    scores = scores[keep]

    polys = xywhr_to_polygon(xywhr)
    polys = scale_polygons(polys, ratio, pad, orig_shape)

    # The box dimensions also live in letterboxed space — rescale them so the
    # reported xywhr describes the original image too.
    xywhr_orig = xywhr.copy()
    xywhr_orig[:, 0] -= pad[0]
    xywhr_orig[:, 1] -= pad[1]
    xywhr_orig[:, :4] /= ratio

    return [
        {
            "confidence": float(scores[i]),
            "polygon": polys[i].tolist(),
            "xywhr": xywhr_orig[i].tolist(),
            "class_id": 0,
        }
        for i in range(len(scores))
    ]


def detect(session, img_bgr: np.ndarray,
           conf_threshold: float = CONF_THRESHOLD,
           iou_threshold: float = IOU_THRESHOLD) -> list[dict]:
    """End-to-end: BGR image in, detections out.

    `session` is a live onnxruntime.InferenceSession for detector.onnx.
    Build it once at startup and reuse it — constructing a session per request
    re-optimizes the graph every time and is the single easiest way to make a
    fast model look slow.
    """
    blob, ratio, pad = preprocess(img_bgr)
    input_name = session.get_inputs()[0].name
    raw = session.run(None, {input_name: blob})[0]
    return postprocess(raw, ratio, pad, img_bgr.shape[:2],
                       conf_threshold, iou_threshold)
