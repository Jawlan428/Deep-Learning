"""
pipeline.py — the whole LFT reading pipeline, with no torch and no ultralytics.

WHAT THIS IS
    A faithful port of python/detect.py's Stage-2 logic to a torch-free stack:

        onnxruntime + opencv + pillow + numpy

    Same three-part decision your desktop app makes:
        1. YOLOv8-OBB locates the cassette      (obb_postprocess.py)
        2. MobileNetV3 classifies the crop      (here)
        3. A classical red-line detector counts C/T bands, and the two
           signals are FUSED with a bias against missing a positive  (here)

TWO DELIBERATE DIFFERENCES FROM detect.py, both measured, both improvements
    Documented here because they change behaviour and must not be silent.

    1. THE CLASSIFIER RESIZES WITH PIL, NOT cv2.INTER_LINEAR.
       detect.py downscales crops with cv2.INTER_LINEAR, which does not
       antialias — it samples a 2x2 neighbourhood and discards everything
       between. A faint test line is exactly the thin, low-contrast feature
       that destroys. Measured on the 81-image held-out set:

           cv2.INTER_LINEAR (detect.py) : 0.8395   48/60 positives found
           PIL resize (as validated)    : 0.9630   60/60 positives found

       The model was never weak at faint positives. It was being shown
       images with the evidence already removed.

    2. NO 4-ROTATION PASS.
       detect.py classifies each crop at all four 90-degree rotations and
       keeps whichever is most confident. On correctly-oriented crops that
       LOSES accuracy, because three of the four rotations are out of
       distribution and produce confidently wrong answers:

           no rotation              : 0.9630
           4 rot, max-confidence    : 0.9259
           4 rot, mean probability  : 0.9012

       to_portrait() already normalises orientation, so the rotation pass is
       mostly adding noise. CAVEAT: the val set is made of correctly-oriented
       crops, so it cannot test the case the rotation was written for — a
       near-square OBB whose orientation is genuinely ambiguous. If that shows
       up in production, revisit this with data that contains it.

ALSO NOT PORTED
    The YOLOv8-cls fallback (python/classifier.pt). detect.py falls back to it
    if the MobileNet checkpoint is missing; here the MobileNet ONNX is always
    present, so the fallback is dead code in this context.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

import obb_postprocess as obb

log = logging.getLogger("lft.pipeline")

# ==========================================================================
# Tunables — copied verbatim from python/detect.py.
# If you calibrate these on new images, change them in BOTH files or the
# desktop app and the API will start disagreeing.
# ==========================================================================
ENHANCE_CLAHE_CLIP = 2.0
ENHANCE_CLAHE_GRID = 8
LINE_SAT_GAIN = 1.6
LINE_SAT_FLOOR = 35
LINE_VAL_FLOOR = 40
LINE_AREA_MIN = 12
LINE_WIDTH_FRAC_MIN = 0.10
LINE_HEIGHT_FRAC_MAX = 0.22
LINE_ASPECT_MIN = 1.2
LINE_FULL_WIDTH_FRAC = 0.60
LINE_MERGE_FRAC = 0.04
LINE_SIDE_MARGIN_FRAC = 0.12
LINE_MIN_SEPARATION_FRAC = 0.08
POSITIVE_DECISION_THRESHOLD = 0.40
CNN_OVERRIDE_INVALID_CONF = 0.65

CROP_PAD_FRAC = 0.04


# ==========================================================================
# GEOMETRY — ported from detect.py unchanged
# ==========================================================================
def order_polygon(poly) -> np.ndarray:
    """Put the 4 corners in a consistent order before warping.

    The detector emits corners in whatever order the box parameterization
    produced. Without normalising, the same physical cassette can come out
    upside down between runs. Anchor on the longest edge, then ensure the
    first edge is the top one.
    """
    poly = np.asarray(poly, dtype=np.float32).reshape(4, 2)
    edges = []
    for i in range(4):
        p1, p2 = poly[i], poly[(i + 1) % 4]
        edges.append((i, float(np.linalg.norm(p2 - p1))))
    edges.sort(key=lambda e: e[1], reverse=True)
    s = edges[0][0]
    ordered = np.array([poly[s], poly[(s + 1) % 4],
                        poly[(s + 2) % 4], poly[(s + 3) % 4]], dtype=np.float32)
    if (ordered[0][1] + ordered[1][1]) / 2 > (ordered[2][1] + ordered[3][1]) / 2:
        ordered = np.array([ordered[2], ordered[3], ordered[0], ordered[1]],
                           dtype=np.float32)
    return ordered


def rotated_crop(image_bgr: np.ndarray, polygon, pad_frac: float = CROP_PAD_FRAC) -> np.ndarray:
    """Perspective-warp a tilted cassette into an upright rectangle.

    BORDER_REPLICATE matters: the box legitimately runs past the image edge on
    tightly-framed photos, and replicating edge pixels is better than filling
    with black, which would look like a dark band to the line detector.
    """
    import cv2

    src = order_polygon(polygon)
    width = float(np.linalg.norm(src[1] - src[0]))
    height = float(np.linalg.norm(src[2] - src[1]))
    long_axis, short_axis = max(width, height), min(width, height)
    pad = int(round(pad_frac * long_axis))
    out_w = int(round(short_axis)) + 2 * pad
    out_h = int(round(long_axis)) + 2 * pad

    if width >= height:
        dst = np.array([[out_w - 1 - pad, pad],
                        [out_w - 1 - pad, out_h - 1 - pad],
                        [pad, out_h - 1 - pad],
                        [pad, pad]], dtype=np.float32)
    else:
        dst = np.array([[pad, pad],
                        [out_w - 1 - pad, pad],
                        [out_w - 1 - pad, out_h - 1 - pad],
                        [pad, out_h - 1 - pad]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image_bgr, M, (out_w, out_h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def to_portrait(crop_bgr: np.ndarray) -> np.ndarray:
    """Stand the cassette upright (taller than wide)."""
    import cv2
    h, w = crop_bgr.shape[:2]
    return cv2.rotate(crop_bgr, cv2.ROTATE_90_CLOCKWISE) if w > h else crop_bgr


# ==========================================================================
# CLASSIFIER
# ==========================================================================
class Classifier:
    """MobileNetV3-Small over ONNX Runtime, with training-identical preprocessing."""

    def __init__(self, onnx_path: Path, meta_path: Path, threads: int = 2):
        import onnxruntime as ort

        self.meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        self.classes: list[str] = self.meta["classes"]
        self.size: int = int(self.meta["img_size"])
        self.mean = np.array(self.meta["mean"], dtype=np.float32)
        self.std = np.array(self.meta["std"], dtype=np.float32)

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(onnx_path), so,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """BGR crop -> normalized NCHW tensor, resized the way TRAINING did.

        The resize goes through PIL rather than cv2. PIL's resize is
        area-averaged (antialiased), so shrinking a large crop to 224px lets
        every source pixel contribute. cv2.INTER_LINEAR does not, and it
        erases faint test lines — a measured 12-point accuracy loss.

        Measured separately: the JPEG DECODER makes no difference at all
        (cv2-decode + PIL-resize scored identically to PIL-decode +
        PIL-resize). Only the resize kernel mattered.
        """
        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((self.size, self.size), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        return arr.transpose(2, 0, 1)[None, ...].astype(np.float32)

    def classify(self, crop_bgr: np.ndarray) -> dict:
        """Single forward pass. No rotation sweep — it measured worse."""
        try:
            x = self.preprocess(crop_bgr)
            logits = self.session.run(None, {self.input_name: x})[0][0]
            e = np.exp(logits - logits.max())
            probs = e / e.sum()
            top = int(probs.argmax())
            return {
                "label": self.classes[top],
                "confidence": round(float(probs[top]), 4),
                "probs": {self.classes[i]: round(float(probs[i]), 4)
                          for i in range(len(self.classes))},
            }
        except Exception as exc:
            log.exception("classifier failed")
            return {"error": f"classification failed: {exc}"}


# ==========================================================================
# CLASSICAL LINE ANALYSIS — ported from detect.py
# ==========================================================================
def enhance_for_lines(crop_bgr: np.ndarray) -> np.ndarray:
    """CLAHE on V + saturation boost, to lift faint pink bands off the membrane.

    Used ONLY by the line detector and the debug overlay — never as the neural
    classifier's input, which must stay in the model's training distribution.
    """
    import cv2

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    v8 = np.clip(v, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=ENHANCE_CLAHE_CLIP,
                            tileGridSize=(ENHANCE_CLAHE_GRID, ENHANCE_CLAHE_GRID))
    v8 = clahe.apply(v8)
    s = np.clip(s * LINE_SAT_GAIN, 0, 255)
    hsv2 = cv2.merge([h, s, v8.astype(np.float32)]).astype(np.uint8)
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)


def find_red_lines(crop_bgr: np.ndarray):
    """Find every line-shaped red region in an upright crop.

    Red straddles the hue wrap-around at 0/180, so it needs two ranges.
    Shape gates then reject anything that is not a band: it must span >=10%
    of the width, be <=22% of the height, and be wider than tall.
    """
    import cv2

    h, w = crop_bgr.shape[:2]
    enh = enhance_for_lines(crop_bgr)
    hsv = cv2.cvtColor(enh, cv2.COLOR_BGR2HSV)

    red = (cv2.inRange(hsv,
                       np.array([0, LINE_SAT_FLOOR, LINE_VAL_FLOOR], np.uint8),
                       np.array([15, 255, 255], np.uint8))
           | cv2.inRange(hsv,
                         np.array([160, LINE_SAT_FLOOR, LINE_VAL_FLOOR], np.uint8),
                         np.array([179, 255, 255], np.uint8)))

    # Ignore the outer 12% — red there is usually a finger or the housing.
    side = int(w * LINE_SIDE_MARGIN_FRAC)
    band = np.zeros_like(red)
    band[:, side: w - side] = 255
    red = cv2.bitwise_and(red, band)

    # Open only. Closing would fatten thin lines and break the aspect gate.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel)

    n_labels, _lab, stats, _cent = cv2.connectedComponentsWithStats(red, connectivity=8)
    raw = []
    for lbl in range(1, n_labels):
        x, y, cw, ch, area = stats[lbl]
        if area < LINE_AREA_MIN:
            continue
        width_frac = cw / w
        if width_frac < LINE_WIDTH_FRAC_MIN:
            continue
        if ch / h > LINE_HEIGHT_FRAC_MAX:
            continue
        if cw / max(1, ch) < LINE_ASPECT_MIN:
            continue
        raw.append({
            "x": int(x), "y": int(y), "w": int(cw), "h": int(ch),
            "yc": float(y + ch / 2.0), "area": int(area),
            "width_frac": round(float(width_frac), 3),
            "aspect": round(float(cw / max(1, ch)), 2),
            "strength": round(float(min(1.0, width_frac / LINE_FULL_WIDTH_FRAC)), 3),
        })

    # Merge fragments at nearly the same height — one line broken by a gap.
    raw.sort(key=lambda d: d["yc"])
    merge_dist = LINE_MERGE_FRAC * h
    merged: list[dict] = []
    for ln in raw:
        if merged and abs(ln["yc"] - merged[-1]["yc"]) <= merge_dist:
            m = merged[-1]
            x1, y1 = min(m["x"], ln["x"]), min(m["y"], ln["y"])
            x2 = max(m["x"] + m["w"], ln["x"] + ln["w"])
            y2 = max(m["y"] + m["h"], ln["y"] + ln["h"])
            m["x"], m["y"], m["w"], m["h"] = x1, y1, x2 - x1, y2 - y1
            m["yc"] = (y1 + y2) / 2.0
            m["area"] += ln["area"]
            m["width_frac"] = round((x2 - x1) / w, 3)
            m["strength"] = round(float(min(1.0, (x2 - x1) / w / LINE_FULL_WIDTH_FRAC)), 3)
        else:
            merged.append(dict(ln))

    return merged, enh, red


def analyze_lines(crop_bgr: np.ndarray) -> dict | None:
    """Count bands and apply the clinical rule.

        0 lines -> invalid   (no control line: the test did not run)
        1 line  -> negative  (control only)
        2 lines -> positive  (control + test), if far enough apart

    We cannot tell C from T without brand calibration, so the decision rests
    on count plus a mandatory separation of 8% of crop height — two blobs
    closer than that are one broken band, not two results.
    """
    try:
        lines, _enh, _mask = find_red_lines(crop_bgr)
    except Exception:
        log.exception("line analysis failed")
        return None

    h, _w = crop_bgr.shape[:2]
    for ln in lines:
        ln["y_frac"] = round(float(ln["yc"]) / h, 3)
    lines.sort(key=lambda d: d["yc"])

    if len(lines) >= 2:
        by_strength = sorted(lines, key=lambda d: d["strength"], reverse=True)
        top2 = by_strength[:2]
        if abs(top2[0]["yc"] - top2[1]["yc"]) / h < LINE_MIN_SEPARATION_FRAC:
            lines = [max(top2, key=lambda d: d["strength"])]
        else:
            lines = sorted(top2, key=lambda d: d["yc"])

    n = len(lines)
    if n >= 2:
        # Confidence is the WEAKER band's strength: a barely-visible test line
        # is the evidence in question, so it caps how sure we can be.
        label, conf = "positive", min(ln["strength"] for ln in lines)
    elif n == 1:
        label, conf = "negative", lines[0]["strength"]
    else:
        label, conf = "invalid", 0.5

    log.debug("line-analysis: %d line(s) -> %s (%.3f)", n, label, conf)

    return {"label": label, "confidence": round(float(conf), 4),
            "num_lines": n, "lines": lines}


# ==========================================================================
# FUSION — ported from detect.py unchanged
# ==========================================================================
def _mk(label: str, conf: float, source: str, pos_score: float) -> dict:
    return {"label": label,
            "confidence": round(float(min(max(conf, 0.0), 1.0)), 4),
            "source": source,
            "pos_score": round(float(pos_score), 4)}


def fuse_decision(cnn: dict | None, line: dict | None) -> dict:
    """Combine the neural and classical verdicts, biased against false negatives.

    The asymmetry is clinical, not statistical: telling someone they are
    negative when they are positive is the costlier error, so positive
    evidence from EITHER source is enough.

    NOTE ON WHY THIS MATTERS LESS NOW
        This layer was written to rescue faint positives the CNN missed. Those
        misses were largely caused by the resize bug, which is fixed here — so
        the "weak-positive rescue" tier should fire far less often than it did
        in the desktop app. The layer still earns its place as an independent
        second opinion on a medical readout, but do not describe it as
        compensating for a model weakness that no longer exists.
    """
    cnn_ok = bool(cnn) and cnn.get("error") is None and cnn.get("label")
    cnn_label = cnn["label"] if cnn_ok else None
    cnn_probs = cnn.get("probs") if cnn_ok else None
    cnn_pos = float((cnn_probs or {}).get("positive", 0.0))
    cnn_conf = float(cnn.get("confidence", 0.0)) if cnn_ok else 0.0

    line_label = line["label"] if line else None
    line_conf = float(line["confidence"]) if line else 0.0
    line_pos_ev = line_conf if line_label == "positive" else 0.0
    n_lines = line["num_lines"] if line else 0

    pos_score = max(cnn_pos, line_pos_ev)

    # Tier 1 — either source is confident enough about a positive.
    if pos_score >= POSITIVE_DECISION_THRESHOLD:
        if cnn_label == "positive" and line_label == "positive":
            src = "cnn+lines agree (positive)"
        elif cnn_label == "positive":
            src = "cnn (positive)"
        elif line_label == "positive" and cnn_label is None:
            src = f"line-analysis ({n_lines} lines, no classifier)"
        elif line_label == "positive":
            src = f"weak-positive rescue: {n_lines} lines detected (CNN said {cnn_label})"
        else:
            src = f"line-analysis ({n_lines} lines)"
        return _mk("positive", pos_score, src, pos_score)

    # Tier 2 — no classifier available, trust the line counter.
    if not cnn_ok:
        return _mk(line_label or "invalid", max(line_conf, 0.3),
                   "line-analysis only (classifier unavailable)", pos_score)

    # Tier 3 — CNN says positive but below threshold; respect it anyway.
    if cnn_label == "positive":
        return _mk("positive", cnn_conf,
                   "cnn (positive, no corroborating 2nd line)", pos_score)

    # Tier 4 — invalid only when the CNN agrees.
    if cnn_label == "invalid":
        src = "cnn+lines agree (invalid)" if line_label == "invalid" else "cnn (invalid)"
        conf = max(cnn_conf, line_conf if line_label == "invalid" else 0.0)
        return _mk("invalid", conf, src, pos_score)

    # Tier 4.5 — no bands at all means the control line may be missing, which
    # is INVALID regardless of what the CNN thinks, unless it is very sure.
    if n_lines == 0 and line_label == "invalid" and cnn_label == "negative":
        if cnn_conf < CNN_OVERRIDE_INVALID_CONF:
            return _mk("invalid", 0.5,
                       f"0 lines detected — CNN says negative (conf {cnn_conf:.2f}) "
                       f"but below override threshold ({CNN_OVERRIDE_INVALID_CONF}); "
                       f"treating as invalid (possible absent control line)",
                       pos_score)

    # Tier 5 — negative.
    conf = max(cnn_conf, line_conf if line_label == "negative" else 0.0, 0.3)
    src = "cnn+lines agree (negative)" if line_label == "negative" else "cnn (negative)"
    return _mk("negative", conf, src, pos_score)


# ==========================================================================
# THE PIPELINE
# ==========================================================================
class Pipeline:
    """Load once at startup, call analyze() per request.

    Building the ONNX sessions re-optimizes both graphs, which takes far
    longer than an inference. On Cloud Run this happens during cold start,
    which is exactly where you want it — not on the request path.
    """

    def __init__(self, model_dir: Path, threads: int = 2):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.detector = ort.InferenceSession(
            str(model_dir / "detector.onnx"), so,
            providers=["CPUExecutionProvider"])
        self.classifier = Classifier(model_dir / "classifier.onnx",
                                     model_dir / "classifier_meta.json",
                                     threads=threads)
        log.info("pipeline ready (models from %s)", model_dir)

    def analyze(self, img_bgr: np.ndarray,
                conf_threshold: float = obb.CONF_THRESHOLD,
                iou_threshold: float = obb.IOU_THRESHOLD) -> dict:
        """Full read: detect -> crop -> classify + count lines -> fuse.

        Returns every detection with its own verdict, plus a top-level result
        taken from the highest-confidence detection. A photo with no cassette
        returns detections=[] and result=None — that is a valid answer, not an
        error, and the API layer turns it into a clear message rather than a
        stack trace.
        """
        import cv2

        detections = obb.detect(self.detector, img_bgr,
                                conf_threshold, iou_threshold)

        results = []
        for det in detections:
            poly = np.array(det["polygon"], dtype=np.float32)
            try:
                crop = to_portrait(rotated_crop(img_bgr, poly))
            except Exception as exc:
                log.exception("crop failed")
                results.append({"detection_confidence": det["confidence"],
                                "error": f"crop failed: {exc}"})
                continue

            cnn = self.classifier.classify(crop)
            line = analyze_lines(crop)
            fused = fuse_decision(cnn, line)

            results.append({
                "label": fused["label"],
                "confidence": fused["confidence"],
                "decision_source": fused["source"],
                "detection_confidence": round(det["confidence"], 4),
                "polygon": [[round(float(x), 2), round(float(y), 2)]
                            for x, y in det["polygon"]],
                "classifier": cnn,
                "line_analysis": None if line is None else {
                    "label": line["label"],
                    "confidence": line["confidence"],
                    "num_lines": line["num_lines"],
                    "lines": [{k: v for k, v in ln.items()
                               if k in ("y_frac", "width_frac", "aspect",
                                        "area", "strength")}
                              for ln in line["lines"]],
                },
                "crop_size": {"width": int(crop.shape[1]), "height": int(crop.shape[0])},
            })

        results.sort(key=lambda r: -r.get("detection_confidence", 0.0))

        return {
            "image_size": {"width": int(img_bgr.shape[1]),
                           "height": int(img_bgr.shape[0])},
            "num_detections": len(results),
            "result": results[0] if results else None,
            "detections": results,
        }
