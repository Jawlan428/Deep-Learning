"""
detect.py — Detection bridge for the JavaFX UI.

Operates on already-normalized images (no EXIF). Runs YOLOv8-OBB inference,
draws the OBB polygon onto the image, saves a clean annotated JPEG, and prints
JSON metadata.

Two-stage operation:
  * Stage 1 (default): localize the cassette, draw the OBB, return box metadata.
        python detect.py <model_path> <image_path> <output_image_path>

  * Stage 2 (when a crop output path is given): additionally perform a rotated
    crop of the highest-confidence cassette, save that upright close-up to
    <crop_output_path>, classify it positive/negative/invalid, and include a
    "classification" block (label + probabilities + decision trace) in the JSON.
        python detect.py <model_path> <image_path> <output_image_path> <crop_output_path>

Stage-2 accuracy notes (2026 update — faint-positive fix)
---------------------------------------------------------
The neural classifier (MobileNetV3 if present, else the YOLOv8-cls fallback
`classifier.pt`) tends to miss FAINT positive test lines and report "negative".
To make positives reliable WITHOUT retraining, Stage 2 now runs a second,
deterministic signal alongside the CNN: a classical red-line detector (CLAHE +
saturation boost + HSV threshold + connected-component shape gating, adapted
from synthesize_invalids.detect_red_line_mask). It COUNTS lines in the cropped
cassette and applies the clinical rule directly:
        0 lines -> invalid (no control line)
        1 line  -> negative (control only)
        2+ lines-> positive (control + test)
The two signals are then fused (`_fuse_decision`) with a deliberate bias against
missing a positive: if either the CNN or the line counter shows solid positive
evidence, the verdict is positive. All the tunables live in the constants block
below so you can calibrate sensitivity on your own images, and Stage 2 writes
debug images (enhanced crop, detected-line overlay, red mask, classifier input)
next to the crop so you can SEE why a verdict was reached.
"""
import json
import math
import sys
from pathlib import Path


# ===================== Stage-2 line-analysis + fusion tunables =====================
# These let you bias the final decision toward catching faint POSITIVE lines
# without retraining the model. Lower the thresholds to catch more faint
# positives (at the cost of more false positives); raise them if clear negatives
# start reading positive. Calibrate on a handful of your OWN positive AND
# negative images using the *_lines.jpg / *_red_mask.jpg debug outputs.
ENHANCE_CLAHE_CLIP = 2.0       # CLAHE clip limit applied to V before thresholding
ENHANCE_CLAHE_GRID = 8         # CLAHE tile grid size
LINE_SAT_GAIN = 1.6            # saturation multiplier — lifts faint pink lines over the floor
LINE_SAT_FLOOR = 35            # HSV saturation floor — raised from 30 to reduce shadow/housing hits
LINE_VAL_FLOOR = 40            # HSV value floor (ignore near-black shadows)
LINE_AREA_MIN = 12             # min component area (px) to be considered at all
LINE_WIDTH_FRAC_MIN = 0.10     # a line must span >=10% of the crop width
LINE_HEIGHT_FRAC_MAX = 0.22    # ...and be <=22% of crop height (lines are thin)
LINE_ASPECT_MIN = 1.2          # ...and wider than tall (horizontal line)
LINE_FULL_WIDTH_FRAC = 0.60    # width fraction that maps to "strength 1.0"
LINE_MERGE_FRAC = 0.04         # merge fragments whose centers are within 4% crop height (one broken line)
LINE_SIDE_MARGIN_FRAC = 0.12   # ignore red within the outer 12% (fingers / housing / background)
LINE_MIN_SEPARATION_FRAC = 0.08  # two distinct C/T lines must be >=8% crop height apart after merging
POSITIVE_DECISION_THRESHOLD = 0.40  # combined positive evidence (max of CNN/line) needed to call positive
# CNN must have at least this confidence to override "0 lines detected → invalid" with "negative"
CNN_OVERRIDE_INVALID_CONF = 0.65
# ==================================================================================


def fail(msg: str) -> None:
    print(json.dumps({"ok": False, "error": msg}))
    sys.exit(0)

def _order_polygon(poly):
    import numpy as np
    poly = np.asarray(poly, dtype=np.float32).reshape(4, 2)
    edges = []
    for i in range(4):
        p1 = poly[i]
        p2 = poly[(i + 1) % 4]
        edges.append((i, p2 - p1, float(np.linalg.norm(p2 - p1))))
    edges.sort(key=lambda e: e[2], reverse=True)
    long_start_idx = edges[0][0]
    ordered = np.array([
        poly[long_start_idx],
        poly[(long_start_idx + 1) % 4],
        poly[(long_start_idx + 2) % 4],
        poly[(long_start_idx + 3) % 4],
    ], dtype=np.float32)
    top_mean_y = (ordered[0][1] + ordered[1][1]) / 2
    bot_mean_y = (ordered[2][1] + ordered[3][1]) / 2
    if top_mean_y > bot_mean_y:
        ordered = np.array([ordered[2], ordered[3], ordered[0], ordered[1]],
                           dtype=np.float32)
    return ordered


def rotated_crop(image_bgr, polygon, pad_frac: float = 0.04):
    import cv2
    import numpy as np
    src_pts = _order_polygon(polygon)
    width = float(np.linalg.norm(src_pts[1] - src_pts[0]))
    height = float(np.linalg.norm(src_pts[2] - src_pts[1]))
    long_axis = max(width, height)
    short_axis = min(width, height)
    pad = int(round(pad_frac * long_axis))
    out_w = int(round(short_axis)) + 2 * pad
    out_h = int(round(long_axis)) + 2 * pad
    if width >= height:
        dst_pts = np.array([
            [out_w - 1 - pad, pad],
            [out_w - 1 - pad, out_h - 1 - pad],
            [pad,             out_h - 1 - pad],
            [pad,             pad],
        ], dtype=np.float32)
    else:
        dst_pts = np.array([
            [pad,             pad],
            [out_w - 1 - pad, pad],
            [out_w - 1 - pad, out_h - 1 - pad],
            [pad,             out_h - 1 - pad],
        ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(
        image_bgr, M, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


# Cache the torch classifier across calls within one process run.
_TORCH_CLF = None
_TORCH_META = None

# Cache the YOLOv8-cls fallback classifier too, so classifying multiple
# detections in one "Read Result" doesn't reload the model for each one.
_YOLO_CLF = None
_YOLO_CLF_PATH = None


def _load_torch_classifier(model_file):
    """Load a MobileNetV3 classifier saved by train_classifier_mobilenet.py.

    The checkpoint is a dict: {state_dict, classes, arch, img_size, mean, std}.
    Returns (model, meta) or raises.
    """
    global _TORCH_CLF, _TORCH_META
    if _TORCH_CLF is not None:
        return _TORCH_CLF, _TORCH_META
    import torch
    import torch.nn as nn
    from torchvision import models

    ckpt = torch.load(str(model_file), map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    arch = ckpt.get("arch", "mobilenet_v3_small")
    if arch != "mobilenet_v3_small":
        raise ValueError(f"Unsupported classifier arch: {arch}")

    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    meta = {
        "classes": classes,
        "img_size": int(ckpt.get("img_size", 224)),
        "mean": ckpt.get("mean", [0.485, 0.456, 0.406]),
        "std": ckpt.get("std", [0.229, 0.224, 0.225]),
    }
    _TORCH_CLF, _TORCH_META = model, meta
    return model, meta


def _to_portrait(crop_bgr):
    """Rotate the crop so it is taller than wide (cassette standing upright).
    This only fixes the *displayed* orientation; classification is rotation-
    invariant via the 4-way pass below, so this is purely cosmetic."""
    import cv2 as _cv2
    h, w = crop_bgr.shape[:2]
    if w > h:
        return _cv2.rotate(crop_bgr, _cv2.ROTATE_90_CLOCKWISE)
    return crop_bgr


def _classify_crop_torch(crop_bgr, model, meta):
    """Classify a BGR crop with the MobileNet model.

    Orientation-invariant: the crop is evaluated at all four 90-degree
    rotations and the rotation with the highest top-class probability wins.
    This removes the dependence on the crop coming out perfectly upright (the
    near-square OBB ambiguity that otherwise flips the cassette sideways).

    Returns (label, conf, prob_map).
    """
    import torch
    import cv2 as _cv2
    import numpy as _np

    size = meta["img_size"]
    mean = _np.array(meta["mean"], dtype=_np.float32)
    std = _np.array(meta["std"], dtype=_np.float32)
    classes = meta["classes"]

    rotations = [
        crop_bgr,
        _cv2.rotate(crop_bgr, _cv2.ROTATE_90_CLOCKWISE),
        _cv2.rotate(crop_bgr, _cv2.ROTATE_180),
        _cv2.rotate(crop_bgr, _cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]

    best = None  # (top_conf, probs_array)
    for rot in rotations:
        rgb = _cv2.cvtColor(rot, _cv2.COLOR_BGR2RGB)
        rgb = _cv2.resize(rgb, (size, size), interpolation=_cv2.INTER_LINEAR)
        arr = rgb.astype(_np.float32) / 255.0
        arr = (arr - mean) / std
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
        top_conf = float(probs.max())
        if best is None or top_conf > best[0]:
            best = (top_conf, probs)

    probs = best[1]
    top = int(probs.argmax())
    prob_map = {classes[i]: round(float(probs[i]), 4) for i in range(len(classes))}
    return classes[top], round(float(probs[top]), 4), prob_map


# ============================================================================
# Stage-2 accuracy layer: classical line counting + CNN fusion.
# Everything below is the faint-positive fix. None of it can throw out of
# Stage 2 — failures degrade gracefully to "CNN only" so the app never breaks.
# ============================================================================

def _enhance_for_lines(crop_bgr):
    """Lift faint pink lines out of the membrane so the HSV threshold can see
    them. Applies CLAHE to the V channel (evens out lighting, boosts local
    contrast) and multiplies saturation so a faded test line clears the floor.
    Returns an enhanced BGR image — used ONLY for the classical line detector
    and for the debug overlay, never as the neural classifier's input (that
    must match the model's training distribution)."""
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    v8 = np.clip(v, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=ENHANCE_CLAHE_CLIP,
                            tileGridSize=(ENHANCE_CLAHE_GRID, ENHANCE_CLAHE_GRID))
    v8 = clahe.apply(v8)
    s = np.clip(s * LINE_SAT_GAIN, 0, 255)
    hsv2 = cv2.merge([h, s, v8.astype(np.float32)]).astype(np.uint8)
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)


def _find_red_lines(crop_bgr):
    """Find ALL line-shaped red regions (candidate C/T lines) in an upright
    cassette crop. Adapted from synthesize_invalids.detect_red_line_mask, but
    returns every line that passes the shape gates (not just the best one) so
    we can COUNT them. Returns (lines, enhanced_bgr, red_mask).

    Each line is a dict with bbox + 'strength' (0..1 from how much of the crop
    width it spans). Components at nearly the same vertical position are merged
    so a single line broken by a small gap is not counted twice."""
    import cv2
    import numpy as np

    h, w = crop_bgr.shape[:2]
    enh = _enhance_for_lines(crop_bgr)
    hsv = cv2.cvtColor(enh, cv2.COLOR_BGR2HSV)

    # Red wraps the hue circle, so two ranges (low ~0deg and high ~180deg).
    lower1 = np.array([0,   LINE_SAT_FLOOR, LINE_VAL_FLOOR], dtype=np.uint8)
    upper1 = np.array([15,  255, 255], dtype=np.uint8)
    lower2 = np.array([160, LINE_SAT_FLOOR, LINE_VAL_FLOOR], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)
    red = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    # Restrict to the central column band — red near the edges is more likely a
    # finger / the plastic housing / background than a result line.
    side = int(w * LINE_SIDE_MARGIN_FRAC)
    band = np.zeros_like(red)
    band[:, side: w - side] = 255
    red = cv2.bitwise_and(red, band)

    # Small open to drop specks (no close — closing fattens thin lines and
    # breaks the aspect-ratio gate below).
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel)

    n_labels, _labels, stats, _cent = cv2.connectedComponentsWithStats(red, connectivity=8)
    raw = []
    for lbl in range(1, n_labels):
        x, y, cw, ch, area = stats[lbl]
        if area < LINE_AREA_MIN:
            continue
        width_frac = cw / w
        height_frac = ch / h
        aspect = cw / max(1, ch)
        if width_frac < LINE_WIDTH_FRAC_MIN:
            continue
        if height_frac > LINE_HEIGHT_FRAC_MAX:
            continue
        if aspect < LINE_ASPECT_MIN:
            continue
        raw.append({
            "x": int(x), "y": int(y), "w": int(cw), "h": int(ch),
            "yc": float(y + ch / 2.0), "area": int(area),
            "width_frac": round(float(width_frac), 3),
            "aspect": round(float(aspect), 2),
            "strength": round(float(min(1.0, width_frac / LINE_FULL_WIDTH_FRAC)), 3),
        })

    # Merge components at nearly the same vertical position (one broken line).
    raw.sort(key=lambda d: d["yc"])
    merge_dist = LINE_MERGE_FRAC * h
    merged = []
    for ln in raw:
        if merged and abs(ln["yc"] - merged[-1]["yc"]) <= merge_dist:
            m = merged[-1]
            x1 = min(m["x"], ln["x"]); y1 = min(m["y"], ln["y"])
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


def _analyze_lines(crop_bgr):
    """Classify the crop by counting and positioning result lines.

    Medical rules being implemented:
      POSITIVE = control line (C) visible AND test line (T) visible (even faint)
      NEGATIVE = control line visible, no test line
      INVALID  = control line absent (no matter what else appears)

    We cannot identify which detected band is C vs T without brand-specific
    calibration, so we use physical count + mandatory separation:
      0 lines                       → invalid (no control line)
      1 line                        → negative (assume it is the control)
      2 well-separated lines (≥8%)  → positive (both C and T present)
      2 lines closer than 8%        → treated as 1 broken line → negative

    Every detected line is annotated with its vertical position (y_frac) so
    the caller and the debug overlay can show exactly where in the crop each
    band was found.
    """
    try:
        lines, enh, mask = _find_red_lines(crop_bgr)
    except Exception:
        return None

    h, w = crop_bgr.shape[:2]

    # Annotate each candidate with its relative vertical position in the crop.
    for ln in lines:
        ln["y_frac"] = round(float(ln["yc"]) / h, 3)

    # Sort by vertical position top → bottom (ascending y).
    lines.sort(key=lambda d: d["yc"])

    # ---- Separation check -----------------------------------------------
    # After the merge step in _find_red_lines we may still have two fragments
    # that belong to the same line band. Two entries count as *distinct* C/T
    # lines only if their vertical centres are at least LINE_MIN_SEPARATION_FRAC
    # of the crop height apart.  Closer pairs are collapsed to the stronger one.
    if len(lines) >= 2:
        by_strength = sorted(lines, key=lambda d: d["strength"], reverse=True)
        top2 = by_strength[:2]
        sep_px = abs(top2[0]["yc"] - top2[1]["yc"])
        sep_frac = sep_px / h
        if sep_frac < LINE_MIN_SEPARATION_FRAC:
            # Too close → one broken line, not two distinct bands.
            survivor = max(top2, key=lambda d: d["strength"])
            lines = [survivor]
        else:
            # Keep only the two strongest, back in positional (top→bot) order.
            lines = sorted(top2, key=lambda d: d["yc"])

    n = len(lines)

    # Print per-line debug summary to stderr so it shows in the console without
    # polluting stdout (which must remain the JSON line).
    import sys as _sys
    _sys.stderr.write(f"[line-analysis] crop {w}x{h}px — {n} line(s) after separation check\n")
    for i, ln in enumerate(lines):
        _sys.stderr.write(
            f"  line {i+1}: y_frac={ln['y_frac']:.3f}  "
            f"width_frac={ln['width_frac']:.3f}  "
            f"aspect={ln['aspect']:.1f}  "
            f"area={ln['area']}  "
            f"strength={ln['strength']:.3f}\n"
        )
    if len(lines) == 2:
        sep = abs(lines[0]["yc"] - lines[1]["yc"]) / h
        _sys.stderr.write(f"  separation between lines: {sep:.3f} of crop height\n")

    if n >= 2:
        # Both bands present — positive.
        # Confidence = strength of the WEAKER band (the faint test line is the
        # evidence; a very faint T line means less certainty even if C is clear).
        conf = min(ln["strength"] for ln in lines)
        label = "positive"
    elif n == 1:
        label = "negative"
        conf = lines[0]["strength"]
    else:
        label = "invalid"
        conf = 0.5

    _sys.stderr.write(f"  → line-analysis verdict: {label}  conf={conf:.3f}\n")

    return {
        "label": label,
        "confidence": round(float(conf), 4),
        "num_lines": n,
        "lines": lines,
        "_enhanced": enh,
        "_mask": mask,
    }


def _run_classifier(crop_bgr, mnv3_path, yolo_path):
    """Run the neural classifier on the crop. Prefers the MobileNetV3 checkpoint
    (classifier_mnv3.pt); falls back to the YOLOv8-cls model (classifier.pt).
    Returns (result_dict_or_None, backend_str). backend_str is one of
    'mobilenet_v3' / 'yolov8-cls' / 'none' for visibility into which model ran.
    result_dict is {label, confidence, probs} on success or {error} on failure."""
    if mnv3_path.exists():
        try:
            mdl, meta = _load_torch_classifier(mnv3_path)
            label, conf, prob_map = _classify_crop_torch(crop_bgr, mdl, meta)
            return {"label": label, "confidence": conf, "probs": prob_map}, "mobilenet_v3"
        except Exception as e:
            return {"error": f"classification failed (mnv3): {e}"}, "mobilenet_v3"
    if yolo_path.exists():
        try:
            from ultralytics import YOLO
            global _YOLO_CLF, _YOLO_CLF_PATH
            if _YOLO_CLF is None or _YOLO_CLF_PATH != str(yolo_path):
                _YOLO_CLF = YOLO(str(yolo_path))
                _YOLO_CLF_PATH = str(yolo_path)
            cres = _YOLO_CLF.predict(source=crop_bgr, imgsz=224, verbose=False)[0]
            names_c = cres.names
            probs = cres.probs
            top_i = int(probs.top1)
            prob_map = {names_c[i]: round(float(probs.data[i]), 4)
                        for i in range(len(names_c))}
            return {"label": names_c[top_i],
                    "confidence": round(float(probs.top1conf), 4),
                    "probs": prob_map}, "yolov8-cls"
        except Exception as e:
            return {"error": f"classification failed (yolo): {e}"}, "yolov8-cls"
    return None, "none"


def _mk(label, conf, source, pos_score):
    """Build a fused-decision record with clamped confidence."""
    return {
        "label": label,
        "confidence": round(float(min(max(conf, 0.0), 1.0)), 4),
        "source": source,
        "pos_score": round(float(pos_score), 4),
    }


def _fuse_decision(cnn, line):
    """Combine the neural classifier with the classical line counter.

    Policy is intentionally asymmetric — the cost of telling a positive user
    they are negative is high, so we bias toward catching positives:

      * If EITHER source shows positive evidence >= POSITIVE_DECISION_THRESHOLD
        -> positive. This is what rescues faint positives the CNN missed.
      * Otherwise fall back to the CNN. The line counter is only allowed to push
        the verdict to 'invalid' when the CNN ALSO says invalid (a missed faint
        control line must not masquerade as 'invalid').
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

    # Combined positive evidence = the stronger of the two sources.
    pos_score = max(cnn_pos, line_pos_ev)

    # --- Tier 1: positive wins if either source is confident enough ---
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

    # --- Tier 2: classifier unavailable -> trust the line counter ---
    if not cnn_ok:
        return _mk(line_label or "invalid", max(line_conf, 0.3),
                   "line-analysis only (classifier unavailable)", pos_score)

    # --- Tier 3: CNN's own top pick is positive (but sub-threshold) -> respect it ---
    if cnn_label == "positive":
        return _mk("positive", cnn_conf,
                   "cnn (positive, no corroborating 2nd line)", pos_score)

    # --- Tier 4: invalid only when the CNN says so (lines may corroborate) ---
    if cnn_label == "invalid":
        src = "cnn+lines agree (invalid)" if line_label == "invalid" else "cnn (invalid)"
        conf = max(cnn_conf, line_conf if line_label == "invalid" else 0.0)
        return _mk("invalid", conf, src, pos_score)

    # --- Tier 4.5: 0 lines detected but CNN says negative ---
    # Medically: if the classical detector finds NO coloured bands at all, the
    # control line may be absent, which is INVALID regardless of what the CNN
    # says.  We only trust the CNN's "negative" here if its confidence is high
    # enough to override the physical absence-of-lines signal.
    if n_lines == 0 and line_label == "invalid" and cnn_label == "negative":
        if cnn_conf < CNN_OVERRIDE_INVALID_CONF:
            return _mk(
                "invalid", 0.5,
                f"0 lines detected — CNN says negative (conf {cnn_conf:.2f}) "
                f"but below override threshold ({CNN_OVERRIDE_INVALID_CONF}); "
                f"treating as invalid (possible absent control line)",
                pos_score,
            )
        # CNN is very confident negative despite 0 lines → trust it, but the
        # UI amber warning already surfaces this discrepancy to the reviewer.

    # --- Tier 5: negative ---
    conf = max(cnn_conf, line_conf if line_label == "negative" else 0.0, 0.3)
    src = "cnn+lines agree (negative)" if line_label == "negative" else "cnn (negative)"
    return _mk("negative", conf, src, pos_score)


def _line_to_json(line):
    """Strip the private numpy fields so the line block is JSON-serializable."""
    if not line:
        return None
    return {
        "label": line["label"],
        "confidence": line["confidence"],
        "num_lines": line["num_lines"],
        "lines": [{k: ln[k] for k in ("x", "y", "w", "h", "y_frac", "width_frac", "aspect", "strength")
                   if k in ln}
                  for ln in line.get("lines", [])],
    }


def _save_debug_images(crop_path, crop_bgr, line):
    """Write debug artifacts next to the crop so you can SEE the pipeline:
    *_enhanced.jpg (contrast/saturation-boosted), *_lines.jpg (detected lines
    boxed + verdict), *_red_mask.jpg (what the threshold matched), and
    *_clf_input.jpg (what the classifier sees). Returns a dict of paths."""
    import cv2
    out = {}

    def name(suffix):
        return str(crop_path.with_name(crop_path.stem + "_" + suffix + ".jpg")).replace("\\", "/")

    # Classifier input (the crop resized the way the model receives it).
    try:
        p = name("clf_input")
        if cv2.imwrite(p, cv2.resize(crop_bgr, (224, 224)),
                       [cv2.IMWRITE_JPEG_QUALITY, 90]):
            out["classifier_input"] = p
    except Exception:
        pass

    if line is not None:
        # Enhanced crop used for line detection.
        try:
            enh = line.get("_enhanced")
            if enh is not None:
                p = name("enhanced")
                if cv2.imwrite(p, enh, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                    out["enhanced"] = p
        except Exception:
            pass

        # Detected lines drawn on the crop + verdict caption + per-line y_frac.
        try:
            viz = crop_bgr.copy()
            h, w = viz.shape[:2]
            side = int(w * LINE_SIDE_MARGIN_FRAC)
            cv2.rectangle(viz, (side, 0), (w - side, h - 1), (0, 180, 0), 1)
            for i, ln in enumerate(line.get("lines", [])):
                cv2.rectangle(viz, (ln["x"], ln["y"]),
                              (ln["x"] + ln["w"], ln["y"] + ln["h"]), (0, 0, 255), 2)
                y_frac = ln.get("y_frac", ln["y"] / h)
                label_str = (f"L{i+1} y={y_frac:.2f} "
                             f"str={ln['strength']:.2f}")
                cv2.putText(viz, label_str,
                            (ln["x"], max(14, ln["y"] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 80, 0), 1,
                            cv2.LINE_AA)
            cap = f"{line['num_lines']} line(s) -> {line['label']} (conf {line['confidence']:.2f})"
            cv2.putText(viz, cap, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        (0, 0, 200), 2, cv2.LINE_AA)
            if len(line.get("lines", [])) == 2:
                lns = line["lines"]
                sep = abs(lns[0].get("y_frac", lns[0]["y"]/h) -
                          lns[1].get("y_frac", lns[1]["y"]/h))
                cv2.putText(viz, f"sep={sep:.2f}",
                            (4, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (0, 140, 0), 1, cv2.LINE_AA)
            p = name("lines")
            if cv2.imwrite(p, viz, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                out["line_analysis"] = p
        except Exception:
            pass

        # Red mask overlaid on a grayscale copy (see exactly what matched).
        try:
            mask = line.get("_mask")
            if mask is not None:
                bg = cv2.cvtColor(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY),
                                  cv2.COLOR_GRAY2BGR)
                bg[mask > 0] = (0, 0, 255)
                p = name("red_mask")
                if cv2.imwrite(p, bg, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                    out["red_mask"] = p
        except Exception:
            pass

    return out


def _build_classification(fused, cnn, line_json, backend, debug_images):
    """Assemble the final 'classification' JSON block. Keeps the original
    fields (label / confidence / probs) that the Java UI already reads, and adds
    a transparent decision trace (decision_source, cnn_*, line_*, backend)."""
    cnn_ok = bool(cnn) and cnn.get("error") is None
    probs = cnn.get("probs") if cnn_ok else None
    if not probs:
        # No CNN probabilities (failed / absent) — synthesize a minimal map from
        # the fused decision so the UI still has something to render.
        probs = {"positive": 0.0, "negative": 0.0, "invalid": 0.0}
        if fused["label"] in probs:
            probs[fused["label"]] = fused["confidence"]

    return {
        "label": fused["label"],
        "confidence": fused["confidence"],
        "probs": probs,
        "decision_source": fused["source"],
        "classifier_backend": backend,
        "pos_score": fused["pos_score"],
        "num_lines": (line_json or {}).get("num_lines"),
        "line_label": (line_json or {}).get("label"),
        "line_confidence": (line_json or {}).get("confidence"),
        "lines": (line_json or {}).get("lines"),
        "cnn_label": cnn.get("label") if cnn_ok else None,
        "cnn_confidence": cnn.get("confidence") if cnn_ok else None,
        "cnn_error": cnn.get("error") if (cnn and not cnn_ok) else None,
        "debug_images": debug_images,
    }


def _classify_detection(img, poly, crop_path, mnv3_path, yolo_path):
    """Full Stage-2 pipeline for ONE detection: rotated crop -> line analysis ->
    neural classifier -> fuse -> debug images. Returns (classification, crop_out).
    Fully guarded so one bad detection can never abort the others. The accuracy
    logic itself is unchanged — this just applies it per detection."""
    import cv2
    try:
        crop = rotated_crop(img, poly, pad_frac=0.04)
        crop = _to_portrait(crop)
        crop_out = None
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            crop_out = str(crop_path).replace("\\", "/")
    except Exception as e:
        return {"error": f"crop failed: {e}"}, None
    try:
        line = _analyze_lines(crop)
        cnn, backend = _run_classifier(crop, mnv3_path, yolo_path)
        fused = _fuse_decision(cnn, line)
        debug_images = _save_debug_images(crop_path, crop, line) if crop_out else {}
        line_json = _line_to_json(line)
        classification = _build_classification(fused, cnn, line_json, backend, debug_images)
    except Exception as e:
        classification = {"error": f"classification failed: {e}"}
    return classification, crop_out


def main() -> None:
    if len(sys.argv) not in (4, 5):
        fail("Usage: detect.py <model_path> <image_path> <output_image_path> [crop_output_path]")

    model_path = Path(sys.argv[1])
    image_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    crop_output_path = Path(sys.argv[4]) if len(sys.argv) == 5 else None
    do_classify = crop_output_path is not None

    if not model_path.exists():
        fail(f"Model not found: {model_path}")
    if not image_path.exists():
        fail(f"Image not found: {image_path}")

    try:
        from ultralytics import YOLO
        import cv2
        import numpy as np
    except ImportError as e:
        fail(f"Missing dependency: {e}. Run: pip install ultralytics opencv-python")

    try:
        img = cv2.imread(str(image_path))
        if img is None:
            fail(f"Failed to read image: {image_path}")
        img_h, img_w = img.shape[:2]
    except Exception as e:
        fail(f"Failed to read image: {e}")

    try:
        model = YOLO(str(model_path))
        results = model.predict(
            source=img,
            imgsz=640,
            conf=0.4,
            iou=0.5,
            verbose=False,
        )
    except Exception as e:
        fail(f"Inference failed: {e}")

    detections_meta = []
    detection_polys = []
    annotated = img.copy()

    line_thick = max(1, int(round(min(img_w, img_h) / 800)))
    text_scale = max(0.3, min(img_w, img_h) / 4000)
    text_thick = max(1, int(round(text_scale * 2)))
    GREEN = (86, 110, 15)        # BGR for #0F6E56 medical green
    WHITE = (255, 255, 255)

    for r in results:
        obb = getattr(r, "obb", None)
        if obb is None or obb.xyxyxyxy is None:
            continue
        polys = obb.xyxyxyxy.cpu().numpy()
        confs = obb.conf.cpu().numpy()
        cls_ids = obb.cls.cpu().numpy().astype(int)
        names = r.names

        for poly, conf, cid in zip(polys, confs, cls_ids):
            pts_int = poly.astype(np.int32)

            cv2.polylines(annotated, [pts_int], isClosed=True,
                          color=WHITE, thickness=line_thick + 3, lineType=cv2.LINE_AA)
            cv2.polylines(annotated, [pts_int], isClosed=True,
                          color=GREEN, thickness=line_thick, lineType=cv2.LINE_AA)

            label = f"{names.get(int(cid), str(int(cid)))}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thick
            )
            top_idx = int(np.argmin(pts_int[:, 1]))
            anchor = pts_int[top_idx]
            pad_x, pad_y = 4, 3
            box_x1 = int(anchor[0])
            box_y1 = int(max(0, anchor[1] - th - pad_y * 2 - 4))
            box_x2 = box_x1 + tw + pad_x * 2
            box_y2 = box_y1 + th + pad_y * 2

            cv2.rectangle(annotated, (box_x1, box_y1), (box_x2, box_y2),
                          GREEN, thickness=cv2.FILLED)
            cv2.putText(
                annotated, label,
                (box_x1 + pad_x, box_y2 - pad_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, text_scale, WHITE, text_thick,
                lineType=cv2.LINE_AA,
            )

            xs = poly[:, 0]
            ys = poly[:, 1]
            box_w = float(xs.max() - xs.min())
            box_h = float(ys.max() - ys.min())

            edges = [(poly[(i + 1) % 4][0] - poly[i][0],
                      poly[(i + 1) % 4][1] - poly[i][1]) for i in range(4)]
            longest = max(edges, key=lambda e: e[0] * e[0] + e[1] * e[1])
            rot_deg = math.degrees(math.atan2(longest[1], longest[0]))
            while rot_deg > 90:
                rot_deg -= 180
            while rot_deg < -90:
                rot_deg += 180

            detections_meta.append({
                "class_id": int(cid),
                "class_name": names.get(int(cid), str(int(cid))),
                "confidence": float(conf),
                "rotation_deg": round(float(rot_deg), 2),
                "bbox_width": round(box_w, 1),
                "bbox_height": round(box_h, 1),
                "_poly": poly.copy(),   # stashed for Stage-2 cropping; stripped before JSON
            })
            detection_polys.append((float(conf), poly.copy()))

    detections_meta.sort(key=lambda d: d["confidence"], reverse=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        fail(f"Failed to write annotated image to {output_path}")

    # ---- Stage 2: crop + classify EACH detected cassette ----
    # Runs only when a crop output path was supplied (the "Read Result" step).
    # Every detection goes through the IDENTICAL pipeline (line analysis +
    # classifier fusion). The top-level classification/crop_image mirror the
    # highest-confidence detection so older callers keep working. Each detection
    # also carries its own classification + crop_image for per-cassette review.
    classification = None
    crop_image_out = None
    if do_classify and detections_meta:
        # Prefer the MobileNetV3 classifier; fall back to the old YOLOv8-cls.
        base = model_path.parent
        mnv3_path = base / "python" / "classifier_mnv3.pt"
        if not mnv3_path.exists():
            mnv3_path = base / "classifier_mnv3.pt"
        yolo_path = base / "python" / "classifier.pt"
        if not yolo_path.exists():
            yolo_path = base / "classifier.pt"

        # detections_meta is already sorted by confidence (desc); each dict still
        # carries its own "_poly", so per-detection crops stay correctly aligned.
        for k, d in enumerate(detections_meta):
            poly_k = d.get("_poly")
            if poly_k is None:
                d["classification"], d["crop_image"] = None, None
                continue
            crop_path_k = crop_output_path.with_name(
                f"{crop_output_path.stem}_det{k + 1}{crop_output_path.suffix}")
            clf_k, crop_k = _classify_detection(img, poly_k, crop_path_k,
                                                mnv3_path, yolo_path)
            d["classification"], d["crop_image"] = clf_k, crop_k

        # Top-level mirrors the best (first) detection — backward compatible.
        classification = detections_meta[0].get("classification")
        crop_image_out = detections_meta[0].get("crop_image")

    # Strip the private polygon we stashed for cropping before serializing.
    for d in detections_meta:
        d.pop("_poly", None)

    print(json.dumps({
        "ok": True,
        "annotated_image": str(output_path).replace("\\", "/"),
        "crop_image": crop_image_out,
        "image_width": img_w,
        "image_height": img_h,
        "detections": detections_meta,
        "classification": classification,
    }))


if __name__ == "__main__":
    main()
