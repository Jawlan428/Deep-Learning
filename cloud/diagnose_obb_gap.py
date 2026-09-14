"""
diagnose_obb_gap.py — why does exactly one image still differ?

CONTEXT
    After fixing the letterbox padding bug, 19 of 20 images match Ultralytics
    at IoU 1.0000. invalid_05 sits at 0.9674 and barely moved, so it has a
    different cause.

    What we know: the confidence difference is EXACTLY 0.00000, and both
    pipelines return exactly 1 detection. Same anchor, same score. So the
    network output is identical and the divergence is purely in the
    coordinate transform.

WHAT THIS DUMPS
    Every intermediate value in the transform, side by side, for one image:
    shapes, scale factor, unpadded size, the padding we applied, the padding
    Ultralytics *un*-applies, the raw xywhr, and both final polygons.

THE LEADING SUSPECT
    Ultralytics pads and un-pads with two DIFFERENT formulas.

      padding (LetterBox):   new_unpad = int(round(w0 * r))
                             dw        = (640 - new_unpad) / 2
      un-padding (scale_boxes):
                             pad_x     = round((640 - w0 * r) / 2 - 0.1)

    The first rounds the resized width to a whole pixel BEFORE halving; the
    second halves the unrounded float. When w0*r lands near a .5 boundary
    these disagree by one pixel. Our code uses the true applied padding,
    which is geometrically right but does not match the reference.

    If that is the cause, the fix is to reproduce Ultralytics' un-pad formula
    — deliberately inheriting its rounding quirk, because the goal here is
    agreement with the desktop app, not abstract correctness.

RUN
    ..\\.venv\\Scripts\\python.exe cloud\\diagnose_obb_gap.py ^
        --image "C:\\Users\\user\\Downloads\\lft-detector-ui-main\\lft-detector-ui-main\\demo_examples\\invalid\\invalid_05.png"

    (adjust the subfolder if invalid_05.png sits elsewhere)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import obb_postprocess as pp

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUT_DIR = HERE / "models"
DETECTOR_ONNX = OUT_DIR / "detector.onnx"


def log(m=""):
    print(m, flush=True)


def section(t):
    log("\n" + "=" * 72)
    log(t)
    log("=" * 72)


def imread_unicode(path: Path):
    import cv2
    data = np.fromfile(str(path), dtype=np.uint8)
    return None if data.size == 0 else cv2.imdecode(data, cv2.IMREAD_COLOR)


def poly_iou(p1, p2) -> float:
    import cv2
    a = np.asarray(p1, dtype=np.float32).reshape(-1, 2)
    b = np.asarray(p2, dtype=np.float32).reshape(-1, 2)
    inter, _ = cv2.intersectConvexConvex(a, b)
    aa, ab = abs(cv2.contourArea(a)), abs(cv2.contourArea(b))
    u = aa + ab - inter
    return float(inter / u) if u > 0 else 0.0


def poly_metrics(p) -> dict:
    """Centre, side lengths and area — easier to compare than 8 raw numbers."""
    p = np.asarray(p, dtype=np.float64).reshape(4, 2)
    side1 = np.linalg.norm(p[1] - p[0])
    side2 = np.linalg.norm(p[2] - p[1])
    import cv2
    area = abs(cv2.contourArea(p.astype(np.float32)))
    return {"cx": p[:, 0].mean(), "cy": p[:, 1].mean(),
            "w": max(side1, side2), "h": min(side1, side2), "area": area}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    path = Path(args.image)
    if not path.exists():
        raise SystemExit(f"Not found: {path}\nCheck the subfolder name.")

    img = imread_unicode(path)
    if img is None:
        raise SystemExit(f"Could not decode {path}")

    h0, w0 = img.shape[:2]
    size = pp.IMGSZ

    section("Letterbox arithmetic")
    r = min(size / h0, size / w0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    left = int(round(dw - 0.1))
    top = int(round(dh - 0.1))

    # Ultralytics' scale_boxes un-pad formula (note: uses the UNROUNDED float)
    ul_pad_x = round((size - w0 * r) / 2 - 0.1)
    ul_pad_y = round((size - h0 * r) / 2 - 0.1)

    log(f"  original (w x h)      : {w0} x {h0}")
    log(f"  scale r               : {r:.10f}")
    log(f"  w0*r, h0*r (float)    : {w0 * r:.6f}, {h0 * r:.6f}")
    log(f"  new_unpad (rounded)   : {new_w} x {new_h}")
    log(f"  dw, dh                : {dw}, {dh}")
    log("")
    log(f"  padding WE applied    : left={left}  top={top}")
    log(f"  padding WE un-apply   : left={left}  top={top}   (the fix)")
    log(f"  padding ULTRALYTICS   : pad_x={ul_pad_x}  pad_y={ul_pad_y}   (scale_boxes)")
    log("")
    dx, dy = ul_pad_x - left, ul_pad_y - top
    if dx or dy:
        log(f"  >> MISMATCH of ({dx}, {dy}) px in 640-space")
        log(f"     = ({dx / r:.2f}, {dy / r:.2f}) px in the original image")
        log("     Ultralytics halves the UNROUNDED float; the actual padding")
        log("     rounds the resized size first. They disagree near .5.")
    else:
        log("  >> pad formulas agree on this image — look elsewhere.")

    section("Raw detection, before any scaling")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(DETECTOR_ONNX), providers=["CPUExecutionProvider"])
    blob, ratio, pad = pp.preprocess(img)
    raw = sess.run(None, {sess.get_inputs()[0].name: blob})[0]

    pred = raw[0].T
    scores = pred[:, 4]
    best = int(scores.argmax())
    log(f"  best anchor index : {best}")
    log(f"  confidence        : {scores[best]:.8f}")
    log(f"  cx, cy, w, h      : {pred[best, 0]:.4f}, {pred[best, 1]:.4f}, "
        f"{pred[best, 2]:.4f}, {pred[best, 3]:.4f}   (640-space)")
    log(f"  angle (rad)       : {pred[best, 5]:.6f}  = {math.degrees(pred[best, 5]):.3f} deg")
    log(f"  candidates >= {pp.CONF_THRESHOLD}: {int((scores >= pp.CONF_THRESHOLD).sum())}")

    section("Final polygons, side by side")
    ours = pp.detect(sess, img)

    from ultralytics import YOLO
    model = YOLO(str(DETECTOR_ONNX), task="obb")
    res = model.predict(source=img, imgsz=size, conf=pp.CONF_THRESHOLD,
                        iou=pp.IOU_THRESHOLD, verbose=False)
    theirs = []
    for rr in res:
        obb = getattr(rr, "obb", None)
        if obb is not None and obb.xyxyxyxy is not None:
            for poly, conf in zip(obb.xyxyxyxy.cpu().numpy(), obb.conf.cpu().numpy()):
                theirs.append({"polygon": poly.reshape(4, 2).tolist(),
                               "confidence": float(conf)})

    if not ours or not theirs:
        raise SystemExit(f"Need one detection from each (ours={len(ours)}, theirs={len(theirs)})")

    po, pt = ours[0]["polygon"], theirs[0]["polygon"]
    mo, mt = poly_metrics(po), poly_metrics(pt)

    log(f"  {'':<8}{'ours':>14}{'ultralytics':>16}{'delta':>12}")
    for k in ("cx", "cy", "w", "h", "area"):
        log(f"  {k:<8}{mo[k]:>14.3f}{mt[k]:>16.3f}{mo[k] - mt[k]:>12.3f}")
    log(f"\n  IoU: {poly_iou(po, pt):.6f}")
    log(f"  confidence delta: {abs(ours[0]['confidence'] - theirs[0]['confidence']):.8f}")

    log("\n  corner points (x, y):")
    log(f"  {'#':<4}{'ours':>22}{'ultralytics':>24}")
    for i in range(4):
        log(f"  {i:<4}{f'({po[i][0]:.2f}, {po[i][1]:.2f})':>22}"
            f"{f'({pt[i][0]:.2f}, {pt[i][1]:.2f})':>24}")

    section("Interpretation")
    dcx, dcy = mo["cx"] - mt["cx"], mo["cy"] - mt["cy"]
    dw_, dh_ = mo["w"] - mt["w"], mo["h"] - mt["h"]

    if abs(dw_) < 0.5 and abs(dh_) < 0.5 and (abs(dcx) > 0.5 or abs(dcy) > 0.5):
        log("  Same SIZE, different POSITION -> pure translation.")
        log(f"  Offset ({dcx:.2f}, {dcy:.2f}) px. This is the pad formula.")
        log("  Fix: reproduce Ultralytics' un-pad formula exactly.")
    elif abs(dw_) > 0.5 or abs(dh_) > 0.5:
        log("  Different SIZE -> the scale factor differs, not just the offset.")
        log("  Check whether Ultralytics used a different `r` (e.g. scaleup).")
    else:
        log("  Position and size both agree closely — the difference must be")
        log("  rotation. Compare the angle and the corner ordering above.")


if __name__ == "__main__":
    main()
