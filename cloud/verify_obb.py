"""
verify_obb.py — does our NumPy postprocessing match Ultralytics?

THE EXPERIMENT DESIGN
    Three pipelines run over the same photos:

      A. YOLO("best.pt")        PyTorch weights  + Ultralytics postprocessing
      B. YOLO("detector.onnx")  our ONNX export  + Ultralytics postprocessing
      C. onnxruntime            our ONNX export  + OUR NumPy postprocessing

    Comparing them pairwise isolates two independent questions:

      B vs C  -> Is OUR POSTPROCESSING CORRECT?
                 Identical model, identical preprocessing (Ultralytics also
                 letterboxes to a full 640x640 square when the backend is ONNX,
                 because the graph has a fixed input shape). So any difference
                 here is purely our decode / NMS / geometry. This must match
                 almost exactly.

      A vs B  -> Does the EXPORT + SQUARE LETTERBOX change detections?
                 With PyTorch weights Ultralytics uses auto=True and pads only
                 to the next multiple of 32 (e.g. 640x480). Our fixed graph
                 forces a full square with more grey padding. Small differences
                 here are expected and acceptable; large ones are not.

    Running only A vs C would confound the two and tell us nothing useful about
    which half was wrong.

HOW POLYGONS ARE COMPARED
    Exact convex-polygon intersection via cv2.intersectConvexConvex, then
    IoU = inter / (area1 + area2 - inter). Not the Gaussian approximation --
    for checking agreement we want the real number.

RUN
    ..\\.venv\\Scripts\\python.exe cloud\\verify_obb.py --images "<folder of photos>"

    e.g.
    ..\\.venv\\Scripts\\python.exe cloud\\verify_obb.py ^
        --images "C:\\Users\\user\\Downloads\\lft-detector-ui-main\\lft-detector-ui-main\\demo_examples"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import obb_postprocess as pp

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUT_DIR = HERE / "models"
DETECTOR_PT = PROJECT_ROOT / "best.pt"
DETECTOR_ONNX = OUT_DIR / "detector.onnx"
REPORT = OUT_DIR / "obb_verify.json"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def log(m=""):
    print(m, flush=True)


def section(t):
    log("\n" + "=" * 72)
    log(t)
    log("=" * 72)


def imread_unicode(path: Path):
    """cv2.imread fails on non-ASCII Windows paths; decode from bytes instead."""
    import cv2
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def poly_iou(p1: np.ndarray, p2: np.ndarray) -> float:
    """Exact IoU between two convex quadrilaterals."""
    import cv2
    a = np.asarray(p1, dtype=np.float32).reshape(-1, 2)
    b = np.asarray(p2, dtype=np.float32).reshape(-1, 2)
    inter, _ = cv2.intersectConvexConvex(a, b)
    area_a = abs(cv2.contourArea(a))
    area_b = abs(cv2.contourArea(b))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def run_ultralytics(model, img_bgr) -> list[dict]:
    """Detections from an Ultralytics model (works for both .pt and .onnx)."""
    res = model.predict(source=img_bgr, imgsz=pp.IMGSZ,
                        conf=pp.CONF_THRESHOLD, iou=pp.IOU_THRESHOLD,
                        verbose=False)
    out = []
    for r in res:
        obb = getattr(r, "obb", None)
        if obb is None or obb.xyxyxyxy is None:
            continue
        polys = obb.xyxyxyxy.cpu().numpy()
        confs = obb.conf.cpu().numpy()
        for poly, conf in zip(polys, confs):
            out.append({"confidence": float(conf),
                        "polygon": poly.reshape(4, 2).tolist()})
    out.sort(key=lambda d: -d["confidence"])
    return out


def compare(name_a: str, list_a: list[dict],
            name_b: str, list_b: list[dict]) -> dict:
    """Match detections greedily by IoU and report the differences."""
    n_a, n_b = len(list_a), len(list_b)
    matched = []
    used = set()

    for i, da in enumerate(list_a):
        best_j, best_iou = -1, 0.0
        for j, db in enumerate(list_b):
            if j in used:
                continue
            v = poly_iou(da["polygon"], db["polygon"])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou > 0.5:
            used.add(best_j)
            matched.append({
                "iou": best_iou,
                "conf_a": da["confidence"],
                "conf_b": list_b[best_j]["confidence"],
                "conf_delta": abs(da["confidence"] - list_b[best_j]["confidence"]),
            })

    return {
        "pair": f"{name_a} vs {name_b}",
        "count_a": n_a,
        "count_b": n_b,
        "count_match": n_a == n_b,
        "matched": len(matched),
        "unmatched_a": n_a - len(matched),
        "unmatched_b": n_b - len(matched),
        "min_iou": min([m["iou"] for m in matched], default=None),
        "mean_iou": float(np.mean([m["iou"] for m in matched])) if matched else None,
        "max_conf_delta": max([m["conf_delta"] for m in matched], default=None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Folder of FULL photos (searched recursively)")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.images)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)[:args.limit]
    if not files:
        raise SystemExit(f"No images found under {root}")

    section("Setup")
    log(f"  images   : {len(files)} from {root}")
    log(f"  settings : imgsz={pp.IMGSZ} conf={pp.CONF_THRESHOLD} iou={pp.IOU_THRESHOLD}")

    from ultralytics import YOLO
    import onnxruntime as ort

    log("\n  loading A: YOLO(best.pt) ...")
    model_pt = YOLO(str(DETECTOR_PT))
    log("  loading B: YOLO(detector.onnx) ...")
    model_onnx = YOLO(str(DETECTOR_ONNX), task="obb")
    log("  loading C: onnxruntime session ...")
    sess = ort.InferenceSession(str(DETECTOR_ONNX), providers=["CPUExecutionProvider"])

    rows = []
    agg_bc, agg_ab = [], []
    t_c = []

    section("Per-image comparison")
    log(f"  {'image':<34} {'A':>3} {'B':>3} {'C':>3}  {'B~C IoU':>9} {'A~B IoU':>9}")
    log(f"  {'-' * 34} {'-' * 3} {'-' * 3} {'-' * 3}  {'-' * 9} {'-' * 9}")

    for f in files:
        img = imread_unicode(f)
        if img is None:
            log(f"  {f.name[:34]:<34}  !! unreadable")
            continue

        det_a = run_ultralytics(model_pt, img)
        det_b = run_ultralytics(model_onnx, img)
        det_c = pp.detect(sess, img)
        # NOT timed here on purpose — see the timing pass at the end. Two
        # Ultralytics models are alive in this loop and their thread pools
        # spin-wait, which poisons any measurement taken alongside them.

        cmp_bc = compare("B", det_b, "C", det_c)
        cmp_ab = compare("A", det_a, "B", det_b)
        agg_bc.append(cmp_bc)
        agg_ab.append(cmp_ab)

        bc = f"{cmp_bc['min_iou']:.4f}" if cmp_bc["min_iou"] is not None else "   --   "
        ab = f"{cmp_ab['min_iou']:.4f}" if cmp_ab["min_iou"] is not None else "   --   "
        flag = " " if (cmp_bc["count_match"] and (cmp_bc["min_iou"] or 1) > 0.99) else "!"
        log(f" {flag}{f.name[:34]:<34} {len(det_a):>3} {len(det_b):>3} {len(det_c):>3}  {bc:>9} {ab:>9}")

        rows.append({"file": f.name, "n_a": len(det_a), "n_b": len(det_b),
                     "n_c": len(det_c), "bc": cmp_bc, "ab": cmp_ab})

    # ---------------- summary ----------------
    def summarize(aggs, title, strict: bool):
        section(title)
        count_ok = sum(1 for a in aggs if a["count_match"])
        ious = [a["min_iou"] for a in aggs if a["min_iou"] is not None]
        deltas = [a["max_conf_delta"] for a in aggs if a["max_conf_delta"] is not None]

        log(f"  images with same detection count : {count_ok}/{len(aggs)}")
        if ious:
            log(f"  worst polygon IoU                : {min(ious):.5f}")
            log(f"  mean polygon IoU                 : {np.mean(ious):.5f}")
        if deltas:
            log(f"  worst confidence difference      : {max(deltas):.5f}")

        if strict:
            ok = count_ok == len(aggs) and (not ious or min(ious) > 0.99) \
                 and (not deltas or max(deltas) < 0.01)
            log("")
            log("  >> PASS — our NumPy postprocessing reproduces Ultralytics." if ok
                else "  >> FAIL — our postprocessing differs. Do NOT build on it yet.")
            return ok
        else:
            log("")
            log("  (Differences here come from the square-letterbox change, not")
            log("   from our code. Small shifts are expected and acceptable.)")
            return True

    ok = summarize(agg_bc, "B vs C — IS OUR POSTPROCESSING CORRECT?", strict=True)
    summarize(agg_ab, "A vs B — DOES THE EXPORT CHANGE DETECTIONS?", strict=False)

    # ---------------- timing, in isolation ----------------
    # The first version of this script timed detect() inside the comparison
    # loop, where two Ultralytics models were also alive. Their thread pools
    # spin-wait after finishing work, so they were burning cores during our
    # measurement — it reported 1010 ms for a path that takes ~310 ms. Same
    # interleaving mistake that produced the bogus 113x classifier speedup.
    #
    # Fix: drop every other model, force a GC, then measure alone.
    section("Speed of the full NumPy path (measured in isolation)")

    import gc
    del model_pt, model_onnx
    gc.collect()

    timing_imgs = []
    for f in files[:8]:
        im = imread_unicode(f)
        if im is not None:
            timing_imgs.append(im)

    if timing_imgs:
        for im in timing_imgs[:2]:          # warm up
            pp.detect(sess, im)
        t_c = []
        for im in timing_imgs:
            t0 = time.perf_counter()
            pp.detect(sess, im)
            t_c.append((time.perf_counter() - t0) * 1000)

        log(f"  median detect() end-to-end : {np.median(t_c):7.1f} ms")
        log(f"  min / max                  : {min(t_c):7.1f} / {max(t_c):.1f} ms")
        log("  (letterbox + inference + NMS + geometry, nothing else running)")
        log(f"\n  For reference, bench_detector.py measured the bare model at")
        log(f"  ~310 ms on 2 threads. Anything much above that is our overhead.")

    REPORT.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    log(f"\n  report: {REPORT}")
    log("")
    log("  NEXT: if B vs C passed, the detector half of Stage 2 is done and we")
    log("  can drop ultralytics from the container entirely.")


if __name__ == "__main__":
    main()
