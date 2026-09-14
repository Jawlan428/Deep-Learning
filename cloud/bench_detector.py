"""
bench_detector.py — the number we actually need before touching AWS.

WHY THIS EXISTS
    The classifier is 1.5M parameters / ~0.06 GFLOPs. The detector is 11.4M
    parameters / 29.4 GFLOPs — about 500x the arithmetic, and it runs on every
    request. Benchmarking the classifier and calling it done was a mistake:
    it was never the bottleneck.

    If the detector takes 2 seconds per image on 2 vCPUs, a t3.micro serving a
    live demo is a bad idea and we should know that BEFORE we build a container
    around it, not after.

    Also measured in isolation (separate passes, never interleaved) — the
    interleaving bug is exactly what produced the bogus 113x number earlier.

RUN
    ..\\.venv\\Scripts\\python.exe cloud\\bench_detector.py

    Strip the absolute Windows path out of the ONNX metadata too (it names
    another machine and is about to go into a public repo):

    ..\\.venv\\Scripts\\python.exe cloud\\bench_detector.py --clean-metadata
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "models"
DETECTOR_ONNX = OUT_DIR / "detector.onnx"
CLASSIFIER_ONNX = OUT_DIR / "classifier.onnx"
REPORT = OUT_DIR / "detector_bench.json"

# t3.micro / t2.micro both expose 2 vCPUs. That row is production.
PROD_THREADS = 2


def log(m=""):
    print(m, flush=True)


def section(t):
    log("\n" + "=" * 72)
    log(t)
    log("=" * 72)


def make_session(path: Path, threads: int):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def time_session(sess, feed: dict, runs: int = 15, warmup: int = 3) -> dict:
    for _ in range(warmup):
        sess.run(None, feed)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "median_ms": round(statistics.median(times), 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
    }


def bench_detector() -> dict:
    section("STEP 1  Detector latency vs thread count")
    log("  YOLOv8s-obb, 640x640, 29.4 GFLOPs per image.")
    log("  Measured alone — no other runtime alive during timing.\n")

    img = np.random.default_rng(0).random((1, 3, 640, 640), dtype=np.float32)

    log(f"  {'threads':>8} | {'median':>10} | {'min':>9} | {'max':>9}")
    log(f"  {'-' * 8}-+-{'-' * 10}-+-{'-' * 9}-+-{'-' * 9}")

    rows = []
    import onnxruntime as ort
    for th in (1, 2, 4, 0):  # 0 = let ORT decide
        sess = make_session(DETECTOR_ONNX, th) if th else \
            ort.InferenceSession(str(DETECTOR_ONNX), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        r = time_session(sess, {name: img})
        label = str(th) if th else "auto"
        log(f"  {label:>8} | {r['median_ms']:>7.2f} ms | {r['min_ms']:>6.2f} ms | {r['max_ms']:>6.2f} ms")
        rows.append({"threads": label, **r})
        del sess
    return {"detector": rows}


def bench_pipeline() -> dict:
    section("STEP 2  Full pipeline at production thread count")
    log(f"  Simulating t3.micro: {PROD_THREADS} threads.")
    log("  detector(1 image) + classifier(4 rotations, batched) = one API request.\n")

    det = make_session(DETECTOR_ONNX, PROD_THREADS)
    clf = make_session(CLASSIFIER_ONNX, PROD_THREADS)
    det_in = det.get_inputs()[0].name
    clf_in = clf.get_inputs()[0].name

    img = np.random.default_rng(0).random((1, 3, 640, 640), dtype=np.float32)
    crops4 = np.random.default_rng(1).random((4, 3, 224, 224), dtype=np.float32)

    d = time_session(det, {det_in: img})
    c = time_session(clf, {clf_in: crops4})
    total = d["median_ms"] + c["median_ms"]

    log(f"  detector            : {d['median_ms']:8.2f} ms   ({d['median_ms'] / total * 100:.0f}% of total)")
    log(f"  classifier (batch 4): {c['median_ms']:8.2f} ms   ({c['median_ms'] / total * 100:.0f}% of total)")
    log(f"  {'-' * 44}")
    log(f"  model time / request: {total:8.2f} ms")
    log("")
    log("  Not included: JPEG decode, EXIF normalize, the rotated perspective")
    log("  crop, the classical line analysis (_analyze_lines), and HTTP overhead.")
    log("  Budget roughly 1.5-2x this figure for a real request.")

    log("\n  --- Extrapolating to t3.micro ---")
    log("  Your i7-1255U hit 132.6 GFLOPS on the matmul sanity check. A t3.micro")
    log("  vCPU is roughly 3-5x slower per core for this kind of work. So expect")
    log(f"  somewhere near {total * 3 / 1000:.1f}-{total * 5 / 1000:.1f} s of model time per request in production.")
    log("  That is an estimate from a ratio, not a measurement. Measure it on the")
    log("  real instance in Stage 4 and quote THAT number, not this one.")

    return {
        "threads": PROD_THREADS,
        "detector_ms": d["median_ms"],
        "classifier_batch4_ms": c["median_ms"],
        "model_total_ms": round(total, 2),
        "note": "laptop CPU; t3.micro expected 3-5x slower",
    }


def clean_metadata() -> dict:
    """Remove the absolute path of somebody's Desktop from the model file.

    The exported detector carries:
        description = "...trained on C:\\Users\\ASUS\\Desktop\\Deep Learning\\..."
    That is not a secret, but it is a path from another machine and this file
    is heading for a public repository. Strip it on principle: the less
    incidental information a public artifact carries, the better.
    """
    import onnx

    section("STEP 3  Strip machine-specific metadata from detector.onnx")
    m = onnx.load(str(DETECTOR_ONNX))

    before = {p.key: p.value for p in m.metadata_props}
    keep = {"stride", "task", "names", "imgsz", "batch", "channels", "end2end"}

    removed = []
    new_props = []
    for p in m.metadata_props:
        if p.key in keep:
            new_props.append((p.key, p.value))
        else:
            removed.append(p.key)

    del m.metadata_props[:]
    for k, v in new_props:
        e = m.metadata_props.add()
        e.key, e.value = k, v

    onnx.save(m, str(DETECTOR_ONNX))
    log(f"  removed : {', '.join(removed)}")
    log(f"  kept    : {', '.join(k for k, _ in new_props)}")
    log("  (kept keys are the ones postprocessing needs: stride, names, imgsz.)")

    # Confirm the model still loads and runs after the edit.
    sess = make_session(DETECTOR_ONNX, PROD_THREADS)
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: np.zeros((1, 3, 640, 640), dtype=np.float32)})[0]
    log(f"  re-checked: model still runs, output shape {out.shape}")

    return {"removed_keys": removed, "description_before": before.get("description", "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-metadata", action="store_true")
    args = ap.parse_args()

    if not DETECTOR_ONNX.exists():
        raise SystemExit(f"Missing {DETECTOR_ONNX} — run cloud/export_onnx.py first.")

    report = {}
    report.update(bench_detector())
    report["pipeline"] = bench_pipeline()
    if args.clean_metadata:
        report["metadata_clean"] = clean_metadata()

    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"\n  report: {REPORT}")


if __name__ == "__main__":
    main()
