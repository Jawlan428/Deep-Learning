"""
verify_stage1.py — the honest re-check of Stage 1.

The first export script reported a 113x speedup. That number is not believable,
so this script exists to find out what is actually going on. Two jobs:

  PART A — Benchmark forensics.
      Work out why eager PyTorch took 665 ms for a model that should take ~15 ms,
      by controlling the one variable the first benchmark left free: thread count.

  PART B — Accuracy parity on your real validation set (optional but important).
      Run every held-out image through BOTH runtimes and confirm the exported
      model still scores what the PyTorch one scored. This is the check that
      earns you the right to say "96.3%" about the deployed system.

RUN

    Part A only:
        ..\\.venv\\Scripts\\python.exe cloud\\verify_stage1.py

    Both, if you still have the classifier dataset:
        ..\\.venv\\Scripts\\python.exe cloud\\verify_stage1.py --dataset <path-to>\\classifier_dataset\\val

    (That folder should contain subfolders named positive/ negative/ invalid/ —
     the same layout train_classifier_mobilenet.py read via CLASSIFIER_DATASET.)
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUT_DIR = HERE / "models"

CLASSIFIER_PT = PROJECT_ROOT / "python" / "classifier_mnv3.pt"
CLASSIFIER_ONNX = OUT_DIR / "classifier.onnx"
META_JSON = OUT_DIR / "classifier_meta.json"
REPORT = OUT_DIR / "verify_report.json"

CLASSES_ORDER = ["positive", "negative", "invalid"]


def log(m=""):
    print(m, flush=True)


def section(t):
    log("\n" + "=" * 72)
    log(t)
    log("=" * 72)


# ==========================================================================
# Shared: rebuild the torch model (same logic as export_onnx.py)
# ==========================================================================
def load_torch_model(meta):
    import torch
    import torch.nn as nn
    from torchvision import models

    ckpt = torch.load(str(CLASSIFIER_PT), map_location="cpu", weights_only=False)
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(meta["classes"]))
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m


def preprocess(crop_bgr, meta) -> np.ndarray:
    """Identical to detect.py._classify_crop_torch preprocessing."""
    import cv2
    size = meta["img_size"]
    mean = np.array(meta["mean"], dtype=np.float32)
    std = np.array(meta["std"], dtype=np.float32)
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)[None, ...].astype(np.float32)


# ==========================================================================
# PART A — benchmark forensics
# ==========================================================================
def cpu_sanity_check() -> dict:
    """Is the CPU itself healthy, or is the whole machine slow right now?

    Before blaming PyTorch's convolutions, check PyTorch doing something it is
    unambiguously good at: a big dense matmul, which goes straight to the MKL /
    oneDNN BLAS kernel. A 2048x2048x2048 matmul is 2*2048^3 = 17.2 GFLOP.
    A healthy 12th-gen i7 should clear ~30-100 GFLOPS here.

    If this number is ALSO terrible, the problem is the machine (power profile,
    thermal throttling, another process eating the CPU) and not the model.
    If this number is fine, the problem is specific to how MobileNet executes.
    That is the fork in the diagnosis.
    """
    import torch

    n = 2048
    a = torch.randn(n, n)
    b = torch.randn(n, n)
    for _ in range(2):          # warm up BLAS
        a @ b
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        a @ b
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    gflops = (2 * n ** 3) / med / 1e9
    log(f"  2048^3 matmul : {med * 1000:7.1f} ms  ->  {gflops:6.1f} GFLOPS")
    verdict = "CPU healthy" if gflops > 20 else "CPU UNDERPERFORMING (power/thermal/contention?)"
    log(f"  verdict       : {verdict}")
    return {"matmul_ms": round(med * 1000, 2), "gflops": round(gflops, 1), "verdict": verdict}


def bench_torch(model, arr, threads: int, runs: int, inference_mode: bool) -> float:
    """Time eager PyTorch with an explicit thread count.

    torch.set_num_threads() controls the INTRA-op pool: how many threads split
    the work inside a single operator. The default is your core count. That is
    the right call for a ResNet-50 at batch 32; for MobileNetV3-Small at batch 1
    each layer is so small that the cost of waking, synchronising and joining
    N threads can exceed the arithmetic they perform. On a hybrid P-core/E-core
    chip this is worse, because the pool gets scheduled across cores of two
    different speeds and every barrier waits for the slowest one.

    inference_mode is a strictly stronger no_grad: it also disables version
    counting and view tracking, so tensors skip autograd bookkeeping entirely.
    """
    import torch

    torch.set_num_threads(threads)
    t = torch.from_numpy(arr)
    ctx = torch.inference_mode if inference_mode else torch.no_grad

    with ctx():
        for _ in range(5):
            model(t)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(t)
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def bench_ort(arr, threads: int, runs: int) -> float:
    """Time ONNX Runtime with a MATCHING thread count, so the comparison is fair.

    intra_op_num_threads is ORT's equivalent knob. The first benchmark let each
    runtime choose its own default, which meant we were partly measuring two
    different threading strategies rather than two different execution engines.
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(CLASSIFIER_ONNX), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    for _ in range(5):
        sess.run(None, {name: arr})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: arr})
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def part_a(meta) -> dict:
    import torch

    section("PART A / STEP 1  Is the CPU itself healthy?")
    log(f"  torch {torch.__version__}   default intra-op threads: {torch.get_num_threads()}")
    sanity = cpu_sanity_check()

    section("PART A / STEP 2  Same model, same input, thread count controlled")
    log("  Both runtimes pinned to the SAME number of threads at each row.")
    log("  t3.micro has 2 vCPUs, so the '2' row is the one that predicts production.\n")

    model = load_torch_model(meta)
    size = meta["img_size"]
    arr = np.random.default_rng(1).standard_normal((1, 3, size, size)).astype(np.float32)

    log(f"  {'threads':>8} | {'torch no_grad':>14} | {'torch inf_mode':>15} | {'onnxruntime':>12} | {'ratio':>7}")
    log(f"  {'-' * 8}-+-{'-' * 14}-+-{'-' * 15}-+-{'-' * 12}-+-{'-' * 7}")

    rows = []
    for th in (1, 2, 4, torch.get_num_threads()):
        if any(r["threads"] == th for r in rows):
            continue
        t_ng = bench_torch(model, arr, th, 20, inference_mode=False)
        t_im = bench_torch(model, arr, th, 20, inference_mode=True)
        o = bench_ort(arr, th, 20)
        ratio = min(t_ng, t_im) / o
        log(f"  {th:>8} | {t_ng:>11.2f} ms | {t_im:>12.2f} ms | {o:>9.2f} ms | {ratio:>6.2f}x")
        rows.append({
            "threads": th,
            "torch_no_grad_ms": round(t_ng, 2),
            "torch_inference_mode_ms": round(t_im, 2),
            "onnx_ms": round(o, 2),
            "speedup": round(ratio, 2),
        })

    section("PART A / STEP 3  Batched rotations — the thing we actually care about")
    log("  detect.py classifies each crop at 4 rotations. Today: 4 separate calls.")
    log("  With the dynamic batch axis we can send all 4 at once. Is it worth it?\n")

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2          # simulate t3.micro
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(CLASSIFIER_ONNX), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    batch4 = np.repeat(arr, 4, axis=0)

    for _ in range(5):
        sess.run(None, {name: arr})
        sess.run(None, {name: batch4})

    seq = []
    for _ in range(20):
        t0 = time.perf_counter()
        for _ in range(4):
            sess.run(None, {name: arr})
        seq.append((time.perf_counter() - t0) * 1000)
    bat = []
    for _ in range(20):
        t0 = time.perf_counter()
        sess.run(None, {name: batch4})
        bat.append((time.perf_counter() - t0) * 1000)

    s_med, b_med = statistics.median(seq), statistics.median(bat)
    log(f"  4 sequential calls : {s_med:7.2f} ms")
    log(f"  1 batched call (4) : {b_med:7.2f} ms")
    log(f"  saving             : {(1 - b_med / s_med) * 100:5.1f}%")

    return {
        "cpu_sanity": sanity,
        "thread_sweep": rows,
        "rotation_batching": {
            "sequential_ms": round(s_med, 2),
            "batched_ms": round(b_med, 2),
            "saving_pct": round((1 - b_med / s_med) * 100, 1),
        },
    }


# ==========================================================================
# PART B — accuracy parity on the real validation set
# ==========================================================================
def part_b(meta, dataset: Path) -> dict:
    """Score the ENTIRE validation set through both runtimes.

    Parity on 10 assorted images tells you the graph converted correctly.
    It does NOT tell you the deployed model is as accurate as the one you
    trained. Only re-scoring the held-out set does that. If PyTorch and ONNX
    both land on 0.963, you have earned the right to put that number next to
    a live URL.
    """
    import cv2
    import torch
    import onnxruntime as ort

    classes = meta["classes"]
    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    items: list[tuple[Path, int]] = []
    for idx, cname in enumerate(classes):
        d = dataset / cname
        if not d.is_dir():
            log(f"  !! missing class folder: {d}")
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in exts:
                items.append((f, idx))

    if not items:
        log(f"  no images found under {dataset} — skipping Part B")
        return {"skipped": True, "reason": f"no images under {dataset}"}

    log(f"  {len(items)} images across {len(classes)} classes")
    for idx, c in enumerate(classes):
        log(f"    {c:<10} {sum(1 for _, y in items if y == idx)}")
    log("")

    model = load_torch_model(meta)
    torch.set_num_threads(2)
    sess = ort.InferenceSession(str(CLASSIFIER_ONNX), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    n = len(items)
    t_correct = o_correct = agree = 0
    max_diff = 0.0
    t_cm = np.zeros((len(classes), len(classes)), dtype=int)
    o_cm = np.zeros((len(classes), len(classes)), dtype=int)
    mismatches = []

    for i, (f, y) in enumerate(items, 1):
        img = cv2.imread(str(f))
        if img is None:
            continue
        arr = preprocess(img, meta)

        with torch.inference_mode():
            tl = model(torch.from_numpy(arr)).numpy()
        ol = sess.run(None, {name: arr})[0]

        max_diff = max(max_diff, float(np.abs(tl - ol).max()))
        ti, oi = int(tl.argmax()), int(ol.argmax())

        t_cm[y, ti] += 1
        o_cm[y, oi] += 1
        t_correct += (ti == y)
        o_correct += (oi == y)
        if ti == oi:
            agree += 1
        else:
            mismatches.append({"file": f.name, "torch": classes[ti], "onnx": classes[oi]})

        if i % 25 == 0 or i == n:
            log(f"    {i}/{n}")

    log("")
    log(f"  PyTorch accuracy : {t_correct}/{n} = {t_correct / n:.4f}")
    log(f"  ONNX    accuracy : {o_correct}/{n} = {o_correct / n:.4f}")
    log(f"  prediction agreement : {agree}/{n} = {agree / n:.4f}")
    log(f"  max logit difference : {max_diff:.3e}")

    def show_cm(cm, title):
        log(f"\n  {title}  (rows = true, cols = predicted)")
        log("    " + " " * 10 + "".join(f"{c:>10}" for c in classes))
        for i, c in enumerate(classes):
            log("    " + f"{c:>10}" + "".join(f"{cm[i][j]:>10}" for j in range(len(classes))))

    show_cm(t_cm, "PyTorch confusion matrix")
    show_cm(o_cm, "ONNX confusion matrix")

    if mismatches:
        log(f"\n  !! {len(mismatches)} image(s) where the two runtimes disagreed:")
        for m in mismatches[:10]:
            log(f"     {m['file']:<40} torch={m['torch']:<10} onnx={m['onnx']}")

    return {
        "n": n,
        "pytorch_accuracy": round(t_correct / n, 4),
        "onnx_accuracy": round(o_correct / n, 4),
        "agreement": round(agree / n, 4),
        "max_abs_logit_diff": max_diff,
        "pytorch_confusion": t_cm.tolist(),
        "onnx_confusion": o_cm.tolist(),
        "mismatches": mismatches,
        "classes": classes,
    }


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=None,
                    help="Path to classifier_dataset/val (with positive/ negative/ invalid/ subfolders)")
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()

    if not CLASSIFIER_ONNX.exists():
        raise SystemExit(f"Missing {CLASSIFIER_ONNX} — run cloud/export_onnx.py first.")

    meta = json.loads(META_JSON.read_text(encoding="utf-8"))
    report = {}

    if not args.skip_bench:
        report["part_a"] = part_a(meta)

    if args.dataset:
        section("PART B  Accuracy parity on the held-out validation set")
        report["part_b"] = part_b(meta, Path(args.dataset))
    else:
        section("PART B  SKIPPED")
        log("  No --dataset given. Parity was only checked on 10 assorted inputs,")
        log("  which proves the GRAPH converted correctly but says nothing about")
        log("  whether accuracy survived. Re-run with --dataset to close that gap.")

    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"\n  report: {REPORT}")


if __name__ == "__main__":
    main()
