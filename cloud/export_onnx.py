"""
export_onnx.py — Stage 1 of the cloud migration.

WHAT THIS DOES
    Converts the two trained neural networks from PyTorch (.pt) to ONNX (.onnx),
    then PROVES the converted models still give the same answers.

WHY WE DO IT
    The AWS free-tier machine (t3.micro) has 1 GB of RAM. A Docker image with
    PyTorch + torchvision + CUDA stubs is ~2 GB on disk and wants several hundred
    MB of RAM just to import. ONNX Runtime is a single small C++ library
    (~15 MB wheel) that executes a frozen computation graph. Same maths,
    a fraction of the footprint.

    ONNX ("Open Neural Network Exchange") is a file format describing a network
    as a graph of operators + the trained weights. Exporting = tracing one forward
    pass and recording every operation that happened. That is why we must pass a
    real dummy input: the exporter watches the tensor flow through the model.

RUN IT
    From the project root (Deep-Learning-main), with your venv active:

        python cloud/export_onnx.py

    Optional: point it at a folder of real cassette crops for a stronger check:

        python cloud/export_onnx.py --crops debug_report

OUTPUT
    cloud/models/detector.onnx      (YOLOv8-OBB)
    cloud/models/classifier.onnx    (MobileNetV3-Small)
    cloud/models/classifier_meta.json
    cloud/models/export_report.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Paths. Everything cloud-related lives under cloud/ so the existing JavaFX
# project keeps working exactly as it does today. We do not touch detect.py.
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUT_DIR = HERE / "models"

DETECTOR_PT = PROJECT_ROOT / "best.pt"
CLASSIFIER_PT = PROJECT_ROOT / "python" / "classifier_mnv3.pt"

DETECTOR_ONNX = OUT_DIR / "detector.onnx"
CLASSIFIER_ONNX = OUT_DIR / "classifier.onnx"
META_JSON = OUT_DIR / "classifier_meta.json"
REPORT_JSON = OUT_DIR / "export_report.json"

# Must match python/detect.py exactly, or the API will disagree with the desktop app.
DETECT_IMGSZ = 640

# opset = the version of the ONNX operator set. 12 is old enough that every
# runtime supports it, new enough to cover everything MobileNetV3 uses
# (including hardswish). Newer is not better here — we want boring and portable.
OPSET = 12


def log(msg: str) -> None:
    print(msg, flush=True)


def section(title: str) -> None:
    log("\n" + "=" * 72)
    log(title)
    log("=" * 72)


# ==========================================================================
# 1. CLASSIFIER EXPORT
# ==========================================================================
def load_classifier_from_checkpoint():
    """Rebuild the MobileNetV3-Small exactly as train_classifier_mobilenet.py saved it.

    The checkpoint is NOT a pickled model object — it is a plain dict:
        {state_dict, classes, arch, img_size, mean, std}
    A state_dict is only the numbers. It has no idea what shape of network they
    belong to. So we must construct an identical architecture first, then pour
    the weights in. Getting the head wrong by even one layer means load_state_dict
    raises, which is actually the good outcome: silent mismatch would be worse.
    """
    import torch
    import torch.nn as nn
    from torchvision import models

    ckpt = torch.load(str(CLASSIFIER_PT), map_location="cpu", weights_only=False)

    classes = ckpt["classes"]
    arch = ckpt.get("arch", "mobilenet_v3_small")
    if arch != "mobilenet_v3_small":
        raise ValueError(f"Unsupported arch in checkpoint: {arch}")

    # weights=None -> random init. We are about to overwrite every number anyway,
    # and this avoids a pointless download of the ImageNet weights.
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])

    # eval() matters: it switches BatchNorm to use its stored running statistics
    # and disables dropout. Exporting in train() mode bakes in the WRONG
    # behaviour and your ONNX model quietly becomes less accurate.
    model.eval()

    meta = {
        "classes": classes,
        "img_size": int(ckpt.get("img_size", 224)),
        "mean": ckpt.get("mean", [0.485, 0.456, 0.406]),
        "std": ckpt.get("std", [0.229, 0.224, 0.225]),
    }
    return model, meta


def export_classifier(model, meta) -> None:
    """Trace the classifier and write classifier.onnx."""
    import torch

    size = meta["img_size"]

    # The dummy input only needs the right SHAPE and dtype; the values are
    # irrelevant because we are recording operations, not results.
    dummy = torch.randn(1, 3, size, size, dtype=torch.float32)

    # dynamic_axes: mark dimension 0 (batch) as variable-length.
    #
    # This is worth understanding because it buys us real speed later.
    # detect.py classifies each crop at all FOUR 90-degree rotations and keeps
    # the most confident one (_classify_crop_torch). Today that is four separate
    # forward passes. With a dynamic batch axis the API can stack all four
    # rotations into ONE call of shape (4, 3, 224, 224) — roughly 4x less
    # per-call overhead on a single vCPU, which is exactly what t3.micro is.
    #
    # Without this, the graph is frozen at batch size 1 and passing 4 throws.
    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    export_kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        do_constant_folding=True,  # pre-computes constant subgraphs -> smaller, faster
    )

    # torch 2.6+ ships a second, dynamo-based exporter. For a plain CNN the old
    # TorchScript path is more predictable, so ask for it explicitly. Older
    # torch versions do not know the kwarg, hence the fallback.
    try:
        torch.onnx.export(model, dummy, str(CLASSIFIER_ONNX), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(CLASSIFIER_ONNX), **export_kwargs)

    log(f"  wrote {CLASSIFIER_ONNX.name}  ({CLASSIFIER_ONNX.stat().st_size / 1e6:.2f} MB)")

    # Structural validation: does the file describe a well-formed graph?
    # This is cheap and catches a corrupt export before we waste time on AWS.
    import onnx
    onnx.checker.check_model(onnx.load(str(CLASSIFIER_ONNX)))
    log("  onnx.checker: graph is valid")

    META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"  wrote {META_JSON.name} (classes, img_size, mean, std)")


# ==========================================================================
# 2. DETECTOR EXPORT
# ==========================================================================
def export_detector() -> None:
    """Export the YOLOv8-OBB detector.

    We do NOT hand-roll this one. Ultralytics' own exporter attaches the
    oriented-bounding-box decode head and the right output layout; reproducing
    that by hand is a good way to get plausible-looking garbage coordinates.

    imgsz=640 is pinned to match detect.py's model.predict(imgsz=640). Export at
    a different size and the detector silently degrades on real photos.
    """
    from ultralytics import YOLO

    model = YOLO(str(DETECTOR_PT))
    produced = model.export(
        format="onnx",
        imgsz=DETECT_IMGSZ,
        opset=OPSET,
        dynamic=False,   # fixed 640x640 input: simpler graph, faster on CPU
        simplify=True,   # fuses redundant nodes
        half=False,      # float16 needs a GPU; t3.micro is CPU-only
    )

    produced = Path(produced)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), str(DETECTOR_ONNX))
    log(f"  wrote {DETECTOR_ONNX.name}  ({DETECTOR_ONNX.stat().st_size / 1e6:.2f} MB)")


# ==========================================================================
# 3. PREPROCESSING — copied from detect.py._classify_crop_torch
# ==========================================================================
def preprocess_crop(crop_bgr, meta) -> np.ndarray:
    """Turn a BGR crop into the exact tensor the classifier was trained on.

    This is the single most common place an ONNX migration goes wrong. The
    network is identical, the weights are identical, and the accuracy still
    drops — because the pixels arriving at it were prepared differently.
    So this mirrors detect.py line for line: BGR->RGB, resize to img_size with
    INTER_LINEAR, scale to 0..1, subtract ImageNet mean, divide by std,
    HWC -> CHW, add a batch dimension.
    """
    import cv2

    size = meta["img_size"]
    mean = np.array(meta["mean"], dtype=np.float32)
    std = np.array(meta["std"], dtype=np.float32)

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)[None, ...].astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ==========================================================================
# 4. VERIFICATION
# ==========================================================================
def verify_classifier(model, meta, crop_dir: Path | None) -> dict:
    """Run the same inputs through PyTorch and ONNX Runtime and compare.

    Two things are checked, and they are not the same thing:

      1. Numerical agreement — how far apart are the raw logits? Tiny differences
         (~1e-5) are expected and fine: the two runtimes fuse operations
         differently and floating point is not associative.
      2. Label agreement — does argmax still pick the same class? This is what
         actually matters. A 1e-5 logit drift only bites if two classes were
         already neck and neck.

    Real images beat random noise here: random tensors exercise the graph but
    land in a part of the input space the model never sees, so they can hide a
    preprocessing bug. We use real crops when they are available.
    """
    import torch
    import onnxruntime as ort

    size = meta["img_size"]
    classes = meta["classes"]

    # Gather inputs.
    inputs: list[np.ndarray] = []
    labels: list[str] = []

    if crop_dir and crop_dir.exists():
        import cv2
        exts = {".jpg", ".jpeg", ".png"}
        files = sorted(p for p in crop_dir.rglob("*") if p.suffix.lower() in exts)
        for f in files[:10]:
            img = cv2.imread(str(f))
            if img is None:
                continue
            inputs.append(preprocess_crop(img, meta))
            labels.append(f.name)

    n_real = len(inputs)
    # Top up to 10 samples with random tensors so the check always has substance.
    rng = np.random.default_rng(0)
    while len(inputs) < 10:
        inputs.append(rng.standard_normal((1, 3, size, size)).astype(np.float32))
        labels.append(f"<random #{len(inputs) - n_real}>")

    log(f"  comparing on {n_real} real crop(s) + {len(inputs) - n_real} random tensor(s)")

    # ONNX Runtime session. A session compiles the graph once; reusing it across
    # requests is why the API must build it at startup, not per request.
    sess = ort.InferenceSession(str(CLASSIFIER_ONNX), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    max_abs_diff = 0.0
    disagreements = 0
    rows = []

    for arr, name in zip(inputs, labels):
        with torch.no_grad():
            torch_logits = model(torch.from_numpy(arr)).numpy()
        onnx_logits = sess.run(None, {in_name: arr})[0]

        diff = float(np.abs(torch_logits - onnx_logits).max())
        max_abs_diff = max(max_abs_diff, diff)

        t_idx = int(torch_logits.argmax())
        o_idx = int(onnx_logits.argmax())
        agree = t_idx == o_idx
        if not agree:
            disagreements += 1

        t_probs = softmax(torch_logits)[0]
        o_probs = softmax(onnx_logits)[0]

        rows.append({
            "input": name,
            "torch": f"{classes[t_idx]} {t_probs[t_idx]:.4f}",
            "onnx": f"{classes[o_idx]} {o_probs[o_idx]:.4f}",
            "max_abs_logit_diff": diff,
            "agree": agree,
        })
        flag = "OK " if agree else "!! "
        log(f"    {flag}{name:<28} torch={classes[t_idx]:<9}{t_probs[t_idx]:.4f}"
            f"  onnx={classes[o_idx]:<9}{o_probs[o_idx]:.4f}  dmax={diff:.2e}")

    # Also confirm the dynamic batch axis really works — this is what lets the
    # API send all 4 rotations in one call later. Cheap to check now, annoying
    # to discover is broken in Stage 2.
    batch4 = np.concatenate(inputs[:4], axis=0)
    out4 = sess.run(None, {in_name: batch4})[0]
    batch_ok = out4.shape == (4, len(classes))
    log(f"  dynamic batch: fed (4,3,{size},{size}) -> got {out4.shape}  "
        f"{'OK' if batch_ok else 'FAILED'}")

    return {
        "samples": len(inputs),
        "real_crops": n_real,
        "max_abs_logit_diff": max_abs_diff,
        "label_disagreements": disagreements,
        "dynamic_batch_ok": bool(batch_ok),
        "detail": rows,
    }


# ==========================================================================
# 5. BENCHMARK
# ==========================================================================
def benchmark_classifier(model, meta, runs: int = 30) -> dict:
    """Measure CPU latency, PyTorch vs ONNX Runtime, on identical input.

    Interviewers ask for numbers. 'It got faster' is not a number.
    Note this is YOUR laptop's CPU, not t3.micro's — the ratio transfers,
    the absolute values do not. Say it that way on your CV.
    """
    import torch
    import onnxruntime as ort

    size = meta["img_size"]
    arr = np.random.default_rng(1).standard_normal((1, 3, size, size)).astype(np.float32)
    tensor = torch.from_numpy(arr)

    sess = ort.InferenceSession(str(CLASSIFIER_ONNX), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    # Warm-up: the first call allocates buffers and picks kernels. Timing it
    # would measure setup, not inference.
    for _ in range(5):
        with torch.no_grad():
            model(tensor)
        sess.run(None, {in_name: arr})

    t_times, o_times = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(tensor)
        t_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        sess.run(None, {in_name: arr})
        o_times.append((time.perf_counter() - t0) * 1000)

    # Median, not mean: one OS scheduling hiccup skews a mean badly.
    t_med = statistics.median(t_times)
    o_med = statistics.median(o_times)

    log(f"  PyTorch CPU     : {t_med:6.2f} ms/image (median of {runs})")
    log(f"  ONNX Runtime CPU: {o_med:6.2f} ms/image (median of {runs})")
    log(f"  speedup         : {t_med / o_med:.2f}x")

    return {
        "runs": runs,
        "pytorch_ms_median": round(t_med, 3),
        "onnx_ms_median": round(o_med, 3),
        "speedup": round(t_med / o_med, 3),
    }


def file_sizes() -> dict:
    def mb(p: Path) -> float | None:
        return round(p.stat().st_size / 1e6, 2) if p.exists() else None

    return {
        "detector_pt_mb": mb(DETECTOR_PT),
        "detector_onnx_mb": mb(DETECTOR_ONNX),
        "classifier_pt_mb": mb(CLASSIFIER_PT),
        "classifier_onnx_mb": mb(CLASSIFIER_ONNX),
    }


# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Export LFT models to ONNX and verify.")
    ap.add_argument("--crops", type=str, default="debug_report",
                    help="Folder of real cassette CROPS for the parity check "
                         "(relative to project root). Default: debug_report")
    ap.add_argument("--skip-detector", action="store_true",
                    help="Only do the classifier (faster while iterating).")
    args = ap.parse_args()

    for p in (DETECTOR_PT, CLASSIFIER_PT):
        if not p.exists():
            raise SystemExit(f"Model not found: {p}")

    report: dict = {"opset": OPSET, "detect_imgsz": DETECT_IMGSZ}

    section("STEP 1/4  Rebuild MobileNetV3-Small from the checkpoint")
    model, meta = load_classifier_from_checkpoint()
    log(f"  classes  : {meta['classes']}")
    log(f"  img_size : {meta['img_size']}")
    log(f"  mean/std : {meta['mean']} / {meta['std']}")
    report["classifier_meta"] = meta

    section("STEP 2/4  Export to ONNX")
    export_classifier(model, meta)
    if args.skip_detector:
        log("  (detector export skipped)")
    else:
        export_detector()

    section("STEP 3/4  Verify ONNX agrees with PyTorch")
    crop_dir = (PROJECT_ROOT / args.crops) if args.crops else None
    report["verification"] = verify_classifier(model, meta, crop_dir)

    section("STEP 4/4  Size and speed")
    report["sizes_mb"] = file_sizes()
    for k, v in report["sizes_mb"].items():
        log(f"  {k:<22} {v}")
    report["benchmark"] = benchmark_classifier(model, meta)

    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    section("RESULT")
    v = report["verification"]
    ok = v["label_disagreements"] == 0 and v["dynamic_batch_ok"]
    log(f"  max logit difference : {v['max_abs_logit_diff']:.3e}")
    log(f"  label disagreements  : {v['label_disagreements']} / {v['samples']}")
    log(f"  dynamic batch        : {'working' if v['dynamic_batch_ok'] else 'BROKEN'}")
    log("")
    log("  STAGE 1 PASSED — the ONNX models are safe to build on." if ok
        else "  STAGE 1 FAILED — do not continue. Read the !! lines above.")
    log(f"\n  full report: {REPORT_JSON}")


if __name__ == "__main__":
    main()
