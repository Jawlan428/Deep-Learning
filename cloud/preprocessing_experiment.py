"""
preprocessing_experiment.py — why did accuracy come out at 84%, not 96%?

THE FINDING THAT PROMPTED THIS
    Part B scored the held-out val set at ~0.84. Your training report says
    0.963 on the SAME 81 images with the SAME model. ONNX is not the culprit:
    PyTorch and ONNX produced byte-for-byte identical confusion matrices and
    agreed on 100% of images they both saw. Something else is losing accuracy.

THE HYPOTHESIS
    The images are resized differently at training time and at inference time.

      * train_classifier_mobilenet.py validated with torchvision's eval_tf,
        which resizes a PIL image. PIL's resize is AREA-AVERAGED (antialiased):
        shrinking 1000px -> 224px, every source pixel contributes.

      * detect.py (and therefore my ONNX preprocessing, which deliberately
        copies it) uses cv2.resize(..., INTER_LINEAR). On a large DOWNSCALE
        that samples only a 2x2 neighbourhood per output pixel and ignores
        everything between. It aliases.

    A faint LFT test line is a thin, low-contrast, near-horizontal feature —
    exactly the kind of detail aliasing destroys. Drop the line, and a positive
    reads as negative.

    Which is suspicious, because detect.py's own docstring says:
        "The neural classifier ... tends to miss FAINT positive test lines
         and report 'negative'."
    and the whole classical line-detection + fusion layer was built to
    compensate for it. If this hypothesis holds, that layer is compensating
    for a resize bug.

WHAT THIS SCRIPT DOES
    Runs the same model over the same images with four preprocessing variants
    and prints accuracy for each. One variable changed at a time.

      A. PIL + torchvision      — exactly what training/validation used
      B. cv2 INTER_LINEAR       — exactly what detect.py + our ONNX path use
      C. cv2 INTER_AREA         — antialiased downscale; the candidate fix
      D. cv2 INTER_AREA x4 rot  — plus the 4-rotation trick detect.py applies

    If A ~ 0.96 and B ~ 0.84, the hypothesis is confirmed and we fix the
    resize before shipping anything to the cloud.

ALSO FIXED HERE
    * Unicode filenames. cv2.imread() calls the ANSI Windows API and returns
      None for any path with Hebrew (or any non-Latin-1) characters. Seven of
      your files hit this. np.fromfile + cv2.imdecode goes through Python's
      own file handling and works.
    * The denominator bug in verify_stage1.py: it divided by len(items)=81
      while silently skipping the 7 unreadable files, so 62/74 was reported
      as 62/81. Real accuracy was 0.838, not 0.765.

RUN
    ..\\.venv\\Scripts\\python.exe cloud\\preprocessing_experiment.py ^
        --dataset "C:\\Users\\user\\Downloads\\classifier_dataset\\val"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUT_DIR = HERE / "models"
CLASSIFIER_PT = PROJECT_ROOT / "python" / "classifier_mnv3.pt"
META_JSON = OUT_DIR / "classifier_meta.json"
REPORT = OUT_DIR / "preprocessing_experiment.json"


def log(m=""):
    print(m, flush=True)


def section(t):
    log("\n" + "=" * 72)
    log(t)
    log("=" * 72)


# --------------------------------------------------------------------------
# Unicode-safe image read (the cv2.imread-on-Windows trap)
# --------------------------------------------------------------------------
def imread_unicode(path: Path):
    """cv2.imread() replacement that survives non-ASCII paths on Windows.

    cv2.imread passes the filename to the C++ layer, which on Windows uses the
    ANSI API. A Hebrew filename cannot be represented in the system ANSI code
    page, so the open fails and imread returns None — with no exception, which
    is why it silently skipped 7 files instead of crashing.

    Reading the bytes in Python (which handles UTF-16 paths properly) and
    decoding them from memory sidesteps the filename entirely.
    """
    import cv2
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# --------------------------------------------------------------------------
# The four preprocessing variants
# --------------------------------------------------------------------------
def prep_pil(path: Path, meta) -> np.ndarray | None:
    """A. Exactly what train_classifier_mobilenet.py's eval_tf did."""
    from PIL import Image
    from torchvision import transforms

    size = meta["img_size"]
    tf = transforms.Compose([
        transforms.Resize((size, size)),        # PIL resize == antialiased
        transforms.ToTensor(),
        transforms.Normalize(meta["mean"], meta["std"]),
    ])
    try:
        with Image.open(str(path)) as im:
            return tf(im.convert("RGB")).unsqueeze(0).numpy()
    except Exception:
        return None


def _cv2_common(img_bgr, meta, interp) -> np.ndarray:
    import cv2
    size = meta["img_size"]
    mean = np.array(meta["mean"], dtype=np.float32)
    std = np.array(meta["std"], dtype=np.float32)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=interp)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)[None, ...].astype(np.float32)


def prep_cv2_linear(path: Path, meta):
    """B. Exactly what detect.py and our ONNX path do today."""
    import cv2
    img = imread_unicode(path)
    return None if img is None else _cv2_common(img, meta, cv2.INTER_LINEAR)


def prep_cv2_area(path: Path, meta):
    """C. Same, but area-averaged — the standard choice for downscaling."""
    import cv2
    img = imread_unicode(path)
    return None if img is None else _cv2_common(img, meta, cv2.INTER_AREA)


def prep_cv2_area_rot4(path: Path, meta):
    """D. Area resize + the 4-rotation pass detect.py applies at inference.

    Returns a (4,3,H,W) batch. The caller takes whichever rotation is most
    confident, mirroring _classify_crop_torch.
    """
    import cv2
    img = imread_unicode(path)
    if img is None:
        return None
    rots = [
        img,
        cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(img, cv2.ROTATE_180),
        cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    return np.concatenate([_cv2_common(r, meta, cv2.INTER_AREA) for r in rots], axis=0)


VARIANTS = {
    "A  PIL + torchvision (training-time)": prep_pil,
    "B  cv2 INTER_LINEAR (detect.py today)": prep_cv2_linear,
    "C  cv2 INTER_AREA (candidate fix)": prep_cv2_area,
    "D  cv2 INTER_AREA + 4 rotations": prep_cv2_area_rot4,
}


# --------------------------------------------------------------------------
def load_model(meta):
    import torch
    import torch.nn as nn
    from torchvision import models

    ckpt = torch.load(str(CLASSIFIER_PT), map_location="cpu", weights_only=False)
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(meta["classes"]))
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    torch.set_num_threads(2)
    return m


def evaluate(model, meta, items, prep_fn, rot4: bool) -> dict:
    import torch

    classes = meta["classes"]
    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    correct = scored = skipped = 0

    for path, y in items:
        arr = prep_fn(path, meta)
        if arr is None:
            skipped += 1
            continue
        with torch.inference_mode():
            logits = model(torch.from_numpy(arr)).numpy()
        if rot4:
            probs = np.exp(logits - logits.max(1, keepdims=True))
            probs /= probs.sum(1, keepdims=True)
            best = int(probs.max(1).argmax())     # most confident rotation
            pred = int(probs[best].argmax())
        else:
            pred = int(logits.argmax())
        cm[y, pred] += 1
        correct += (pred == y)
        scored += 1

    return {
        "scored": scored,
        "skipped": skipped,
        "correct": correct,
        # NOTE: denominator is `scored`, not len(items). That was the bug.
        "accuracy": round(correct / scored, 4) if scored else 0.0,
        "confusion": cm.tolist(),
    }


def show_cm(cm, classes, indent="    "):
    log(indent + " " * 10 + "".join(f"{c:>10}" for c in classes))
    for i, c in enumerate(classes):
        log(indent + f"{c:>10}" + "".join(f"{cm[i][j]:>10}" for j in range(len(classes))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    meta = json.loads(META_JSON.read_text(encoding="utf-8"))
    classes = meta["classes"]
    dataset = Path(args.dataset)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    items = []
    for idx, cname in enumerate(classes):
        d = dataset / cname
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in exts:
                    items.append((f, idx))

    section("Dataset")
    log(f"  {dataset}")
    log(f"  {len(items)} images")
    for idx, c in enumerate(classes):
        log(f"    {c:<10} {sum(1 for _, y in items if y == idx)}")

    # Confirm the Unicode fix actually recovers the 7 lost files.
    import cv2
    lost = sum(1 for f, _ in items if cv2.imread(str(f)) is None)
    saved = sum(1 for f, _ in items if imread_unicode(f) is not None)
    log(f"\n  cv2.imread failures      : {lost}")
    log(f"  imread_unicode successes : {saved}/{len(items)}")

    model = load_model(meta)

    section("Accuracy by preprocessing variant")
    log("  Same model. Same images. One variable changed.\n")

    results = {}
    for label, fn in VARIANTS.items():
        rot4 = "4 rotations" in label
        r = evaluate(model, meta, items, fn, rot4)
        results[label] = r
        log(f"  {label:<40} {r['accuracy']:.4f}   "
            f"({r['correct']}/{r['scored']}, skipped {r['skipped']})")

    section("Confusion matrices")
    for label, r in results.items():
        log(f"\n  {label}")
        show_cm(r["confusion"], classes)

    # ---- interpretation -------------------------------------------------
    section("What this means")
    a = results["A  PIL + torchvision (training-time)"]["accuracy"]
    b = results["B  cv2 INTER_LINEAR (detect.py today)"]["accuracy"]
    c = results["C  cv2 INTER_AREA (candidate fix)"]["accuracy"]
    d = results["D  cv2 INTER_AREA + 4 rotations"]["accuracy"]

    log(f"  A training-time preprocessing : {a:.4f}")
    log(f"  B what detect.py does today   : {b:.4f}   (delta vs A: {b - a:+.4f})")
    log(f"  C antialiased downscale       : {c:.4f}   (delta vs B: {c - b:+.4f})")
    log(f"  D C + 4-rotation vote         : {d:.4f}   (delta vs C: {d - c:+.4f})")
    log("")

    if a - b > 0.03:
        log("  >> CONFIRMED: the resize is losing real accuracy.")
        log("     The model is fine. detect.py's INTER_LINEAR downscale aliases")
        log("     away faint test lines before the network ever sees them.")
        if c >= a - 0.02:
            log("     INTER_AREA recovers it. That is a one-line fix, and it")
            log("     affects your DESKTOP APP too, not just the cloud port.")
        else:
            log("     INTER_AREA helps but does not fully recover it — there is")
            log("     a second difference between PIL and cv2 still to find.")
    else:
        log("  >> NOT CONFIRMED: resize is not the main cause. The gap between")
        log("     0.963 and what we measured lies somewhere else — look next at")
        log("     whether classifier_mnv3.pt is the same checkpoint that")
        log("     produced figures/classifier_classification_report.txt.")

    REPORT.write_text(json.dumps(
        {"dataset": str(dataset), "results": results, "classes": classes},
        indent=2), encoding="utf-8")
    log(f"\n  report: {REPORT}")


if __name__ == "__main__":
    main()
