"""
preprocessing_experiment2.py — close the last 2-image gap, and test the
rotation-aggregation hypothesis.

WHERE WE ARE
    A  PIL + torchvision (how you validated)      0.9630   <- reference
    B  cv2 INTER_LINEAR (detect.py today)         0.8395
    C  cv2 INTER_AREA                             0.9383
    D  C + 4 rotations, max-confidence wins       0.8889

    Two open questions:
      1. Why does C still trail A by 2 images?
      2. Why does adding rotations (D) make things WORSE?

QUESTION 1 — the residual gap
    In production we never have a file path; we have a BGR crop in memory
    (output of rotated_crop). So the realistic options are:

      E  cv2 decode -> PIL resize     Does PIL's RESIZE account for the gap?
      A  PIL decode -> PIL resize     (control, already measured at 0.9630)

    If E == A, the JPEG decoder is irrelevant and PIL's resize kernel is the
    whole story — meaning we should simply use PIL to resize in the API and
    the train/inference skew disappears completely.

    If E < A, something about DECODING differs. The prime suspect is EXIF:
    cv2.imdecode(IMREAD_COLOR) applies orientation tags; PIL.Image.open does
    not. That would mean the two paths see literally different pictures.
    G measures that directly.

QUESTION 2 — the rotation aggregation
    _classify_crop_torch picks the rotation with the highest top-class
    probability. Out-of-distribution inputs (a sideways cassette) produce
    CONFIDENTLY WRONG softmax outputs, so "most confident wins" may be
    selecting whichever rotation fooled the model hardest.

      H  4 rotations, AVERAGE the probabilities   (principled aggregation)
      I  no rotation at all, PIL resize           (is rotation needed here?)

    If H > D, the rotation idea is fine and the AGGREGATION RULE is the bug.
    If I >= H, rotation is not earning its keep on correctly-oriented crops.

RUN
    ..\\.venv\\Scripts\\python.exe cloud\\preprocessing_experiment2.py ^
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
REPORT = OUT_DIR / "preprocessing_experiment2.json"


def log(m=""):
    print(m, flush=True)


def section(t):
    log("\n" + "=" * 72)
    log(t)
    log("=" * 72)


def imread_unicode_bgr(path: Path):
    """Unicode-safe cv2 decode. IMREAD_COLOR applies EXIF orientation."""
    import cv2
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imread_unicode_bgr_noexif(path: Path):
    """Same, but explicitly ignore EXIF orientation (matches PIL's behaviour)."""
    import cv2
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)


def pil_tf(meta):
    from torchvision import transforms
    size = meta["img_size"]
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(meta["mean"], meta["std"]),
    ])


# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------
def prep_A_pil_decode(path, meta, tf):
    """Control: PIL decode + PIL resize. Should reproduce 0.9630."""
    from PIL import Image
    with Image.open(str(path)) as im:
        return tf(im.convert("RGB")).unsqueeze(0).numpy()


def prep_E_cv2_decode_pil_resize(path, meta, tf):
    """Production-realistic: we have a BGR array, resize it with PIL."""
    import cv2
    from PIL import Image
    bgr = imread_unicode_bgr(path)
    if bgr is None:
        return None
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return tf(pil).unsqueeze(0).numpy()


def prep_E2_cv2_noexif_pil_resize(path, meta, tf):
    """Same as E but EXIF orientation suppressed, matching PIL exactly."""
    import cv2
    from PIL import Image
    bgr = imread_unicode_bgr_noexif(path)
    if bgr is None:
        return None
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return tf(pil).unsqueeze(0).numpy()


def prep_H_rot4(path, meta, tf):
    """4 rotations via PIL resize; caller AVERAGES the probabilities."""
    import cv2
    from PIL import Image
    bgr = imread_unicode_bgr(path)
    if bgr is None:
        return None
    rots = [
        bgr,
        cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(bgr, cv2.ROTATE_180),
        cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    arrs = [tf(Image.fromarray(cv2.cvtColor(r, cv2.COLOR_BGR2RGB))).unsqueeze(0).numpy()
            for r in rots]
    return np.concatenate(arrs, axis=0)


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


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def evaluate(model, meta, items, prep, tf, mode="single"):
    """mode: 'single' | 'rot_max' | 'rot_mean'"""
    import torch

    k = len(meta["classes"])
    cm = np.zeros((k, k), dtype=int)
    correct = scored = skipped = 0

    for path, y in items:
        arr = prep(path, meta, tf)
        if arr is None:
            skipped += 1
            continue
        with torch.inference_mode():
            logits = model(torch.from_numpy(arr)).numpy()
        probs = softmax(logits)
        if mode == "single":
            pred = int(probs[0].argmax())
        elif mode == "rot_max":
            best = int(probs.max(1).argmax())
            pred = int(probs[best].argmax())
        else:  # rot_mean
            pred = int(probs.mean(axis=0).argmax())
        cm[y, pred] += 1
        correct += (pred == y)
        scored += 1

    return {
        "scored": scored, "skipped": skipped, "correct": correct,
        "accuracy": round(correct / scored, 4) if scored else 0.0,
        "confusion": cm.tolist(),
    }


def exif_audit(items) -> dict:
    """How many images actually carry an EXIF orientation tag?

    If the answer is zero, EXIF cannot explain the A-vs-C gap and the
    difference must be in the resize kernel or the JPEG decoder.
    """
    from PIL import Image, ExifTags

    orient_key = next((k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None)
    tagged = []
    for path, _ in items:
        try:
            with Image.open(str(path)) as im:
                ex = im.getexif()
                val = ex.get(orient_key) if ex else None
                if val and val != 1:
                    tagged.append((path.name, val))
        except Exception:
            pass
    return {"with_orientation_tag": len(tagged), "examples": tagged[:5]}


def show_cm(cm, classes):
    log("    " + " " * 10 + "".join(f"{c:>10}" for c in classes))
    for i, c in enumerate(classes):
        log("    " + f"{c:>10}" + "".join(f"{cm[i][j]:>10}" for j in range(len(classes))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    meta = json.loads(META_JSON.read_text(encoding="utf-8"))
    classes = meta["classes"]
    tf = pil_tf(meta)
    dataset = Path(args.dataset)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    items = []
    for idx, c in enumerate(classes):
        d = dataset / c
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in exts:
                    items.append((f, idx))

    section("EXIF audit — can orientation explain the gap?")
    ex = exif_audit(items)
    log(f"  images carrying a non-trivial EXIF Orientation tag: "
        f"{ex['with_orientation_tag']} / {len(items)}")
    for name, val in ex["examples"]:
        log(f"    {name}  orientation={val}")
    if ex["with_orientation_tag"] == 0:
        log("  -> EXIF is ruled out. The gap is the resize kernel or the decoder.")
    else:
        log("  -> EXIF is a live suspect: cv2 applies it, PIL does not.")

    model = load_model(meta)

    section("Question 1 — does PIL resize close the gap?")
    runs = [
        ("A   PIL decode  + PIL resize  (control)", prep_A_pil_decode, "single"),
        ("E   cv2 decode  + PIL resize", prep_E_cv2_decode_pil_resize, "single"),
        ("E2  cv2 no-EXIF + PIL resize", prep_E2_cv2_noexif_pil_resize, "single"),
    ]
    results = {}
    for label, fn, mode in runs:
        r = evaluate(model, meta, items, fn, tf, mode)
        results[label] = r
        log(f"  {label:<42} {r['accuracy']:.4f}  ({r['correct']}/{r['scored']})")

    section("Question 2 — is the rotation AGGREGATION the bug?")
    for label, mode in [("D2  4 rot, max-confidence (detect.py rule)", "rot_max"),
                        ("H   4 rot, mean probability", "rot_mean")]:
        r = evaluate(model, meta, items, prep_H_rot4, tf, mode)
        results[label] = r
        log(f"  {label:<42} {r['accuracy']:.4f}  ({r['correct']}/{r['scored']})")

    section("Confusion matrices")
    for label, r in results.items():
        log(f"\n  {label}")
        show_cm(r["confusion"], classes)

    section("Verdict")
    a = results["A   PIL decode  + PIL resize  (control)"]["accuracy"]
    e = results["E   cv2 decode  + PIL resize"]["accuracy"]
    e2 = results["E2  cv2 no-EXIF + PIL resize"]["accuracy"]
    dmax = results["D2  4 rot, max-confidence (detect.py rule)"]["accuracy"]
    h = results["H   4 rot, mean probability"]["accuracy"]

    log(f"  A  control                  {a:.4f}")
    log(f"  E  cv2 decode + PIL resize  {e:.4f}   (delta {e - a:+.4f})")
    log(f"  E2 EXIF suppressed          {e2:.4f}   (delta {e2 - a:+.4f})")
    log("")
    if abs(e - a) < 0.005:
        log("  >> Use PIL for the classifier resize in the API. The decoder does")
        log("     not matter; the resize kernel was the whole story. This makes")
        log("     the inference path identical to how you validated.")
    elif e2 > e:
        log("  >> EXIF orientation is part of the gap. Suppress it (or apply it")
        log("     consistently in BOTH paths) as well as using PIL resize.")
    else:
        log("  >> PIL resize does not fully close it either. Next suspect is the")
        log("     JPEG decoder itself (libjpeg vs libjpeg-turbo).")

    log("")
    log(f"  D2 rotations, max-conf      {dmax:.4f}")
    log(f"  H  rotations, mean-prob     {h:.4f}   (delta {h - dmax:+.4f})")
    log(f"  A  no rotation at all       {a:.4f}   (delta vs H {a - h:+.4f})")
    log("")
    if h > dmax + 0.01:
        log("  >> The rotation IDEA is sound; the AGGREGATION RULE is the bug.")
        log("     'Most confident rotation wins' rewards out-of-distribution")
        log("     overconfidence. Averaging the probabilities is better.")
    if a >= h:
        log("  >> But on correctly-oriented crops, NO rotation still beats both.")
        log("     _to_portrait already normalises orientation, so the 4-rotation")
        log("     pass is mostly adding noise here. It may still earn its place")
        log("     on near-square OBBs in production — this val set cannot tell us.")

    REPORT.write_text(json.dumps(
        {"dataset": str(dataset), "exif": ex, "results": results, "classes": classes},
        indent=2), encoding="utf-8")
    log(f"\n  report: {REPORT}")


if __name__ == "__main__":
    main()
