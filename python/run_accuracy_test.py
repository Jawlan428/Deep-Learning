"""
run_accuracy_test.py — End-to-end accuracy test on labeled images.

Picks N images from each class folder, normalizes them, runs full
Stage-1 + Stage-2 detect.py pipeline, and reports verdicts vs expected.
Prints every line-analysis debug line so you can see WHY each decision
was made.

Usage:
    python run_accuracy_test.py
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ---- paths ----
BASE = Path(__file__).resolve().parent.parent
MODEL_PATH  = BASE / "best.pt"
DETECT_PY   = BASE / "python" / "detect.py"
NORM_PY     = BASE / "python" / "normalize.py"
PYTHON      = sys.executable

# Labeled test-image folders, one per class. Defaults are relative to the project
# root (test_images/positive, etc.); point them at your own data via the
# POS_DIR / NEG_DIR / INV_DIR environment variables.
POS_DIR  = Path(os.environ.get("POS_DIR", BASE / "test_images" / "positive"))
NEG_DIR  = Path(os.environ.get("NEG_DIR", BASE / "test_images" / "negative"))
INV_DIR  = Path(os.environ.get("INV_DIR", BASE / "test_images" / "invalid"))

SAMPLES_PER_CLASS = 5
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".avif"}


def pick_images(folder: Path, n: int) -> list:
    imgs = [p for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return imgs[:n]


def normalize(src: Path, dst: Path) -> bool:
    r = subprocess.run(
        [PYTHON, str(NORM_PY), str(src), str(dst)],
        capture_output=True, text=True, timeout=30
    )
    try:
        j = json.loads(r.stdout.strip().splitlines()[-1])
        return j.get("ok", False)
    except Exception:
        return dst.exists()


def run_detect(img: Path, ann: Path, crop: Path):
    """Run Stage-1 + Stage-2 (with crop path → classify).  Returns (json_dict, stderr_text)."""
    r = subprocess.run(
        [PYTHON, str(DETECT_PY),
         str(MODEL_PATH), str(img), str(ann), str(crop)],
        capture_output=True, text=True, timeout=120
    )
    stderr = r.stderr.strip()
    try:
        lines = r.stdout.strip().splitlines()
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line), stderr
    except Exception:
        pass
    return {"ok": False, "error": r.stdout + r.stderr}, stderr


def verdict_for(result: dict) -> str:
    """Extract per-detection verdicts: returns list of (det_idx, label, decision_source, num_lines)."""
    if not result.get("ok"):
        return [("?", "ERROR", result.get("error", ""), None)]
    dets = result.get("detections") or []
    if not dets:
        return [("?", "NO-DETECTION", "YOLO found nothing", None)]
    out = []
    for i, d in enumerate(dets):
        clf = d.get("classification") or {}
        label = clf.get("label", "unread") if not clf.get("error") else "clf-error"
        src = clf.get("decision_source", "")
        n_lines = clf.get("num_lines")
        out.append((i + 1, label, src, n_lines))
    return out


def run_class(label: str, folder: Path, expected: str, tmpdir: Path):
    images = pick_images(folder, SAMPLES_PER_CLASS)
    if not images:
        print(f"\n  [SKIP] {label}: no images found in {folder}")
        return 0, 0

    results = []
    for img in images:
        norm_dst = tmpdir / ("norm_" + img.stem[:30] + ".jpg")
        ok = normalize(img, norm_dst)
        src = norm_dst if ok and norm_dst.exists() else img

        ann  = tmpdir / ("ann_"  + img.stem[:30] + ".jpg")
        crop = tmpdir / ("crop_" + img.stem[:30] + ".jpg")

        result, stderr = run_detect(src, ann, crop)
        verdicts = verdict_for(result)
        results.append((img.name, verdicts, stderr))

    print(f"\n{'='*64}")
    print(f"  CLASS: {label.upper()}   expected → {expected.upper()}")
    print(f"{'='*64}")

    correct = 0
    total = len(results)
    for fname, verdicts, stderr in results:
        # Print any line-analysis debug lines from stderr
        la_lines = [l for l in stderr.splitlines() if "[line-analysis]" in l or l.startswith("  line") or l.startswith("  sep") or l.startswith("  →")]
        for det_idx, label_got, src, n_lines in verdicts:
            status = "PASS" if label_got.lower() == expected.lower() else "FAIL"
            if status == "PASS":
                correct += 1
            n_str = f"  lines={n_lines}" if n_lines is not None else ""
            print(f"  [{status}] {fname[:40]:<42} det{det_idx} → {label_got:<10}{n_str}")
            if la_lines:
                for l in la_lines:
                    print(f"         {l}")
            if src:
                print(f"         how: {src[:80]}")
        if not verdicts:
            print(f"  [FAIL] {fname}: no verdicts")

    acc = correct / total if total > 0 else 0
    print(f"\n  Result: {correct}/{total} correct  ({acc*100:.0f}%)")
    return correct, total


def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: model not found at {MODEL_PATH}"); sys.exit(1)

    print(f"Model:  {MODEL_PATH}")
    print(f"Script: {DETECT_PY}")
    print(f"Python: {PYTHON}")
    print(f"Samples per class: {SAMPLES_PER_CLASS}")

    with tempfile.TemporaryDirectory(prefix="lft_test_") as td:
        tmpdir = Path(td)
        total_correct = total_tests = 0

        for lbl, folder, exp in [
            ("Positive", POS_DIR, "positive"),
            ("Negative", NEG_DIR, "negative"),
            ("Invalid",  INV_DIR, "invalid"),
        ]:
            if not folder.exists():
                print(f"\n[SKIP] {lbl}: folder not found: {folder}")
                continue
            c, t = run_class(lbl, folder, exp, tmpdir)
            total_correct += c
            total_tests   += t

    print(f"\n{'='*64}")
    print(f"  OVERALL: {total_correct}/{total_tests} correct  ({total_correct/total_tests*100:.0f}% accuracy)")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
