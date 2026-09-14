"""
verify_pipeline.py — does the torch-free pipeline agree with detect.py?

THE QUESTION THIS ANSWERS
    You will be demoing a live URL while your JavaFX app sits on your laptop.
    If the two give different verdicts for the same photo, that is the single
    most embarrassing thing that can happen in an interview. So: same images,
    both pipelines, compare.

WHAT IS COMPARED
    OLD  python/detect.py, run exactly as the JavaFX app runs it
         (subprocess, argv, JSON on stdout). torch + ultralytics.
    NEW  cloud/pipeline.py. onnxruntime + opencv + pillow + numpy.

    Both against GROUND TRUTH, taken from the demo_examples folder names
    (positive/ negative/ invalid/ and the corrupted/ set, which should yield
    no detection at all).

EXPECT DISAGREEMENT, AND KNOW WHERE IT COMES FROM
    These are NOT expected to match 100%, and that is the point. Three
    measured differences, all deliberate:

      1. NEW resizes the classifier input with PIL, OLD with
         cv2.INTER_LINEAR. Measured: 0.9630 vs 0.8395 on the held-out set.
         NEW should be RIGHT more often, especially on faint positives.

      2. NEW does one forward pass; OLD does four rotations and keeps the
         most confident. Measured: 0.9630 vs 0.9259.

      3. NEW letterboxes to a full 640x640 square (fixed ONNX input shape);
         OLD lets Ultralytics pad to the nearest multiple of 32. Detection
         boxes differ slightly — mean polygon IoU 0.965.

    So read the table as: where they disagree, WHICH ONE WAS CORRECT?
    If NEW wins the disagreements, the port is an improvement rather than a
    regression. If OLD wins them, something in the port is wrong.

RUN
    ..\\.venv\\Scripts\\python.exe cloud\\verify_pipeline.py ^
        --images "C:\\Users\\user\\Downloads\\lft-detector-ui-main\\lft-detector-ui-main\\demo_examples"

    Skip the slow subprocess half (NEW vs ground truth only):
        ... --skip-old
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import pipeline as pl

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MODEL_DIR = HERE / "models"
DETECT_PY = PROJECT_ROOT / "python" / "detect.py"
DETECTOR_PT = PROJECT_ROOT / "best.pt"
REPORT = MODEL_DIR / "pipeline_verify.json"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABELS = ("positive", "negative", "invalid")


def log(m=""):
    print(m, flush=True)


def section(t):
    log("\n" + "=" * 78)
    log(t)
    log("=" * 78)


def imread_unicode(path: Path):
    import cv2
    data = np.fromfile(str(path), dtype=np.uint8)
    return None if data.size == 0 else cv2.imdecode(data, cv2.IMREAD_COLOR)


def ground_truth(path: Path) -> str | None:
    """Derive the expected answer from the folder / filename.

    demo_examples is laid out as positive/ negative/ invalid/ plus a set of
    corrupted_*.png files which should produce NO detection.
    """
    parts = [p.lower() for p in path.parts]
    for lbl in LABELS:
        if lbl in parts:
            return lbl
    stem = path.stem.lower()
    if stem.startswith("corrupted"):
        return "no_detection"
    for lbl in LABELS:
        if stem.startswith(lbl):
            return lbl
    return None


def run_old(image: Path, workdir: Path) -> dict:
    """Invoke detect.py the way the JavaFX app does and parse its JSON."""
    out_img = workdir / "annotated.jpg"
    out_crop = workdir / "crop.jpg"
    cmd = [sys.executable, str(DETECT_PY), str(DETECTOR_PT),
           str(image), str(out_img), str(out_crop)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=180, cwd=str(PROJECT_ROOT))
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}

    # detect.py prints its JSON as the last stdout line; the line-analysis
    # debug goes to stderr, so stdout should be clean. Be defensive anyway.
    for line in reversed((proc.stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {"error": f"no JSON on stdout (rc={proc.returncode})"}


def old_label(res: dict) -> str:
    if not res or res.get("error"):
        return "error"
    if not res.get("detections"):
        return "no_detection"
    clf = res.get("classification")
    if not clf or clf.get("error"):
        return "error"
    return clf.get("label", "error")


def new_label(res: dict) -> str:
    if res["num_detections"] == 0:
        return "no_detection"
    r = res["result"]
    return r.get("label", "error") if r else "error"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--skip-old", action="store_true")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    root = Path(args.images)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)[:args.limit]
    if not files:
        raise SystemExit(f"No images under {root}")

    section("Setup")
    log(f"  images : {len(files)}")
    log(f"  NEW    : cloud/pipeline.py  (onnxruntime + opencv + pillow)")
    log(f"  OLD    : python/detect.py   (torch + ultralytics)"
        + ("   [SKIPPED]" if args.skip_old else ""))

    t0 = time.perf_counter()
    pipe = pl.Pipeline(MODEL_DIR, threads=2)
    log(f"  pipeline startup: {(time.perf_counter() - t0) * 1000:.0f} ms "
        f"(this is your Cloud Run cold-start cost)")

    rows = []
    workdir = Path(tempfile.mkdtemp(prefix="lft_verify_"))

    section("Per-image results")
    log(f"  {'image':<26} {'truth':<13} {'NEW':<13} {'OLD':<13} {'ms':>7}  note")
    log(f"  {'-' * 26} {'-' * 13} {'-' * 13} {'-' * 13} {'-' * 7}  ----")

    for f in files:
        img = imread_unicode(f)
        if img is None:
            log(f"  {f.name[:26]:<26} unreadable")
            continue

        truth = ground_truth(f) or "?"

        t = time.perf_counter()
        new_res = pipe.analyze(img)
        ms = (time.perf_counter() - t) * 1000
        nl = new_label(new_res)

        ol = "-"
        if not args.skip_old:
            ol = old_label(run_old(f, workdir))

        note = ""
        if not args.skip_old and nl != ol:
            if nl == truth and ol != truth:
                note = "NEW correct, OLD wrong"
            elif ol == truth and nl != truth:
                note = "OLD correct, NEW wrong  <-- investigate"
            else:
                note = "differ, both wrong"
        elif nl != truth and truth != "?":
            note = "both wrong" if ol == nl else ""

        log(f"  {f.name[:26]:<26} {truth:<13} {nl:<13} {ol:<13} {ms:>6.0f}  {note}")

        rows.append({"file": f.name, "truth": truth, "new": nl, "old": ol,
                     "ms": round(ms, 1),
                     "source": (new_res.get("result") or {}).get("decision_source"),
                     "n_det": new_res["num_detections"]})

    # ---------------- summary ----------------
    section("Accuracy against ground truth")
    scored = [r for r in rows if r["truth"] != "?"]
    new_ok = sum(1 for r in scored if r["new"] == r["truth"])
    log(f"  NEW : {new_ok}/{len(scored)} = {new_ok / max(1, len(scored)):.4f}")
    if not args.skip_old:
        old_ok = sum(1 for r in scored if r["old"] == r["truth"])
        log(f"  OLD : {old_ok}/{len(scored)} = {old_ok / max(1, len(scored)):.4f}")

        section("Where they disagreed")
        diff = [r for r in scored if r["new"] != r["old"]]
        if not diff:
            log("  No disagreements at all.")
        else:
            n_win = sum(1 for r in diff if r["new"] == r["truth"])
            o_win = sum(1 for r in diff if r["old"] == r["truth"])
            log(f"  disagreements     : {len(diff)}/{len(scored)}")
            log(f"  NEW was right     : {n_win}")
            log(f"  OLD was right     : {o_win}")
            log(f"  neither was right : {len(diff) - n_win - o_win}")
            log("")
            for r in diff:
                mark = "NEW" if r["new"] == r["truth"] else ("OLD" if r["old"] == r["truth"] else "---")
                log(f"    {r['file']:<28} truth={r['truth']:<10} "
                    f"new={r['new']:<10} old={r['old']:<10} winner={mark}")
            log("")
            if o_win > n_win:
                log("  >> OLD wins more disagreements. Something in the port is")
                log("     wrong — do NOT build the API on this yet.")
            else:
                log("  >> NEW wins the disagreements. The port is an improvement,")
                log("     consistent with the preprocessing fix.")

    section("Decision sources used by NEW")
    from collections import Counter
    for src, n in Counter(r["source"] for r in rows if r["source"]).most_common():
        log(f"  {n:>3}x  {src}")
    log("")
    log("  Watch for 'weak-positive rescue'. In the desktop app that tier fired")
    log("  often, because the CNN was missing faint positives. With the resize")
    log("  fixed it should be rare — if it is, that is direct evidence the")
    log("  fusion layer was compensating for the preprocessing bug.")

    times = [r["ms"] for r in rows]
    if times:
        section("Latency (this laptop, 2 threads)")
        log(f"  median {np.median(times):.0f} ms   min {min(times):.0f}   max {max(times):.0f}")
        log("  A Cloud Run vCPU is ~3-5x slower; measure there, quote that.")

    REPORT.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    log(f"\n  report: {REPORT}")


if __name__ == "__main__":
    main()
