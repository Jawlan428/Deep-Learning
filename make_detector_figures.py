# make_detector_figures.py
# Regenerate ALL detector (YOLOv8-OBB) figures for the presentation from best.pt
# WITHOUT retraining. Runs validation with plots=True, which writes the full
# Ultralytics figure set, then copies them into ./figures with clear names.
#
# Run:
#   python make_detector_figures.py
#
# Requires the dataset YAML you trained the OBB model with. It must point at your
# images/labels and list the class name(s). Override the path via DATA_YAML env var.
#
# NOTE: the real work is inside main() guarded by `if __name__ == "__main__"`.
# On Windows this guard is REQUIRED — Ultralytics' dataloader spawns worker
# processes, and without the guard each worker re-imports the script and tries
# to spawn again (the "freeze_support / bootstrapping phase" error).

import os
import shutil
from pathlib import Path
from ultralytics import YOLO


def main():
    WEIGHTS   = Path(os.environ.get("DETECTOR_PT", "best.pt"))
    DATA_YAML = Path(os.environ.get("DATA_YAML", "dataset.yaml"))  # <-- point this at your data yaml
    FIG_DIR   = Path(os.environ.get("FIG_DIR", "figures"))
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    assert WEIGHTS.exists(), f"Weights not found: {WEIGHTS.resolve()}"
    assert DATA_YAML.exists(), (
        f"Dataset YAML not found: {DATA_YAML.resolve()}\n"
        "Set DATA_YAML to the .yaml you trained with (the one with train/val paths and names)."
    )

    model = YOLO(str(WEIGHTS))

    # Validation with plots=True regenerates: confusion matrices, PR/P/R/F1 curves,
    # and val_batch*_pred.jpg prediction previews. workers=0 avoids Windows
    # multiprocessing spawn issues entirely (val is fast on 59 images).
    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=640,
        conf=0.001,     # standard for metric curves (low conf threshold)
        iou=0.6,
        plots=True,
        save_json=False,
        workers=0,
    )

    # Locate the run folder Ultralytics just wrote (newest under runs/obb/val* etc).
    candidates = []
    for base in ["runs/obb", "runs/detect"]:
        p = Path(base)
        if p.exists():
            candidates += [d for d in p.iterdir() if d.is_dir() and d.name.startswith("val")]
    assert candidates, "Could not find the val run folder under runs/. Check Ultralytics output above."
    run_dir = max(candidates, key=lambda d: d.stat().st_mtime)
    print("Validation outputs in:", run_dir.resolve())

    # Copy every figure into ./figures with a 'detector_' prefix for tidy slide assets.
    for png in sorted(run_dir.glob("*.png")):
        shutil.copy(png, FIG_DIR / f"detector_{png.name}")
        print("Saved:", FIG_DIR / f"detector_{png.name}")
    for jpg in sorted(run_dir.glob("val_batch*_pred.jpg")):
        shutil.copy(jpg, FIG_DIR / f"detector_{jpg.name}")
        print("Saved:", FIG_DIR / f"detector_{jpg.name}")

    # Print the headline metrics so you can quote them on a slide.
    try:
        box = metrics.box
        print("\n=== Detector metrics (quote these on a slide) ===")
        print(f"mAP50    : {box.map50:.4f}")
        print(f"mAP50-95 : {box.map:.4f}")
        print(f"precision: {box.mp:.4f}")
        print(f"recall   : {box.mr:.4f}")
        with open(FIG_DIR / "detector_metrics.txt", "w") as f:
            f.write(f"mAP50={box.map50:.4f}\nmAP50-95={box.map:.4f}\n"
                    f"precision={box.mp:.4f}\nrecall={box.mr:.4f}\n")
    except Exception as e:
        print("Metric summary unavailable:", e)

    print("\nIf you still have your ORIGINAL training run folder (runs/obb/train*/),")
    print("grab results.png from it too — it shows loss + mAP curves over epochs.")


if __name__ == "__main__":
    main()