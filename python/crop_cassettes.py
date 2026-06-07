"""
crop_cassettes.py — Phase 2 dataset preparation.

Walks a labeled image folder, runs the existing YOLOv8-OBB detector on each
image, and saves a rotated, upright crop of the highest-confidence cassette
into a new dataset structure for training the classifier.

Expected input layout (your existing Project 1 train data):

    <input_root>/
        positive/   *.jpg
        negative/   *.jpg
        invalid/    *.jpg

Output layout (ready to be passed to `yolo classify train data=...`):

    <output_root>/
        train/
            positive/  *.jpg
            negative/  *.jpg
            invalid/   *.jpg
        val/
            positive/  *.jpg
            negative/  *.jpg
            invalid/   *.jpg

Usage:
    python crop_cassettes.py <model_path> <input_root> <output_root> [--val-split 0.2] [--conf 0.4] [--pad 0.04]

Notes:
    * EXIF orientation is baked into pixels before detection (same approach as normalize.py).
    * Only the highest-confidence detection per image is kept.
    * If an image has no detection above `--conf`, it is logged and skipped.
    * The crop is a TRUE rotated crop: the cassette comes out upright with
      minimal background, regardless of the original angle in the photo.
    * A small padding (`--pad`, as a fraction of the longer OBB edge) is added
      around the cassette so the lines aren't clipped at the very edge of the crop.
    * A stratified train/val split is performed per class with a fixed seed
      so the result is reproducible.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

CLASSES = ("positive", "negative", "invalid")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


# ---------------------------------------------------------------------------
# Crop math
# ---------------------------------------------------------------------------

def _order_polygon(poly):
    """
    Given 4 points forming a (possibly rotated) rectangle, return them in a
    consistent order: top-left, top-right, bottom-right, bottom-left of the
    rectangle in its OWN frame (long edge horizontal).

    This is what cv2.getPerspectiveTransform expects, and is what gives us a
    consistent upright crop regardless of which corner the detector emitted first.
    """
    import numpy as np

    poly = np.asarray(poly, dtype=np.float32).reshape(4, 2)

    # Compute the four edge vectors and find the longest one.
    edges = []
    for i in range(4):
        p1 = poly[i]
        p2 = poly[(i + 1) % 4]
        edges.append((i, p2 - p1, float(np.linalg.norm(p2 - p1))))
    edges.sort(key=lambda e: e[2], reverse=True)
    long_start_idx = edges[0][0]

    # Walk the polygon starting at the start of the longest edge. That edge
    # becomes the "top" of the upright rectangle.
    ordered = np.array([
        poly[long_start_idx],
        poly[(long_start_idx + 1) % 4],
        poly[(long_start_idx + 2) % 4],
        poly[(long_start_idx + 3) % 4],
    ], dtype=np.float32)

    # The longest edge runs from ordered[0] -> ordered[1]. Decide whether to
    # flip vertically so the crop has a stable "up" direction: we want the
    # mean y of the top edge to be smaller (higher in the image) than the
    # bottom edge in the SOURCE coordinates. If not, rotate the order by 2.
    top_mean_y = (ordered[0][1] + ordered[1][1]) / 2
    bot_mean_y = (ordered[2][1] + ordered[3][1]) / 2
    if top_mean_y > bot_mean_y:
        ordered = np.array([ordered[2], ordered[3], ordered[0], ordered[1]],
                           dtype=np.float32)

    return ordered


def rotated_crop(image_bgr, polygon, pad_frac: float = 0.04):
    """
    Warp the region defined by `polygon` (4 points, oriented bounding box)
    into an upright axis-aligned image.

    Args:
        image_bgr: the source image (OpenCV BGR).
        polygon:   4x2 array-like in source coordinates.
        pad_frac:  fraction of the longer edge to add as padding on every side.

    Returns:
        A new image (BGR) containing the cassette, upright, with minimal background.
    """
    import cv2
    import numpy as np

    src_pts = _order_polygon(polygon)

    # Long edge -> width, short edge -> height.
    width = float(np.linalg.norm(src_pts[1] - src_pts[0]))
    height = float(np.linalg.norm(src_pts[2] - src_pts[1]))

    # The cassette is taller than it is wide in real life. After we placed the
    # longest edge as the "top", `width` will be the long axis. We want the
    # final upright crop to have the cassette VERTICAL (taller than wide), so
    # we'll output width=short_axis, height=long_axis and swap accordingly.
    long_axis = max(width, height)
    short_axis = min(width, height)

    pad = int(round(pad_frac * long_axis))
    out_w = int(round(short_axis)) + 2 * pad
    out_h = int(round(long_axis)) + 2 * pad

    # Build destination points so the cassette ends up upright (vertical).
    # If the polygon's "top" edge (longest) was horizontal, we rotate so that
    # the long axis becomes vertical in the output.
    if width >= height:
        # long edge is currently horizontal in src_pts. Rotate 90deg CW in output.
        dst_pts = np.array([
            [out_w - 1 - pad, pad],            # was top-left
            [out_w - 1 - pad, out_h - 1 - pad],# was top-right
            [pad,             out_h - 1 - pad],# was bottom-right
            [pad,             pad],            # was bottom-left
        ], dtype=np.float32)
    else:
        # long edge is currently vertical, output is already vertical.
        dst_pts = np.array([
            [pad,             pad],
            [out_w - 1 - pad, pad],
            [out_w - 1 - pad, out_h - 1 - pad],
            [pad,             out_h - 1 - pad],
        ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(
        image_bgr, M, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


# ---------------------------------------------------------------------------
# Detector wrapper
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    polygon: list   # 4x2
    confidence: float


def best_detection(model, image_bgr, conf_threshold: float) -> Optional[Detection]:
    """Run the detector once on the image and return the highest-confidence OBB, or None."""
    results = model.predict(
        source=image_bgr,
        imgsz=640,
        conf=conf_threshold,
        iou=0.5,
        verbose=False,
    )
    best: Optional[Detection] = None
    for r in results:
        obb = getattr(r, "obb", None)
        if obb is None or obb.xyxyxyxy is None:
            continue
        polys = obb.xyxyxyxy.cpu().numpy()
        confs = obb.conf.cpu().numpy()
        for poly, conf in zip(polys, confs):
            if best is None or float(conf) > best.confidence:
                best = Detection(polygon=poly.tolist(), confidence=float(conf))
    return best


# ---------------------------------------------------------------------------
# Image loading with EXIF normalization
# ---------------------------------------------------------------------------

def load_image_normalized(path: Path):
    """Load an image, applying EXIF orientation to pixels (returns OpenCV BGR)."""
    from PIL import Image, ImageOps
    import numpy as np
    import cv2

    with Image.open(str(path)) as im:
        up = ImageOps.exif_transpose(im).convert("RGB")
        rgb = np.array(up)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------

def iter_images(folder: Path) -> Iterable[Path]:
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def process_class(
    class_name: str,
    input_dir: Path,
    output_train_dir: Path,
    output_val_dir: Path,
    model,
    val_split: float,
    conf_threshold: float,
    pad_frac: float,
) -> dict:
    import cv2

    output_train_dir.mkdir(parents=True, exist_ok=True)
    output_val_dir.mkdir(parents=True, exist_ok=True)

    images = list(iter_images(input_dir))
    rng = random.Random(SEED + hash(class_name) % 1000)
    # Stratified split: decide train/val per-image up front.
    indices = list(range(len(images)))
    rng.shuffle(indices)
    n_val = int(round(len(images) * val_split))
    val_indices = set(indices[:n_val])

    stats = {
        "class": class_name,
        "total": len(images),
        "train_saved": 0,
        "val_saved": 0,
        "no_detection": 0,
        "failed": 0,
        "skipped_files": [],
    }

    for i, img_path in enumerate(images):
        target_dir = output_val_dir if i in val_indices else output_train_dir
        try:
            img = load_image_normalized(img_path)
        except Exception as e:
            stats["failed"] += 1
            stats["skipped_files"].append({"file": img_path.name, "reason": f"load error: {e}"})
            continue

        det = best_detection(model, img, conf_threshold)
        if det is None:
            stats["no_detection"] += 1
            stats["skipped_files"].append({"file": img_path.name, "reason": "no detection above threshold"})
            continue

        try:
            crop = rotated_crop(img, det.polygon, pad_frac=pad_frac)
        except Exception as e:
            stats["failed"] += 1
            stats["skipped_files"].append({"file": img_path.name, "reason": f"crop error: {e}"})
            continue

        out_path = target_dir / (img_path.stem + ".jpg")
        ok = cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            stats["failed"] += 1
            stats["skipped_files"].append({"file": img_path.name, "reason": "write failed"})
            continue

        if i in val_indices:
            stats["val_saved"] += 1
        else:
            stats["train_saved"] += 1

        # Progress hint every ~25 images so the user knows it's alive.
        if (i + 1) % 25 == 0 or (i + 1) == len(images):
            print(f"  [{class_name}] {i + 1}/{len(images)}", flush=True)

    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Crop cassettes from labeled images using the existing detector.")
    parser.add_argument("model", type=Path, help="Path to YOLOv8-OBB detector .pt file")
    parser.add_argument("input_root", type=Path,
                        help="Folder containing positive/, negative/, invalid/ subfolders")
    parser.add_argument("output_root", type=Path,
                        help="Output folder. Will create train/ and val/ inside it.")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of each class to put in val/ (default 0.2)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Detector confidence threshold (default 0.4)")
    parser.add_argument("--pad", type=float, default=0.04,
                        help="Fraction of long edge to pad around the cassette (default 0.04)")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}", file=sys.stderr)
        sys.exit(2)
    if not args.input_root.exists():
        print(f"Input root not found: {args.input_root}", file=sys.stderr)
        sys.exit(2)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. Run: pip install ultralytics opencv-python pillow", file=sys.stderr)
        sys.exit(2)

    print(f"Loading detector from {args.model} ...", flush=True)
    model = YOLO(str(args.model))
    print("Model loaded.\n", flush=True)

    all_stats = []
    for class_name in CLASSES:
        class_dir = args.input_root / class_name
        if not class_dir.exists():
            print(f"WARNING: missing class folder, skipping: {class_dir}", flush=True)
            continue
        print(f"Processing class '{class_name}' from {class_dir} ...", flush=True)
        out_train = args.output_root / "train" / class_name
        out_val = args.output_root / "val" / class_name
        stats = process_class(
            class_name=class_name,
            input_dir=class_dir,
            output_train_dir=out_train,
            output_val_dir=out_val,
            model=model,
            val_split=args.val_split,
            conf_threshold=args.conf,
            pad_frac=args.pad,
        )
        all_stats.append(stats)
        print(
            f"  -> {stats['train_saved']} train, {stats['val_saved']} val, "
            f"{stats['no_detection']} no-detection, {stats['failed']} failed\n",
            flush=True,
        )

    # Write a summary JSON for the record.
    summary_path = args.output_root / "crop_summary.json"
    args.output_root.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump({"classes": all_stats, "args": {
            "val_split": args.val_split,
            "conf": args.conf,
            "pad": args.pad,
            "seed": SEED,
        }}, f, indent=2)

    print("=" * 60)
    print(f"Done. Summary written to: {summary_path}")
    print("Class totals:")
    for s in all_stats:
        print(f"  {s['class']:>8}: {s['train_saved']:>3} train | {s['val_saved']:>3} val | "
              f"{s['no_detection']:>3} skipped (no detection) | {s['failed']:>3} failed")
    print()
    print("Next step: train the classifier with")
    print(f"  yolo classify train data={args.output_root.resolve()} model=yolov8n-cls.pt epochs=50 imgsz=224")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
