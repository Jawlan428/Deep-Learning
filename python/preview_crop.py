"""
preview_crop.py — Sanity-check a single image's crop before running the full batch.

Useful for verifying that the rotated crop logic produces a clean upright cassette
on YOUR images before you spend time generating hundreds of crops.

Usage:
    python preview_crop.py <model_path> <image_path> <output_path>

Output:
    Writes a side-by-side comparison image:
        [original with OBB drawn]  |  [rotated crop]
    so you can eyeball the result. Also prints the confidence to stdout.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Reuse the helpers from crop_cassettes
sys.path.insert(0, str(Path(__file__).parent))
from crop_cassettes import load_image_normalized, best_detection, rotated_crop


def main():
    if len(sys.argv) != 4:
        print("Usage: preview_crop.py <model_path> <image_path> <output_path>")
        sys.exit(2)

    model_path = Path(sys.argv[1])
    image_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])

    import cv2
    import numpy as np
    from ultralytics import YOLO

    print(f"Loading model from {model_path} ...")
    model = YOLO(str(model_path))

    img = load_image_normalized(image_path)
    det = best_detection(model, img, conf_threshold=0.25)
    if det is None:
        print("No detection found.")
        sys.exit(1)

    print(f"Detection confidence: {det.confidence:.3f}")

    # Draw the OBB on the original
    annotated = img.copy()
    pts = np.array(det.polygon, dtype=np.int32).reshape(4, 2)
    cv2.polylines(annotated, [pts], isClosed=True, color=(86, 110, 15),
                  thickness=max(2, int(min(img.shape[:2]) / 400)))

    crop = rotated_crop(img, det.polygon, pad_frac=0.04)

    # Build a side-by-side comparison at matching heights.
    h_target = 600
    def fit(im):
        h, w = im.shape[:2]
        s = h_target / h
        return cv2.resize(im, (max(1, int(w * s)), h_target), interpolation=cv2.INTER_AREA)

    left = fit(annotated)
    right = fit(crop)
    gap = np.full((h_target, 10, 3), 240, dtype=np.uint8)
    panel = np.hstack([left, gap, right])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), panel)
    print(f"Wrote preview to {output_path}")


if __name__ == "__main__":
    main()
