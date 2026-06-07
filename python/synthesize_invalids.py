"""
synthesize_invalids.py — Generate synthetic "invalid" class samples by removing
the C line from negative crops via inpainting.

WHY
---
Phase 2 of this project trains a 3-class classifier (positive / negative / invalid)
on cropped cassette images. After running crop_cassettes.py we typically have:
    positive: ~300 examples
    negative: ~140 examples
    invalid:  ~20  examples   <-- severe class imbalance

A classifier trained on 20 invalid examples will perform poorly on that class no
matter what we do at training time. The fix is to generate more invalid examples
from data we already have.

APPROACH (conservative)
-----------------------
"Invalid" by the project rules means: NO C LINE. The simplest, most reliable
way to generate guaranteed-invalid samples is to take negative crops (which
have exactly one line, the C line) and remove that line. Result: a blank
cassette = invalid.

We deliberately do NOT synthesize T-only invalids from positive crops, even
though they're valid by the rules. Reason: in upright crops we cannot reliably
tell which red line is C and which is T (cassette flip orientation is essentially
random). Removing the wrong line would create a mislabeled training example,
which is more damaging than having one fewer example. We accept the trade-off
of fewer synthetic samples in exchange for zero label noise.

PIPELINE
--------
For each source negative crop:
    1. Detect red pixels in HSV space (the line is reddish)
    2. Find connected components and pick the most line-shaped one
       (high aspect ratio, roughly centered horizontally)
    3. Dilate the mask slightly so inpainting blends across the line edges
    4. Run cv2.inpaint with TELEA to fill in cassette texture
    5. Save the result and a side-by-side preview for human audit

Quality gates: if no line-shaped red region is found, we SKIP that source
rather than synthesize garbage. Better to produce 60 clean samples than
80 samples where 20 look broken.

OUTPUT LAYOUT
-------------
<dataset_root>/synthetic_invalid/
    crops/      *.jpg       <-- synthetic invalid images
    previews/   *.jpg       <-- side-by-side previews for visual audit
    log.json                <-- per-image stats (kept / skipped + reason)

After running, audit the previews/ folder, delete any crops/<name>.jpg that
look wrong, then copy the survivors into <dataset_root>/train/invalid/.

USAGE
-----
    python synthesize_invalids.py <dataset_root> [--target-count 80] [--seed 42] [--debug]

Where <dataset_root> is the folder that contains train/negative/ etc.

With --debug, the script also writes diagnostic images for SKIPPED negatives
into <dataset_root>/synthetic_invalid/debug/, showing what red pixels were
(or weren't) detected. Useful for tuning the thresholds against real data.

NOTES
-----
* This script handles non-ASCII filenames (including Hebrew, e.g. names like
  'צילום מסך 2025-...'). cv2.imread on Windows can't open them; we go through
  numpy.fromfile + cv2.imdecode instead.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from pathlib import Path
from typing import Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Unicode-safe image I/O (cv2.imread on Windows fails on non-ASCII paths)
# ---------------------------------------------------------------------------

def imread_unicode(path: Path):
    """Read an image regardless of unicode characters in the path."""
    import cv2
    import numpy as np
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
        if raw.size == 0:
            return None
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: Path, img, jpeg_quality: int = 95) -> bool:
    """Write a JPEG regardless of unicode characters in the path."""
    import cv2
    import numpy as np
    ext = path.suffix.lower() or ".jpg"
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        return False
    try:
        buf.tofile(str(path))
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Red line detection
# ---------------------------------------------------------------------------

def detect_red_line_mask(img_bgr, return_debug: bool = False):
    """
    Find the C line in an upright cassette crop and return a binary mask of it.

    Returns None if no line-shaped red region is found (caller should skip).
    If return_debug=True, returns a tuple (line_mask_or_None, red_mask_in_band,
    rejected_components_list) for diagnostic visualization.
    """
    import cv2
    import numpy as np

    h, w = img_bgr.shape[:2]

    # 1. Threshold for red in HSV. Red wraps around the hue circle so we need
    #    two ranges (low end ~0deg and high end ~180deg in OpenCV's 0-179 hue).
    # Tuned for REAL cassette photos: lines are often faded pink/rose, not pure red,
    # so we use a generous saturation floor (35) and slightly wider hue ranges.
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0,   35, 50], dtype=np.uint8)
    upper1 = np.array([15, 255, 255], dtype=np.uint8)
    lower2 = np.array([160, 35, 50], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)
    red_mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    # 2. The cassette body occupies the central column of the crop. Anything
    #    red near the very edges is more likely background/finger than a line.
    #    Restrict the search to the central ~75% horizontal band (relaxed from 60%).
    side_margin = int(w * 0.12)
    band_mask = np.zeros_like(red_mask)
    band_mask[:, side_margin: w - side_margin] = 255
    red_mask = cv2.bitwise_and(red_mask, band_mask)

    # 3. Small open to remove specks. We deliberately SKIP the close step here;
    #    closing fattens narrow lines into near-squares which then fail the
    #    aspect-ratio check downstream. Most line breaks are tiny enough that
    #    the dilate step at the end will handle them.
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_small)

    # 4. Find connected components. We want the one that looks like a horizontal
    #    line: wider than tall, reasonably long, not too small.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(red_mask, connectivity=8)

    best_label = -1
    best_score = -1.0
    rejected = []   # for debug
    for lbl in range(1, n_labels):
        x, y, cw, ch, area = stats[lbl]
        if area < 20:
            rejected.append((x, y, cw, ch, "too small"))
            continue
        aspect = cw / max(1, ch)
        width_frac = cw / w
        height_frac = ch / h
        # Relaxed further: accept narrower lines (12% of crop width) and chunkier
        # ones (height up to 20%). Aspect ratio threshold lowered to 1.2 because
        # on cropped cassettes with real-world line thickness, many genuine lines
        # have aspect 1.2-1.5 which the old 1.5 threshold rejected.
        if width_frac < 0.12:
            rejected.append((x, y, cw, ch, f"too narrow ({width_frac:.2f})"))
            continue
        if height_frac > 0.20:
            rejected.append((x, y, cw, ch, f"too tall ({height_frac:.2f})"))
            continue
        if aspect < 1.2:
            rejected.append((x, y, cw, ch, f"not line-shaped (aspect {aspect:.2f})"))
            continue
        # Score: prefer wider + flatter regions. Centeredness is a tiebreaker.
        cx = x + cw / 2
        center_pen = abs(cx - w / 2) / w
        score = width_frac * aspect - center_pen
        if score > best_score:
            best_score = score
            best_label = lbl

    if best_label < 0:
        if return_debug:
            return None, red_mask, rejected
        return None

    # 5. Build a clean mask of just the chosen component, dilated a bit so
    #    the inpaint covers the line's anti-aliased edges and any faint halo.
    line_mask = np.where(labels == best_label, 255, 0).astype(np.uint8)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    line_mask = cv2.dilate(line_mask, dilate_kernel, iterations=2)

    if return_debug:
        return line_mask, red_mask, rejected
    return line_mask


def remove_line(img_bgr, line_mask):
    """Inpaint the masked region using surrounding texture."""
    import cv2
    return cv2.inpaint(img_bgr, line_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


# ---------------------------------------------------------------------------
# Preview rendering
# ---------------------------------------------------------------------------

def make_preview(original, synthetic, line_mask):
    """Build a 3-panel preview: [original | mask overlay | synthetic]."""
    import cv2
    import numpy as np

    h, w = original.shape[:2]

    # Middle panel: original with the detected mask tinted green.
    overlay = original.copy()
    overlay[line_mask > 0] = (0, 255, 0)
    mid = cv2.addWeighted(original, 0.5, overlay, 0.5, 0)

    # Stack horizontally with thin separators.
    sep = np.full((h, 6, 3), 240, dtype=np.uint8)
    panel = np.hstack([original, sep, mid, sep, synthetic])
    return panel


def make_debug_image(original, red_mask, rejected_components):
    """Build a diagnostic image showing what red pixels were detected and
    what components were rejected and why."""
    import cv2
    import numpy as np

    h, w = original.shape[:2]

    # Panel 1: original
    p1 = original.copy()

    # Panel 2: red_mask overlaid in red on grayscale background (to see detection)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    p2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    p2[red_mask > 0] = (0, 0, 255)   # detected red pixels shown as bright red

    # Panel 3: original with rejected components annotated with their rejection reason
    p3 = original.copy()
    for x, y, cw, ch, reason in rejected_components[:6]:   # cap at 6 to avoid clutter
        cv2.rectangle(p3, (x, y), (x + cw, y + ch), (0, 165, 255), 1)   # orange
        # Tiny label
        label = reason.split("(")[0].strip()[:14]
        cv2.putText(p3, label, (x, max(8, y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 165, 255), 1, cv2.LINE_AA)

    sep = np.full((h, 6, 3), 240, dtype=np.uint8)
    return np.hstack([p1, sep, p2, sep, p3])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic 'invalid' samples by removing C lines from negative crops.")
    parser.add_argument("dataset_root", type=Path,
                        help="Folder containing train/negative/ (the classifier dataset root)")
    parser.add_argument("--target-count", type=int, default=80,
                        help="How many synthetic invalids to attempt to generate (default 80)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for source-image shuffling (default 42)")
    parser.add_argument("--debug", action="store_true",
                        help="If set, save diagnostic images for SKIPPED negatives into synthetic_invalid/debug/")
    parser.add_argument("--only-skipped", action="store_true",
                        help="If set, only process sources that appear in the existing log.json as 'skipped'. "
                             "Use this after relaxing thresholds to recover previously-skipped images "
                             "without overwriting any existing crops/previews.")
    parser.add_argument("--output-prefix", type=str, default="synthetic",
                        help="Filename prefix for generated outputs (default 'synthetic'). "
                             "Use a different prefix for a second pass to avoid collisions, e.g. 'synthetic2'.")
    args = parser.parse_args()

    try:
        import cv2
        import numpy as np
    except ImportError as e:
        print(f"Missing dependency: {e}. Run: pip install opencv-python", file=sys.stderr)
        sys.exit(2)

    src_dir = args.dataset_root / "train" / "negative"
    if not src_dir.exists():
        print(f"Source folder not found: {src_dir}", file=sys.stderr)
        print("Make sure crop_cassettes.py has been run first.", file=sys.stderr)
        sys.exit(2)

    out_dir = args.dataset_root / "synthetic_invalid"
    out_crops = out_dir / "crops"
    out_previews = out_dir / "previews"
    out_debug = out_dir / "debug"
    out_crops.mkdir(parents=True, exist_ok=True)
    out_previews.mkdir(parents=True, exist_ok=True)
    if args.debug:
        out_debug.mkdir(parents=True, exist_ok=True)

    # Gather and shuffle source images.
    sources = sorted([p for p in src_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if not sources:
        print(f"No images found in {src_dir}", file=sys.stderr)
        sys.exit(2)

    # --only-skipped: restrict to sources that were previously skipped.
    # Uses the existing log.json. The first run produced both a 'generated' list
    # and a 'skipped' list; if we just relaxed thresholds, the things worth
    # retrying are the ones that were previously skipped (not the ones that
    # already succeeded, since we don't want to overwrite or duplicate them).
    if args.only_skipped:
        existing_log_path = out_dir / "log.json"
        if not existing_log_path.exists():
            print(f"--only-skipped requires {existing_log_path} from a previous run. "
                  f"Run without --only-skipped first.", file=sys.stderr)
            sys.exit(2)
        with existing_log_path.open("r", encoding="utf-8") as f:
            prev_log = json.load(f)
        skipped_names = {entry["file"] for entry in prev_log.get("skipped", [])}
        before = len(sources)
        sources = [p for p in sources if p.name in skipped_names]
        print(f"--only-skipped: filtered {before} sources down to {len(sources)} "
              f"that were previously skipped.")
        if not sources:
            print("Nothing to retry — all previously-skipped sources are gone "
                  "(maybe the negative folder changed).", file=sys.stderr)
            sys.exit(0)

    rng = random.Random(args.seed)
    rng.shuffle(sources)

    print(f"Source pool: {len(sources)} negative crops in {src_dir}")
    print(f"Target output: up to {args.target_count} synthetic invalids")
    print(f"Output folder: {out_dir}")
    if args.debug:
        print(f"Debug images for skipped sources: {out_debug}")
    print()

    log = {"generated": [], "skipped": [], "args": {
        "target_count": args.target_count,
        "seed": args.seed,
        "debug": args.debug,
    }}

    generated = 0
    attempted = 0
    skipped_safe_name_idx = 0
    for src_path in sources:
        if generated >= args.target_count:
            break
        attempted += 1

        img = imread_unicode(src_path)
        if img is None:
            log["skipped"].append({"file": src_path.name, "reason": "could not read image"})
            continue

        if args.debug:
            line_mask, red_mask, rejected = detect_red_line_mask(img, return_debug=True)
        else:
            line_mask = detect_red_line_mask(img)
            red_mask = None
            rejected = None

        if line_mask is None or line_mask.sum() == 0:
            log["skipped"].append({"file": src_path.name, "reason": "no line-shaped red region found"})
            # In debug mode, save a diagnostic image for this skipped source.
            if args.debug:
                skipped_safe_name_idx += 1
                # Use an ASCII-safe name for the debug file to keep things browsable,
                # but record the original filename in the log.
                dbg_name = f"skipped_{skipped_safe_name_idx:03d}.jpg"
                dbg = make_debug_image(img, red_mask, rejected or [])
                imwrite_unicode(out_debug / dbg_name, dbg, jpeg_quality=85)
                log["skipped"][-1]["debug_image"] = dbg_name
            continue

        synthetic = remove_line(img, line_mask)

        # ASCII-safe output filename (the source may have Hebrew/unicode chars).
        # Take the stem, replace non-alphanum with underscore, cap at 40 chars,
        # and prefix with a generated index so duplicates can't collide.
        safe_stem = "".join(c if (c.isalnum() or c in "_-") else "_" for c in src_path.stem)[:40]
        safe_name = f"{args.output_prefix}_{generated + 1:03d}_{safe_stem}.jpg"
        crop_path = out_crops / safe_name
        preview_path = out_previews / safe_name

        imwrite_unicode(crop_path, synthetic, jpeg_quality=95)
        preview = make_preview(img, synthetic, line_mask)
        imwrite_unicode(preview_path, preview, jpeg_quality=90)

        generated += 1
        log["generated"].append({"source": src_path.name, "output": safe_name})

        if generated % 20 == 0 or generated == args.target_count:
            print(f"  Generated {generated}/{args.target_count} (attempted {attempted})", flush=True)

    log_name = f"log_{args.output_prefix}.json" if args.only_skipped else "log.json"
    log_path = out_dir / log_name
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Done. Generated {generated} synthetic invalids "
          f"(attempted {attempted}, skipped {len(log['skipped'])}).")
    print(f"Crops:    {out_crops}")
    print(f"Previews: {out_previews}")
    if args.debug:
        print(f"Debug:    {out_debug}")
    print(f"Log:      {log_path}")
    print()
    print("Next steps:")
    print(f"  1. Open {out_previews} and flip through. Each preview shows:")
    print("       [original]  |  [original with detected line in green]  |  [synthetic]")
    print("  2. Delete any crops/<name>.jpg whose preview looks wrong.")
    print("     (Look for: line not fully removed, wrong region inpainted, ugly artifacts.)")
    print(f"  3. Copy the survivors from crops/ into {args.dataset_root / 'train' / 'invalid'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
