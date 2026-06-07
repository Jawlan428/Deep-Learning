"""
normalize.py — Reads an image, applies EXIF rotation to the pixels themselves,
and writes a clean JPEG with no EXIF metadata. After this, every downstream
consumer (display, detection, drawing) sees the same upright pixels and there
is no orientation metadata to disagree about.

Usage:
    python normalize.py <input_image> <output_image>

Output (stdout, JSON):
    {"ok": true, "output": "<absolute path>", "width": W, "height": H}
On error:
    {"ok": false, "error": "..."}
"""
import json
import sys
from pathlib import Path


def fail(msg: str) -> None:
    print(json.dumps({"ok": False, "error": msg}))
    sys.exit(0)


def main() -> None:
    if len(sys.argv) != 3:
        fail("Usage: normalize.py <input_image> <output_image>")

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    if not src.exists():
        fail(f"Input not found: {src}")

    try:
        from PIL import Image, ImageOps
    except ImportError:
        fail("Pillow not installed. Run: pip install Pillow")

    try:
        with Image.open(str(src)) as im:
            # exif_transpose physically applies EXIF orientation to pixels and
            # strips the tag. The result has no orientation metadata — which is
            # exactly what we want.
            up = ImageOps.exif_transpose(im)
            up = up.convert("RGB")
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Save as JPEG with no EXIF block at all.
            up.save(str(dst), "JPEG", quality=95)
            w, h = up.size
    except Exception as e:
        fail(f"Failed to normalize: {e}")

    print(json.dumps({
        "ok": True,
        "output": str(dst.resolve()).replace("\\", "/"),
        "width": w,
        "height": h,
    }))


if __name__ == "__main__":
    main()