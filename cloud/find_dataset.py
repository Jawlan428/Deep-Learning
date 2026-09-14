"""
find_dataset.py — locate the classifier dataset on this machine.

WHAT IT LOOKS FOR
    Not a folder called "classifier_dataset" — the name may have changed.
    It looks for the SHAPE of the thing: any directory containing at least two
    of the subfolders positive/ negative/ invalid/ with images inside them.
    Searching by structure instead of by name survives renaming, copying and
    the folder living somewhere you forgot about.

    It also finds the OBB dataset (data.yaml + images/ + labels/) in case we
    later want to re-validate the detector.

RUN
    ..\\.venv\\Scripts\\python.exe cloud\\find_dataset.py

    Search somewhere specific (e.g. another drive):
    ..\\.venv\\Scripts\\python.exe cloud\\find_dataset.py --roots D:\\ E:\\
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {"positive", "negative", "invalid"}

# Directories that are large, irrelevant, and slow to walk. Skipping them is
# the difference between a 20-second search and a 10-minute one.
SKIP = {
    "windows", "$recycle.bin", "system volume information", "program files",
    "program files (x86)", "programdata", "appdata", "node_modules",
    ".git", "__pycache__", ".venv", "venv", "env", "site-packages",
    "onedrivetemp", "temp", "tmp", ".cache", ".gradle", ".m2", ".nuget",
}

MAX_DEPTH = 7


def count_images(d: Path, cap: int = 500) -> int:
    n = 0
    try:
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in IMG_EXT:
                n += 1
                if n >= cap:
                    break
    except (PermissionError, OSError):
        pass
    return n


def walk(root: Path):
    """Depth-limited walk that prunes noisy directories as it goes."""
    root = root.resolve()
    base_depth = len(root.parts)
    for dirpath, dirnames, _ in os.walk(root, topdown=True, onerror=lambda e: None):
        p = Path(dirpath)
        if len(p.parts) - base_depth >= MAX_DEPTH:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d.lower() not in SKIP and not d.startswith("$")]
        yield p, dirnames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=None)
    args = ap.parse_args()

    if args.roots:
        roots = [Path(r) for r in args.roots]
    else:
        home = Path.home()
        roots = [home / "Downloads", home / "Desktop", home / "Documents",
                 home / "Pictures", home / "OneDrive", Path("C:/")]
        roots = [r for r in roots if r.exists()]

    print("Searching (this can take a minute):")
    for r in roots:
        print(f"  {r}")
    print()

    seen: set[Path] = set()
    classifier_hits = []
    obb_hits = []

    for root in roots:
        for p, dirnames in walk(root):
            if p in seen:
                continue
            seen.add(p)

            lower = {d.lower(): d for d in dirnames}

            # --- classifier dataset: positive/ negative/ invalid/ ---
            present = CLASS_NAMES & set(lower)
            if len(present) >= 2:
                counts = {}
                for c in sorted(present):
                    counts[c] = count_images(p / lower[c])
                if sum(counts.values()) > 0:
                    classifier_hits.append((p, counts))
                    print(f"[classifier] {p}")
                    for c, n in counts.items():
                        print(f"               {c:<10} {n} images")
                    print()

            # --- OBB dataset: data.yaml next to images/ + labels/ ---
            try:
                files = {f.name.lower() for f in p.iterdir() if f.is_file()}
            except (PermissionError, OSError):
                files = set()
            if "data.yaml" in files and {"images", "labels"} <= set(lower):
                obb_hits.append(p)
                print(f"[obb]        {p}  (data.yaml + images/ + labels/)")
                print()

    print("=" * 68)
    if not classifier_hits:
        print("No classifier dataset found in the searched locations.")
        print("If it lives on another drive, re-run with e.g.:")
        print("  ..\\.venv\\Scripts\\python.exe cloud\\find_dataset.py --roots D:\\")
    else:
        print("Classifier dataset candidates:\n")
        for p, counts in classifier_hits:
            total = sum(counts.values())
            print(f"  {p}   ({total} images)")
        print()
        # Pick the VALIDATION split, not merely the smallest folder.
        # (The first version of this script used min(total images), which
        # happily recommended a 15-image demo folder over the real 81-image
        # val split. "Smallest" was a proxy for "validation" and it was wrong.)
        #
        # Prefer, in order:
        #   1. a folder actually named val / valid / validation / test
        #   2. a sibling of a folder named train (same parent = same dataset)
        #   3. failing both, the smallest — but say so out loud
        VAL_NAMES = {"val", "valid", "validation", "test"}
        train_parents = {p.parent for p, _ in classifier_hits if p.name.lower() == "train"}

        named = [h for h in classifier_hits if h[0].name.lower() in VAL_NAMES]
        sibling = [h for h in named if h[0].parent in train_parents]

        if sibling:
            best, why = sibling[0][0], "named as a val split AND sits beside a train/ split"
        elif named:
            best, why = named[0][0], "named as a validation split"
        else:
            best, why = min(classifier_hits, key=lambda x: sum(x[1].values()))[0], \
                "GUESS — smallest candidate; no folder named val/test found, so check this is right"

        print(f"Recommended: {best}")
        print(f"  reason: {why}")
        print("\nRun Part B against it like this:\n")
        print(f'  ..\\.venv\\Scripts\\python.exe cloud\\verify_stage1.py --dataset "{best}"')

    if obb_hits:
        print(f"\nOBB dataset(s) found: {len(obb_hits)} — useful later if we")
        print("re-validate the detector after export.")


if __name__ == "__main__":
    main()
