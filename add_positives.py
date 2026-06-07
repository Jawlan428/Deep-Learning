import argparse, os, random, sys
from pathlib import Path
import cv2
from ultralytics import YOLO

sys.path.insert(0, str((Path(__file__).parent / "python").resolve()))
import detect as dt

# Dataset root: defaults to ./classifier_dataset relative to where you run the
# script. Override with the CLASSIFIER_DATASET environment variable if your data
# lives elsewhere.
CLASSIFIER_DATASET = Path(os.environ.get("CLASSIFIER_DATASET", "classifier_dataset"))
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42

def best_poly(model, img, conf=0.4):
    res = model.predict(source=img, imgsz=640, conf=conf, iou=0.5, verbose=False)
    best = None
    for r in res:
        obb = getattr(r, "obb", None)
        if obb is None or obb.xyxyxyxy is None:
            continue
        for p, c in zip(obb.xyxyxyxy.cpu().numpy(), obb.conf.cpu().numpy()):
            if best is None or float(c) > best[0]:
                best = (float(c), p)
    return None if best is None else best[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("detector", type=Path)
    ap.add_argument("source", type=Path)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--prefix", default="newpos")
    args = ap.parse_args()

    train_dir = CLASSIFIER_DATASET / "train" / "positive"
    val_dir = CLASSIFIER_DATASET / "val" / "positive"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in args.source.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    print(f"Found {len(files)} images")

    rng = random.Random(SEED)
    idx = list(range(len(files))); rng.shuffle(idx)
    n_val = max(1, int(round(len(files) * args.val_split)))
    val_idx = set(idx[:n_val])

    model = YOLO(str(args.detector))
    st = sv = nd = fl = 0
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is None: fl += 1; print(f"  [skip] read {f.name}"); continue
        poly = best_poly(model, img, args.conf)
        if poly is None: nd += 1; print(f"  [skip] no detection {f.name}"); continue
        try:
            crop = dt._to_portrait(dt.rotated_crop(img, poly, pad_frac=0.04))
        except Exception as e:
            fl += 1; print(f"  [skip] crop {f.name}: {e}"); continue
        target = val_dir if i in val_idx else train_dir
        if cv2.imwrite(str(target / f"{args.prefix}_{i:03d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            if i in val_idx: sv += 1
         