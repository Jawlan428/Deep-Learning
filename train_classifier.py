# train_classifier.py
# Step B — YOLOv8-cls classifier for LFT positive/negative/invalid
# Run cells with # %% markers in IntelliJ (Python Scientific mode) or top-to-bottom.

# %% [Imports & config]
import os
import shutil
import random
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from ultralytics import YOLO

# ---- Paths (relative by default; override via environment variables) ----
DATA_ROOT = Path(os.environ.get("CLASSIFIER_DATASET", "classifier_dataset"))
BALANCED_ROOT = Path(os.environ.get("CLASSIFIER_DATASET_BALANCED", "classifier_dataset_balanced"))
OUT_DIR = Path(os.environ.get("CLASSIFIER_RUNS", "classifier_runs"))

IMG_SIZE = 224
EPOCHS = 80
BATCH = 32
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 42
CLASSES = ["positive", "negative", "invalid"]

random.seed(SEED)
np.random.seed(SEED)
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


# %% [Inspect dataset: counts + sample viz]
def count_split(root):
    counts = {}
    for split in ["train", "val"]:
        counts[split] = {}
        for cls in CLASSES:
            d = root / split / cls
            n = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png"))) if d.exists() else 0
            counts[split][cls] = n
    return counts

orig_counts = count_split(DATA_ROOT)
print("Original counts:")
for split, c in orig_counts.items():
    print(f"  {split}: {c}  (total {sum(c.values())})")

# Show one sample per class from train
fig, axes = plt.subplots(1, len(CLASSES), figsize=(12, 4))
for ax, cls in zip(axes, CLASSES):
    files = list((DATA_ROOT / "train" / cls).glob("*.jpg")) + \
            list((DATA_ROOT / "train" / cls).glob("*.png"))
    if files:
        img = cv2.cvtColor(cv2.imread(str(files[0])), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
    ax.set_title(f"{cls} (n={orig_counts['train'][cls]})")
    ax.axis("off")
plt.tight_layout()
plt.show()


# %% [Build balanced training set via oversampling]
# YOLOv8-cls has no class_weights arg, so we physically oversample minority
# classes in TRAIN only. Val is copied as-is (never oversample val).
def build_balanced(src_root, dst_root):
    if dst_root.exists():
        shutil.rmtree(dst_root)

    # Copy val unchanged
    for cls in CLASSES:
        src = src_root / "val" / cls
        dst = dst_root / "val" / cls
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*.*"):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                shutil.copy(f, dst / f.name)

    # Oversample train up to the max class size
    train_files = {}
    for cls in CLASSES:
        train_files[cls] = [f for f in (src_root / "train" / cls).glob("*.*")
                            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    target = max(len(v) for v in train_files.values())

    for cls in CLASSES:
        dst = dst_root / "train" / cls
        dst.mkdir(parents=True, exist_ok=True)
        files = train_files[cls]
        # Copy originals
        for f in files:
            shutil.copy(f, dst / f.name)
        # Duplicate (with index suffix) until we reach target
        i = 0
        while len(list(dst.glob("*.*"))) < target:
            src_f = files[i % len(files)]
            new_name = f"{src_f.stem}_dup{i}{src_f.suffix}"
            shutil.copy(src_f, dst / new_name)
            i += 1
    return target

target = build_balanced(DATA_ROOT, BALANCED_ROOT)
bal_counts = count_split(BALANCED_ROOT)
print(f"Balanced each train class to ~{target} images")
for split, c in bal_counts.items():
    print(f"  {split}: {c}")


# %% [Train YOLOv8-cls with heavy augmentation]
model = YOLO("yolov8n-cls.pt")  # nano classification variant

results = model.train(
    data=str(BALANCED_ROOT),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    device=DEVICE,
    workers=0,
    project=str(OUT_DIR),
    name="lft_cls",
    seed=SEED,
    patience=20,
    # ---- Augmentation (heavy, to compensate for small dataset) ----
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=15.0,        # cassettes can be slightly rotated post-crop
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    flipud=0.0,          # don't flip vertically — line order matters
    erasing=0.4,
    auto_augment="randaugment",
)
print("Training done. Best weights:", Path(results.save_dir) / "weights" / "best.pt")


# %% [Accuracy & loss curves]
# Ultralytics logs metrics to results.csv in the run dir.
import pandas as pd
run_dir = Path(results.save_dir)
csv_path = run_dir / "results.csv"
df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
# Loss
if "train/loss" in df.columns:
    ax1.plot(df["epoch"], df["train/loss"], label="train loss")
if "val/loss" in df.columns:
    ax1.plot(df["epoch"], df["val/loss"], label="val loss")
ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.set_title("Loss"); ax1.legend()
# Accuracy
acc_cols = [c for c in df.columns if "accuracy" in c.lower() or "top1" in c.lower()]
for c in acc_cols:
    ax2.plot(df["epoch"], df[c], label=c)
ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.set_title("Accuracy"); ax2.legend()
plt.tight_layout()
plt.show()


# %% [Confusion matrix on val set]
best = YOLO(str(run_dir / "weights" / "best.pt"))

y_true, y_pred = [], []
class_to_idx = {c: i for i, c in enumerate(CLASSES)}
for cls in CLASSES:
    for f in (DATA_ROOT / "val" / cls).glob("*.*"):
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        r = best.predict(str(f), imgsz=IMG_SIZE, device=DEVICE, verbose=False)[0]
        pred_name = r.names[int(r.probs.top1)]
        y_true.append(class_to_idx[cls])
        y_pred.append(class_to_idx[pred_name])

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (val)")
plt.tight_layout()
plt.show()

print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))


# %% [Sample predictions on val images]
val_samples = []
for cls in CLASSES:
    fs = [f for f in (DATA_ROOT / "val" / cls).glob("*.*")
          if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    val_samples += [(f, cls) for f in fs[:3]]

cols = 3
rows = (len(val_samples) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(13, 4 * rows))
axes = np.array(axes).reshape(-1)
for ax, (f, true_cls) in zip(axes, val_samples):
    r = best.predict(str(f), imgsz=IMG_SIZE, device=DEVICE, verbose=False)[0]
    pred = r.names[int(r.probs.top1)]
    conf = float(r.probs.top1conf)
    img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ok = "OK" if pred == true_cls else "WRONG"
    color = "green" if pred == true_cls else "red"
    ax.set_title(f"[{ok}] true={true_cls}\npred={pred} ({conf:.2f})", color=color, fontsize=10)
    ax.axis("off")
for ax in axes[len(val_samples):]:
    ax.axis("off")
plt.tight_layout()
plt.show()


# %% [Save classifier.pt for JavaFX integration]
final_path = Path("python") / "classifier.pt"
final_path.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(run_dir / "weights" / "best.pt", final_path)
