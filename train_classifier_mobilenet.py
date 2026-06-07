# train_classifier_mobilenet.py
# Step B (v2) — Transfer-learning classifier for LFT positive/negative/invalid.
#
# Why this replaces the YOLOv8-cls version:
#   The YOLOv8n-cls model overfit the tiny dataset (perfect on training crops,
#   wrong on new photos). A pretrained MobileNetV3-Small backbone already knows
#   edges/shapes from ImageNet, so it generalizes from few examples far better.
#   We also use REAL class weighting in the loss instead of duplicating files.
#
# Run cell-by-cell (# %% markers) in PyCharm/IntelliJ, or top-to-bottom:
#   python train_classifier_mobilenet.py
#
# Output: python/classifier_mnv3.pt  (a dict: {state_dict, classes, arch, img_size})

# %% [Imports & config]
import json
import os
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ---- Paths (relative by default; override DATA_ROOT via environment variable) ----
DATA_ROOT = Path(os.environ.get("CLASSIFIER_DATASET", "classifier_dataset"))
OUT_MODEL = Path("python") / "classifier_mnv3.pt"

# ---- Hyperparameters ----
IMG_SIZE = 224
BATCH = 32
EPOCHS_HEAD = 25        # train only the new head first
EPOCHS_FINETUNE = 15    # then unfreeze the last block at a low LR
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
SEED = 42
CLASSES = ["positive", "negative", "invalid"]   # fixed order

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


# %% [Transforms]
# ImageNet normalization (MobileNetV3 was pretrained with these stats).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Train augmentation: rotation + horizontal flip + color jitter + slight crop.
# NOTE: NO vertical flip — flipping top/bottom would scramble C/T line order.
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 24, IMG_SIZE + 24)),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(0.6, 1.4)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# %% [Datasets & loaders]
# ImageFolder assigns class indices alphabetically. Force OUR fixed order so the
# saved model's class index mapping is predictable (positive=0, negative=1, invalid=2).
class FixedOrderImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        present = [c for c in CLASSES if (Path(directory) / c).is_dir()]
        if not present:
            present = sorted(d.name for d in Path(directory).iterdir() if d.is_dir())
        return present, {c: i for i, c in enumerate(present)}

train_ds = FixedOrderImageFolder(str(DATA_ROOT / "train"), transform=train_tf)
val_ds = FixedOrderImageFolder(str(DATA_ROOT / "val"), transform=eval_tf)
print("Class -> index:", train_ds.class_to_idx)

train_counts = Counter([lbl for _, lbl in train_ds.samples])
print("Train counts:", {train_ds.classes[i]: train_counts[i] for i in range(len(train_ds.classes))})
val_counts = Counter([lbl for _, lbl in val_ds.samples])
print("Val counts:  ", {val_ds.classes[i]: val_counts[i] for i in range(len(val_ds.classes))})

# Weighted sampler: draw minority classes more often per epoch (real oversampling
# at the batch level, not file duplication).
class_sample_weight = {c: 1.0 / max(1, train_counts[c]) for c in train_counts}
sample_weights = [class_sample_weight[lbl] for _, lbl in train_ds.samples]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

# Class weights for the loss (inverse frequency, normalized).
n_classes = len(train_ds.classes)
freq = torch.tensor([train_counts[i] for i in range(n_classes)], dtype=torch.float)
loss_weights = (freq.sum() / (n_classes * freq)).to(DEVICE)
print("Loss class weights:", loss_weights.cpu().numpy().round(3))


# %% [Build model: MobileNetV3-Small, pretrained]
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

# Freeze the whole feature extractor initially.
for p in model.parameters():
    p.requires_grad = False

# Replace the classifier head with a fresh 3-class head (trainable).
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, n_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=loss_weights)


def run_epoch(loader, train: bool, optimizer=None):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    torch.set_grad_enabled(train)
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        if train:
            optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        if train:
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return loss_sum / total, correct / total


# %% [Phase 1: train the head]
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=LR_HEAD)

best_val_acc = 0.0
best_state = None
for epoch in range(EPOCHS_HEAD):
    tl, ta = run_epoch(train_loader, True, optimizer)
    vl, va = run_epoch(val_loader, False)
    history["train_loss"].append(tl); history["val_loss"].append(vl)
    history["train_acc"].append(ta); history["val_acc"].append(va)
    if va >= best_val_acc:
        best_val_acc = va
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f"[head {epoch+1:02d}/{EPOCHS_HEAD}] train_loss={tl:.3f} acc={ta:.3f} | val_loss={vl:.3f} acc={va:.3f}")


# %% [Phase 2: fine-tune the last feature block]
# Unfreeze the final inverted-residual block + classifier for gentle fine-tuning.
for name, p in model.named_parameters():
    if name.startswith("features.12") or name.startswith("classifier"):
        p.requires_grad = True

optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=LR_FINETUNE)

for epoch in range(EPOCHS_FINETUNE):
    tl, ta = run_epoch(train_loader, True, optimizer)
    vl, va = run_epoch(val_loader, False)
    history["train_loss"].append(tl); history["val_loss"].append(vl)
    history["train_acc"].append(ta); history["val_acc"].append(va)
    if va >= best_val_acc:
        best_val_acc = va
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f"[ft  {epoch+1:02d}/{EPOCHS_FINETUNE}] train_loss={tl:.3f} acc={ta:.3f} | val_loss={vl:.3f} acc={va:.3f}")

# Restore best weights.
if best_state is not None:
    model.load_state_dict(best_state)
print(f"Best val accuracy: {best_val_acc:.3f}")


# %% [Curves]
ep = range(1, len(history["train_loss"]) + 1)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
a1.plot(ep, history["train_loss"], label="train loss")
a1.plot(ep, history["val_loss"], label="val loss")
a1.axvline(EPOCHS_HEAD + 0.5, ls="--", c="gray", lw=1, label="start fine-tune")
a1.set_xlabel("epoch"); a1.set_ylabel("loss"); a1.set_title("Loss"); a1.legend()
a2.plot(ep, history["train_acc"], label="train acc")
a2.plot(ep, history["val_acc"], label="val acc")
a2.axvline(EPOCHS_HEAD + 0.5, ls="--", c="gray", lw=1)
a2.set_xlabel("epoch"); a2.set_ylabel("accuracy"); a2.set_title("Accuracy"); a2.legend()
plt.tight_layout(); plt.show()


# %% [Confusion matrix + report on val]
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        out = model(imgs.to(DEVICE))
        y_pred.extend(out.argmax(1).cpu().tolist())
        y_true.extend(labels.tolist())

idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
labels_order = list(range(n_classes))
names_order = [idx_to_class[i] for i in labels_order]
cm = confusion_matrix(y_true, y_pred, labels=labels_order)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=names_order, yticklabels=names_order)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (val)")
plt.tight_layout(); plt.show()
print(classification_report(y_true, y_pred, labels=labels_order,
                            target_names=names_order, digits=3, zero_division=0))


# %% [Save model in a UI-loadable format]
OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
torch.save({
    "state_dict": model.state_dict(),
    "classes": [idx_to_class[i] for i in range(n_classes)],
    "arch": "mobilenet_v3_small",
    "img_size": IMG_SIZE,
    "mean": IMAGENET_MEAN,
    "std": IMAGENET_STD,
}, str(OUT_MODEL))
print("Saved model to:", OUT_MODEL)
print("Class order:", [idx_to_class[i] for i in range(n_classes)])


# %% [OPTIONAL: test generalization on a folder of NEW photos]
# This is the test that actually matters — images the model never trained on.
# Point it at a folder of full photos (with backgrounds). It runs the full
# pipeline: detect -> rotated crop -> classify, exactly like the UI.
#
# Set this to a folder of brand-new test images (override via the NEW_PHOTOS_DIR
# environment variable), then run this cell.
NEW_PHOTOS_DIR = Path(os.environ.get("NEW_PHOTOS_DIR", "new_photos"))
DETECTOR_PT = Path("best.pt")
MAX_TO_SHOW = 9

if NEW_PHOTOS_DIR.exists():
    import sys, cv2
    sys.path.insert(0, str((OUT_MODEL.parent).resolve()))  # python/ folder
    import detect as dt           # reuse rotated_crop
    from ultralytics import YOLO

    det = YOLO(str(DETECTOR_PT))
    classes_saved = [idx_to_class[i] for i in range(n_classes)]

    def best_poly(img_bgr):
        res = det.predict(source=img_bgr, imgsz=640, conf=0.4, iou=0.5, verbose=False)
        best = None
        for r in res:
            obb = getattr(r, "obb", None)
            if obb is None or obb.xyxyxyxy is None:
                continue
            polys = obb.xyxyxyxy.cpu().numpy(); confs = obb.conf.cpu().numpy()
            for p, c in zip(polys, confs):
                if best is None or float(c) > best[0]:
                    best = (float(c), p)
        return None if best is None else best[1]

    from PIL import Image
    files = [f for f in sorted(NEW_PHOTOS_DIR.iterdir())
             if f.suffix.lower() in {".jpg", ".jpeg", ".png"}][:MAX_TO_SHOW]
    cols = 3; rows = (len(files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, f in zip(axes, files):
        img = cv2.imread(str(f))
        poly = best_poly(img)
        if poly is None:
            ax.set_title("no detection"); ax.axis("off"); continue
        crop = dt.rotated_crop(img, poly, pad_frac=0.04)
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        x = eval_tf(pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.softmax(model(x), 1)[0].cpu().numpy()
        top = int(prob.argmax())
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{classes_saved[top]} ({prob[top]:.2f})", fontsize=10)
        ax.axis("off")
    for ax in axes[len(files):]:
        ax.axis("off")
    plt.tight_layout(); plt.show()
else:
    print("NEW_PHOTOS_DIR not found — set it to a folder of test 