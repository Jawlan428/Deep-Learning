# COVID-19 LFT Detector — JavaFX UI

A clean, clinical-themed JavaFX desktop app that runs YOLOv8-OBB detection on
COVID-19 lateral flow test images. Browse a folder, navigate with prev/next, hit
**Detect** to see the oriented bounding box and a side panel with class,
confidence, rotation, and box size.

## Screenshots

Screenshots live in [`docs/screenshots/`](docs/screenshots). Drop your own PNGs
in that folder and they will render here:

| Main detection view | Two-stage result (positive / negative / invalid) |
|---------------------|--------------------------------------------------|
| ![Detection view](docs/screenshots/detection.png) | ![Result view](docs/screenshots/result.png) |

To capture them: run the app (see **Run** below), open a folder of test images,
press **Detect**, then **Read Result**, and screenshot the window. Save them as
`docs/screenshots/detection.png` and `docs/screenshots/result.png` (or update the
filenames above). If the images are missing, GitHub simply shows a broken-image
icon — nothing else breaks.

## Project layout

```
lft-detector-ui/
├── pom.xml
├── best.pt                          ← put your trained model here
├── python/
│   └── detect.py                    ← inference bridge (runs in subprocess)
└── src/main/
    ├── java/com/lftdetector/
    │   ├── LFTDetectorApp.java      ← main UI
    │   ├── DetectionService.java    ← ProcessBuilder wrapper
    │   └── DetectionResult.java     ← JSON data classes
    └── resources/
        └── app.css                  ← clinical theme
```

## Prerequisites

1. **Java 17+** (the project targets Java 17 — JavaFX 21 modules are pulled in
   by Maven, no separate JavaFX SDK needed).
2. **Maven** on PATH.
3. **Python 3.9+** with the inference dependencies installed (see
   [`requirements.txt`](requirements.txt)):
   ```
   pip install -r requirements.txt
   ```
   This installs `ultralytics` (which pulls in `torch`/`torchvision`),
   `opencv-python`, `Pillow`, and `numpy`.
4. Your trained `best.pt` from Phase 3.5 placed at the project root (or update
   `MODEL_PATH_REL` in `LFTDetectorApp.java`).

## Configuration

The Python interpreter is resolved at startup with **no code edit needed**: the
app uses the `LFT_PYTHON` environment variable if it is set, otherwise it falls
back to `python` on your PATH. If `python` already points at the environment
where you ran `pip install -r requirements.txt`, you don't need to set anything.

To pin a specific interpreter (e.g. a conda/venv), set `LFT_PYTHON` — in
PowerShell:
```powershell
setx LFT_PYTHON "C:/Users/you/anaconda3/envs/yolo/python.exe"
```
(reopen the terminal/IDE after `setx` so the new variable is picked up).

The script and model locations are still simple constants in `LFTDetectorApp.java`
if you ever need to change them:
```java
private static final String DETECT_SCRIPT_REL = "python/detect.py";
private static final String MODEL_PATH_REL    = "best.pt";
```

## Run

> **Run from the project root** (the folder with `pom.xml`). The app resolves
> `python/detect.py`, `best.pt`, etc. relative to the working directory, so it is
> fully self-contained — it reads nothing outside this folder.

**Easiest (IntelliJ):** open this folder, let Maven import, then either run the
`javafx:run` goal from the Maven tool window, or run `LFTDetectorApp` directly.
IntelliJ bundles Maven, so you don't need it on PATH.

**Command line (if Maven is installed):**
```bash
mvn javafx:run
```

Or build a jar:
```bash
mvn clean package
java -jar target/lft-detector-ui-1.0.0.jar
```

### Before the first run — install the Python deps

The Java UI shells out to Python for the actual ML. If the dependencies are not
installed in the interpreter the app calls, **Detect will fail** — the side panel
shows a "Missing dependency" / "Python produced no output" error because
`detect.py` can't `import ultralytics`. Install them once:

```bash
pip install -r requirements.txt
```

Make sure that's the **same** interpreter the app uses — the one on PATH as
`python`, or whatever `LFT_PYTHON` points to (see **Configuration** above).

## Using the app

1. Click **Browse** in the top-right to choose a folder of `.jpg` / `.png` images.
2. Use the circular **←** / **→** buttons (or arrow keys) to flip through images.
3. Click **Detect** (or press **Enter**) to run the model.
   - On success: oriented box drawn over the cassette + side panel shows
     class, confidence, rotation, and box size.
   - If nothing is found: the image stays as-is and the panel shows
     "No POI".
4. Switch folders any time by clicking **Browse** again.

## How detection works

- The Java app spawns `python detect.py best.pt <image>` as a subprocess.
- `detect.py` loads the model via `ultralytics.YOLO`, runs inference, and
  prints a single JSON line on stdout.
- Java reads stdout, finds the JSON line, parses it with Jackson, and draws
  the polygon on a transparent `Canvas` overlaid on the `ImageView`.
- The polygon is drawn in source-image pixel coordinates and scaled to the
  rendered image size, accounting for `preserveRatio` letterboxing.

## Known notes

- First detection per app run is slower (model loading). Subsequent calls reuse
  the OS-level Python startup cost only — the model is reloaded each time
  because each detect call is a fresh subprocess. If this becomes annoying,
  the next step is a long-running Python server (e.g. ZeroMQ or a tiny
  Flask socket) — the current `DetectionService` interface won't change.
- The app supports `.jpg`, `.jpeg`, `.png`, `.bmp`.
- Use **←** / **→** keys for fast navigation; **Enter** to detect.

---

## Phase 2 — Classifier dataset preparation

`python/crop_cassettes.py` runs the existing detector on a labeled folder of
images and saves rotated, upright crops of each cassette into a dataset
structure ready for training the classifier.

### Input layout

Your existing Project 1 training data, organized by class:

```
<input_root>/
    positive/   *.jpg
    negative/   *.jpg
    invalid/    *.jpg
```

### Output layout

```
<output_root>/
    train/
        positive/  *.jpg   ← rotated, upright crops
        negative/  *.jpg
        invalid/   *.jpg
    val/
        positive/  *.jpg
        negative/  *.jpg
        invalid/   *.jpg
    crop_summary.json     ← stats: how many crops per class, what was skipped
```

This format is exactly what `yolo classify train` expects.

### Run

From the project root:

```bash
python python/crop_cassettes.py best.pt <input_root> <output_root>
```

Example (Windows paths — substitute your own dataset locations):

```bash
python python/crop_cassettes.py best.pt ^
    path\to\your\train ^
    path\to\your\classifier_dataset
```

Optional flags:

| Flag             | Default | Meaning                                              |
|------------------|---------|------------------------------------------------------|
| `--val-split`    | 0.2     | Fraction of each class to put in `val/`             |
| `--conf`         | 0.4     | Detector confidence threshold                       |
| `--pad`          | 0.04    | Padding around the cassette (fraction of long edge) |

### Preview a single crop first (recommended)

Before running the full batch, sanity-check the crop quality on one image:

```bash
python python/preview_crop.py best.pt path/to/one_image.jpg preview.jpg
```

This writes a side-by-side comparison (original with OBB | rotated crop) so you
can verify the cassette comes out upright with minimal background. If something
looks off, adjust `--pad` before running the full batch.

### Notes on the crop logic

- **EXIF orientation is normalized first** (same approach as `normalize.py`) —
  the saved JPEG has no orientation metadata to disagree about.
- **Only the highest-confidence detection per image is kept.** Images with no
  detection above the threshold are logged in `crop_summary.json` and skipped.
- The crop is a **true rotated crop** (perspective warp), not a bounding-box
  crop of the rotated rectangle. The cassette comes out upright regardless of
  the angle in the original photo.
- A stratified 80/20 split is performed per class with a fixed seed (42) so
  results are reproducible across runs.

### Next step (if invalid class is undersampled)

If `crop_summary.json` shows fewer than ~50 examples in the invalid class —
which is the common case, since invalid LFT photos are scarce online — you'll
want to generate synthetic invalid samples before training. See
**"Synthetic invalid samples"** below.

Otherwise, jump to training:

```bash
yolo classify train data=<output_root> model=yolov8n-cls.pt epochs=50 imgsz=224
```

---

## Phase 2 — Synthetic invalid samples

`python/synthesize_invalids.py` boosts the `invalid` class by generating
synthetic samples from your existing `negative` crops. It removes the C line
via inpainting, producing photorealistic blank cassettes (which qualify as
invalid by the project rules — "No C line, the test is invalid").

### Why we only synthesize from negatives

A test where only the T line shows (no C) is also technically invalid, and we
could in principle generate those from positive crops. We don't, because in an
upright crop we can't reliably tell which red line is C and which is T —
removing the wrong one creates a mislabeled training sample. We accept fewer
synthetic samples in exchange for **zero label noise**.

### Run

```bash
python python/synthesize_invalids.py "<classifier_dataset folder>" --target-count 80
```

### Output

```
<classifier_dataset>/synthetic_invalid/
    crops/      synthetic_*.jpg     ← the synthetic invalid images
    previews/   synthetic_*.jpg     ← 3-panel side-by-side previews
    log.json                        ← what was generated, what was skipped
```

Each preview shows three panels:
**[original negative]  |  [original with detected line in green]  |  [synthetic invalid]**

This makes it trivial to audit at a glance: if the green region in the middle
panel doesn't cover the red line cleanly, the synthetic on the right will
look wrong, and you delete that crop.

### How to audit & merge

1. Open `synthetic_invalid/previews/` in File Explorer.
2. Flip through. Realistic expectation: ~70% look perfect, ~25% slightly
   imperfect but still usable, ~5% obviously broken.
3. For any preview that looks broken, **delete the matching file in `crops/`**
   (same filename).
4. Copy the survivors from `crops/` into `<classifier_dataset>/train/invalid/`.

### Notes on the inpainting

- Uses HSV thresholding to find red pixels, then connected-component analysis
  to find the most line-shaped region (wide, not too tall, near the center).
- Restricted to the central 60% of the crop horizontally to ignore fingers /
  background red noise at the edges.
- If no line-shaped region is found in an image, the script **skips it** rather
  than producing a bad sample. The log records why.
- Uses `cv2.inpaint` with TELEA algorithm for the actual fill.

### Then train

After merging the audited synthetic samples into `train/invalid/`:

```bash
yolo classify train data=<classifier_dataset> model=yolov8n-cls.pt epochs=50 imgsz=224
```

---

## Phase 2 — Two-stage pipeline (localize → classify)

The app now runs a two-stage flow:

1. **Detect** (Stage 1): the YOLOv8-OBB model localizes the cassette and draws
   the oriented box. Side panel shows `test_device`, confidence, rotation, box
   size — unchanged from Phase 1.
2. **Read Result** (Stage 2): appears after a successful detection. It crops the
   cassette to an upright close-up (background removed, via OpenCV rotated warp)
   and runs a separate classifier to output **positive / negative / invalid**
   with per-class probabilities. The main image swaps to the crop.

### Classifier

`python/detect.py` loads the classifier in this priority order:

1. `python/classifier_mnv3.pt` — MobileNetV3-Small transfer-learning model
   (preferred; generalizes well to new photos).
2. `python/classifier.pt` — older YOLOv8-cls model (fallback).

### Retraining the classifier

`train_classifier_mobilenet.py`