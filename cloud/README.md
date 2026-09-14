# LFT Reader API — cloud deployment

A COVID-19 lateral flow test reader, served as an HTTP API on Google Cloud Run.

**Live:** https://lft-reader-xufdqxj7ja-uc.a.run.app

```bash
curl -X POST -F "file=@your_photo.jpg" \
  https://lft-reader-xufdqxj7ja-uc.a.run.app/predict
```

```json
{
  "detected": true,
  "result": "positive",
  "confidence": 0.9811,
  "decision_source": "cnn+lines agree (positive)",
  "detection_confidence": 0.9084,
  "processing_ms": 523.3
}
```

> First request after ~15 minutes idle takes a few seconds — the service scales
> to zero, so the container has to start. Subsequent requests are ~500 ms.

> **Send a photo of a test, not a pre-cropped cassette.** The detector was
> trained to find a cassette *within a scene*. A tight crop with no surrounding
> context returns `detected: false`.

Research and demonstration use only. **Not a medical device.**

---

## What it does

Three stages, in order:

1. **Locate** — YOLOv8s-OBB finds the cassette as an *oriented* bounding box, so
   a test photographed at an angle is still cropped square.
2. **Read** — the crop is perspective-warped upright and classified
   positive / negative / invalid by a fine-tuned MobileNetV3-Small.
3. **Corroborate** — a classical OpenCV detector independently counts the
   coloured C/T bands (CLAHE → saturation boost → HSV threshold → shape gating).
   The two verdicts are fused with a deliberate bias against missing a positive:
   a false negative is clinically worse than a false positive.

```
photo ──► YOLOv8-OBB ──► rotated crop ──┬──► MobileNetV3 ──┐
          (detector.onnx)               │   (classifier.onnx)│
                                        │                   ├──► fuse ──► verdict
                                        └──► line counter ──┘
                                             (OpenCV, no ML)
```

## Why ONNX

The free-tier container has 1 GiB of RAM. `torch` + `torchvision` +
`ultralytics` is roughly **2 GB on disk** and wants hundreds of MB resident
just to import. ONNX Runtime is a **~15 MB wheel** that executes the same
frozen graphs.

A detail worth stating plainly, because it is counter-intuitive: **the model
files got slightly larger, not smaller.**

| File | PyTorch | ONNX |
|---|---|---|
| Detector | 23.18 MB (fp16) | 45.83 MB (fp32) |
| Classifier | 6.22 MB | 6.10 MB |

Ultralytics saves checkpoints in half precision; ONNX export writes float32, so
the same 11.4M parameters take twice the bytes. The win is the **runtime**, not
the weights.

Ultralytics does OBB decoding, rotated NMS and corner conversion in Python
*after* the model runs — none of it is in the exported graph (`end2end: False`).
Since installing `ultralytics` would drag `torch` back in, all of that is
reimplemented in NumPy in [`obb_postprocess.py`](obb_postprocess.py).

## Measured numbers

Every figure below was measured, not estimated. Scripts that produced them are
in this folder.

### Model accuracy

| Metric | Value |
|---|---|
| Detector mAP50 | **0.993** |
| Detector mAP50-95 | 0.857 |
| Classifier accuracy | **0.963** (78/81 held-out images) |

**Read the classifier number with its caveat.** The held-out set is 81 images
and heavily skewed: 60 positive, 16 negative, **5 invalid**. Macro-average F1 is
0.898 and the `invalid` class scores 0.800. "96.3%" is accurate; "96.3% on a
small, imbalanced set" is honest.

### ONNX conversion is faithful

| Check | Result |
|---|---|
| Classifier, max logit difference (PyTorch vs ONNX) | 3.4e-05 |
| Classifier, label disagreements over the held-out set | **0 / 81** |
| Detector postprocessing vs Ultralytics, polygon IoU | **1.00000** (20/20 images) |
| Detector postprocessing vs Ultralytics, confidence delta | **0.00000** |

### Speed

| | |
|---|---|
| ONNX Runtime vs eager PyTorch (classifier, 2 threads) | **3.4x faster** |
| Detector share of total inference time | **98%** |
| Model load, local container | 373 ms |
| Model load, Cloud Run | **1596 ms** |
| Warm request, Cloud Run | **~523 ms** |

The detector is 29.4 GFLOPs against the classifier's ~0.06 — profiling showed
the classifier was never the bottleneck, which is why no effort went into
optimising it.

## The bug this project found

While verifying that the ONNX export preserved accuracy, the held-out set
scored **0.84** instead of the expected 0.963 — with *identical* confusion
matrices from both runtimes, so ONNX was not the cause.

The cause was a **train/inference preprocessing mismatch**. The model was
validated on images resized by PIL (area-averaged, antialiased) but served
images resized by `cv2.INTER_LINEAR`, which samples a 2×2 neighbourhood and
discards everything between. A faint test line is exactly the thin, low-contrast
feature that aliasing erases.

| Preprocessing | Accuracy | Positives found |
|---|---|---|
| PIL resize (as validated) | **0.9630** | **60 / 60** |
| `cv2.INTER_LINEAR` (as served) | 0.8395 | 48 / 60 |
| `cv2.INTER_AREA` | 0.9383 | 59 / 60 |

**The model was never weak at faint positives.** It was being shown images with
the evidence already removed. This API uses PIL, matching the validation path
exactly.

A consequence worth recording: the classical line-counting layer was originally
built to rescue faint positives the CNN missed. Across the demo set it now fires
as a *corroborating* signal and overruled the CNN **zero** times — evidence that
it had been compensating for the resize bug rather than for a model weakness.

Reproduce with [`preprocessing_experiment.py`](preprocessing_experiment.py) and
[`preprocessing_experiment2.py`](preprocessing_experiment2.py).

## Known differences from the desktop app

The JavaFX app (`python/detect.py`) and this API are not identical. Three
deliberate divergences, all measured:

1. **PIL resize instead of `cv2.INTER_LINEAR`** — see above. The API is more
   accurate. `detect.py` still has the original behaviour.
2. **No 4-rotation classification pass.** `detect.py` classifies each crop at
   all four 90° rotations and keeps the most confident. On correctly-oriented
   crops that *loses* accuracy (0.9630 → 0.9259), because three of the four
   rotations are out-of-distribution and produce confidently wrong answers.
   *Caveat: the held-out set contains only upright crops, so it cannot test the
   near-square-OBB case the rotation was written for.*
3. **Square letterboxing.** The fixed 640×640 ONNX input requires padding to a
   full square; Ultralytics pads to the nearest multiple of 32. Detection boxes
   differ slightly (mean polygon IoU 0.965). `rotated_crop` pads 4% anyway, so
   this does not change verdicts.

End-to-end on the 20-image demo set: **this API 19/20, `detect.py` 18/20**,
and the single disagreement was a faint positive the API got right.

## Known limitations

- **`invalid` is the weak class.** Only 5 held-out examples and 50 training
  images. One demo image labelled `invalid` is misread as `negative` by *both*
  pipelines — a model limitation, not a port regression.
- **Pre-cropped cassettes are not detected.** Send a photo of the test in a
  scene.
- Softmax always sums to 1, so the classifier returns a confident-looking number
  for inputs it has never conceived of. Confidence alone cannot express "this is
  not a test strip"; that is what the detection stage is for.

## Running it

```bash
# local, no container
pip install -r requirements.txt
uvicorn main:app --port 8080

# container (matches Cloud Run limits)
docker build -t lft-reader:v1 .
docker run --rm -p 8080:8080 -e PORT=8080 --memory 1g --cpus 2 lft-reader:v1
```

Deployment walkthrough: [DEPLOY.md](DEPLOY.md).

### Endpoints

| | |
|---|---|
| `GET /` | upload page |
| `GET /health` | readiness probe; 503 until models load |
| `POST /predict` | multipart image upload → verdict |
| `GET /docs` | interactive OpenAPI docs (generated from type hints) |

`POST /predict` returns **200 with `detected: false`** when no cassette is
found. The request was well-formed and the analysis succeeded — "there is no
test strip in this picture" is a result, not a client error. Malformed input
gets 400 / 413 / 415.

## Cloud Run configuration

```
--memory=1Gi --cpu=2 --min-instances=0 --max-instances=3
--concurrency=4 --timeout=60 --allow-unauthenticated
```

`--min-instances=0` is what keeps it inside the always-free tier — nothing runs
when nobody is using it — and is also what causes cold starts.
`--max-instances=3` caps how far a traffic spike can scale, which matters more
than the budget alert: an unbounded service cannot be constrained after the
fact.

## Files

| | |
|---|---|
| `main.py` | FastAPI app |
| `pipeline.py` | detect → crop → classify → count lines → fuse |
| `obb_postprocess.py` | OBB decode, rotated NMS, geometry — NumPy only |
| `export_onnx.py` | PyTorch → ONNX with parity verification |
| `verify_*.py` | parity and accuracy checks |
| `preprocessing_experiment*.py` | the resize investigation |
| `bench_detector.py` | latency profiling |

## Licence

The detector is YOLOv8, which is **AGPL-3.0**. Section 13 covers software
offered over a network, which is what this is — hence the public source.

---

Built by Dareen Tobassy. Original desktop project: [../README.md](../README.md)
