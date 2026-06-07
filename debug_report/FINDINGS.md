# Invalid-Class Diagnostic Report

Generated: 2026-06-07  
Dataset: `Not Valid-20260518T141146Z-3-001/Not Valid/` (10 images tested)  
Results: 5/10 correct overall — **all 5 failures explained below**

---

## Group 1 — 7 YOLO Stage-1 failures (no cassette detected)

Files: `download.png`, `7download.png`, `download (1-5).png`

### What they actually are

These are **not real cassette images**.  
Each file is a 1019×556 RGBA PNG where the entire canvas is fully transparent
(alpha = 0) except for 18–42 solid-red pixels (~RGB 255, 82, 82) near the centre.

| File              | Non-transparent pixels | Colour         |
|-------------------|------------------------|----------------|
| 7download.png     | 42                     | ~(255, 82, 82) |
| download.png      | 19                     | ~(255, 82, 82) |
| download (1-5).png| 18–21 each             | ~(255, 82, 82) |

That red colour and 1-px icon size is characteristic of a browser/CDN **error
icon** (broken-image or 404 placeholder) that was saved to disk instead of the
real photo.  When displayed on a white background the transparent canvas looks
white; the only visible content is the tiny red pixel.

### Why YOLO finds nothing

There is no cassette in these images. YOLO correctly returns zero detections.
The app correctly shows "No cassette detected". This is the right behaviour.

### Root cause

**Dataset corruption** — these 7 files are failed web downloads included in the
invalid-class folder.  They are not test images and should be deleted.

### Recommendation

**Delete all 7 `download*.png` files from the invalid dataset.**  
No model change or threshold change is needed. This is a data-quality issue.

---

## Group 2 — 1 Stage-2 failure (cassette detected, wrong verdict)

Source image: `0f3ab641-8592-407f-bf20-0bd48d88d8f2.png`  
(stock illustration: 4 cassettes, all labelled INVALID, all detected)  
Failing detection: **det3** (confidence 85.75%)

### What the image shows

The source image is a 4-cassette stock illustration.  YOLO detected all four
cassettes correctly.  Three were correctly classified **invalid**; one (det3)
was classified **negative**.

Looking at the annotated output and individual crops:

| Det | Cassette shown                             | Line at y_frac | CNN verdict    | Correct? |
|-----|--------------------------------------------|----------------|----------------|----------|
| 1   | Empty result window (no lines at all)      | — (0 lines)    | invalid (100%) | YES      |
| 2   | Single red line in upper zone (T position) | 0.313 (31%)    | invalid (100%) | YES      |
| 3   | Single red line in middle zone (T position)| 0.577 (58%)    | negative (99%) | **NO**   |
| 4   | Large red smear covering C+T zone          | — (0 lines)    | invalid (100%) | YES      |

### Why det3 was wrong

**Medical rule:** INVALID = control line (C) absent, regardless of T.  
Det3 shows one red band (the T line) with no C line — this is a T-only invalid.

**What the classifier sees:**  
The crop is 257×298 px.  A single red band is at y_frac = 0.577 (58% from top).
Both the line counter (1 line → "negative") and the CNN (99.02% "negative")
agree, so the fusion outputs "negative".

**Why the CNN is wrong here:**  
- The CNN was trained on negative examples where the control line (C) typically
  appears at various vertical positions within the crop.
- A single line at 58% crop height overlaps the positional range the CNN
  associates with a "control-line-only = negative" pattern.
- Det2 (correctly classified invalid) has its single line at 31% height — the
  CNN learned to associate a line in that upper zone with "T-only = invalid",
  but has not generalised that to a line at 58%.
- The result: the CNN cannot reliably distinguish C-only (negative) from T-only
  (invalid) when the T line happens to fall in the lower half of the crop.

### Debug image evidence

- `crop_det3.jpg` — the raw crop: single red horizontal band, no second line.
- `crop_det3_lines.jpg` — debug overlay: "1 line(s) → negative (conf 0.33)",
  line box at y=0.58, strength=0.33.
- `crop_det3_red_mask.jpg` — HSV mask: one clean isolated red blob, no second
  blob at the C position.
- `crop_det3_clf_input.jpg` — 224×224 input to the CNN: same single red band.

### Root cause

**CNN training gap** — the model has not seen enough "T-line-only" invalid
examples where the T line sits in the lower half of the crop.  The current valid
invalid training set (after removing the 7 corrupt files) contains only ~3 real
images, which is insufficient for the model to learn this sub-case.

---

## Summary of recommendations

| Issue                                     | Action needed                         | Priority |
|-------------------------------------------|---------------------------------------|----------|
| 7 corrupt download*.png in invalid folder | Delete from dataset                   | HIGH     |
| CNN confuses T-only-invalid with negative | More real invalid training images     | MEDIUM   |
| C vs T positional ambiguity in general    | Brand-specific positional calibration | LOW / future |
| YOLO Stage-1 detection                    | No change needed — works correctly    | —        |
| Thresholds (SAT_FLOOR, MERGE_FRAC, etc.)  | No change needed — not the cause      | —        |

### What does NOT need to change

- `best.pt` (YOLO detector) — found all real cassettes correctly
- `detect.py` thresholds — not the cause of any failure here
- Fusion logic — Tier 4.5 (CNN conf < 0.65 → invalid) did not fire for det3
  because the CNN was 99% confident. The logic is correct; the model is wrong.

### What would actually fix the det3 failure

1. **More invalid training images** — especially T-line-only invalids where the
   T line is in the lower half of the result window.  10–20 diverse real photos
   of genuine invalid tests (expired reagent, insufficient sample, etc.) would
   substantially improve CNN coverage.

2. **Positional calibration** — if a specific cassette brand is used exclusively,
   hard-coding the C-zone and T-zone Y-ranges for that brand would allow the
   classical line counter to identify which line is C and which is T.  This is
   the most reliable fix but requires brand-specific annotation.

3. **Synthesise T-only invalids** — `synthesize_invalids.py` already removes the
   C line from negatives to create blank invalids.  The inverse (removing the C
   line from positives to leave only the T line) would generate exactly the
   missing sub-class, though care is needed to label which line is which.
