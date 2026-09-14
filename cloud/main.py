"""
main.py — the FastAPI service.

WHAT FASTAPI IS, SINCE YOU HAVE NOT USED IT
    A web framework built on type hints. You declare a function, decorate it
    with the HTTP method and path, and annotate the parameters; FastAPI reads
    those annotations to parse the request, validate it, produce a 422 with a
    useful message when validation fails, and generate OpenAPI docs. The
    interactive docs at /docs are not a library you add — they fall out of the
    type annotations for free.

    Coming from Java: it is roughly Spring Boot's annotation-driven controllers,
    except the schema comes from ordinary Python types rather than a separate
    DTO class hierarchy.

ENDPOINTS
    GET  /         a small upload page, so a browser visit shows something real
    GET  /health   liveness probe. Returns 200 only once models are loaded.
    POST /predict  multipart image upload -> verdict JSON
    GET  /docs     interactive OpenAPI docs (generated)

RUN LOCALLY
    ..\\.venv\\Scripts\\python.exe -m pip install fastapi "uvicorn[standard]" python-multipart
    ..\\.venv\\Scripts\\python.exe -m uvicorn main:app --reload --port 8080
        (run from inside the cloud/ directory)

    Then open http://127.0.0.1:8080/docs
"""

from __future__ import annotations

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

import pipeline as pl

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("lft.api")

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", HERE / "models"))

# Cloud Run gives 2 vCPUs on the free tier; match the thread pool to that.
THREADS = int(os.environ.get("ORT_THREADS", "2"))

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}

STATE: dict = {"pipeline": None, "loaded_at": None, "startup_ms": None}


# ==========================================================================
# Startup
# ==========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the models ONCE, before the first request.

    Building an ONNX Runtime session re-optimizes the whole graph, which costs
    far more than an inference does. Doing it per request is the classic way to
    make a fast model look slow.

    On Cloud Run this runs during cold start. That is the right place for it:
    the container is not marked ready until this finishes, so no user request
    ever waits on model loading.
    """
    t0 = time.perf_counter()
    log.info("loading models from %s", MODEL_DIR)
    STATE["pipeline"] = pl.Pipeline(MODEL_DIR, threads=THREADS)
    STATE["startup_ms"] = round((time.perf_counter() - t0) * 1000)
    STATE["loaded_at"] = time.time()
    log.info("models ready in %d ms", STATE["startup_ms"])
    yield
    log.info("shutting down")


app = FastAPI(
    title="LFT Reader API",
    version="1.0.0",
    description=(
        "Reads a COVID-19 lateral flow test from a photo.\n\n"
        "Two-stage computer vision: a YOLOv8-OBB detector locates the cassette, "
        "then a MobileNetV3-Small classifier reads the result. A classical "
        "red-line detector counts C/T bands independently and the two signals "
        "are fused with a deliberate bias against missing a positive.\n\n"
        "Runs on ONNX Runtime — no PyTorch in the container."
    ),
    lifespan=lifespan,
)


def get_pipeline() -> pl.Pipeline:
    p = STATE.get("pipeline")
    if p is None:
        # 503, not 500: the service is fine, it is just not ready yet.
        raise HTTPException(status_code=503, detail="Models are still loading. Retry shortly.")
    return p


# ==========================================================================
# Image decoding
# ==========================================================================
def decode_image(data: bytes) -> np.ndarray:
    """Bytes -> BGR numpy array, with EXIF orientation applied.

    EXIF matters more than it looks. Phone cameras usually store the sensor
    image unrotated and record 'this is rotated 90 degrees' in a metadata tag.
    Ignore the tag and a portrait photo arrives on its side — the detector
    will often still find the cassette, but the crop comes out sideways and
    the classifier sees something it was never trained on.

    python/normalize.py does exactly this before detect.py runs, so applying
    it here keeps the API on the same footing as the desktop app.
    """
    import cv2
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im)   # bake rotation into pixels
            rgb = np.asarray(im.convert("RGB"))
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the file as an image. Supported: JPEG, PNG, WEBP, BMP.",
        )
    except Exception as exc:
        log.exception("decode failed")
        raise HTTPException(status_code=400, detail=f"Could not read the image: {exc}")

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise HTTPException(status_code=400, detail="Image must be a colour (3-channel) image.")
    if min(rgb.shape[:2]) < 32:
        raise HTTPException(
            status_code=400,
            detail=f"Image is too small ({rgb.shape[1]}x{rgb.shape[0]}). Minimum 32x32.",
        )

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ==========================================================================
# Endpoints
# ==========================================================================
@app.get("/health")
def health():
    """Liveness / readiness probe.

    Deliberately does NOT run inference — a health check that does real work
    can fail under load and get a healthy container killed.
    """
    ready = STATE.get("pipeline") is not None
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ok" if ready else "loading",
            "models_loaded": ready,
            "startup_ms": STATE.get("startup_ms"),
            "uptime_s": round(time.time() - STATE["loaded_at"], 1) if STATE.get("loaded_at") else None,
        },
    )


@app.post("/predict")
async def predict(file: UploadFile = File(..., description="Photo of a lateral flow test")):
    """Read a lateral flow test from an uploaded photo.

    Returns 200 with `detected: false` when no cassette is found. That is a
    deliberate choice: the request was well-formed and the analysis succeeded
    — the answer is simply 'there is no test strip in this picture'. A 4xx
    would imply the CLIENT did something wrong, and sending a photo of a cat
    is not a protocol error. Errors are for malformed input; this is a result.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    if file.content_type and file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(f"Unsupported content type '{file.content_type}'. "
                    f"Send one of: {', '.join(sorted(ALLOWED_TYPES))}."),
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(f"File is {len(data) / 1e6:.1f} MB; the limit is "
                    f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB."),
        )

    img = decode_image(data)
    pipe = get_pipeline()

    t0 = time.perf_counter()
    try:
        out = pipe.analyze(img)
    except Exception as exc:
        # Never leak a stack trace to the caller. Log the detail, return a
        # generic 500 — a traceback in an HTTP response is an information
        # disclosure problem as well as an unhelpful one.
        log.exception("analysis failed for %s", file.filename)
        raise HTTPException(status_code=500, detail="Analysis failed. The error has been logged.")
    elapsed = round((time.perf_counter() - t0) * 1000, 1)

    if out["num_detections"] == 0:
        return {
            "detected": False,
            "result": None,
            "message": ("No lateral flow test was found in this image. "
                        "Photograph the cassette straight on, filling most of the frame, "
                        "in even lighting."),
            "image_size": out["image_size"],
            "processing_ms": elapsed,
        }

    top = out["result"]
    return {
        "detected": True,
        "result": top["label"],
        "confidence": top["confidence"],
        "decision_source": top["decision_source"],
        "detection_confidence": top["detection_confidence"],
        "classifier": top["classifier"],
        "line_analysis": top["line_analysis"],
        "num_detections": out["num_detections"],
        "all_detections": out["detections"] if out["num_detections"] > 1 else None,
        "image_size": out["image_size"],
        "processing_ms": elapsed,
        "disclaimer": ("Research and demonstration use only. Not a medical device "
                       "and not a substitute for a clinical test result."),
    }


@app.get("/", response_class=HTMLResponse)
def index():
    """A minimal upload page, so opening the URL in a browser shows something.

    Without this, a GET on / returns {"detail":"Not Found"} — a poor first
    impression for anyone you send the link to.
    """
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LFT Reader API</title>
<style>
 body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:46rem;
      margin:3rem auto;padding:0 1.25rem;line-height:1.6;color:#1a1a1a}
 h1{margin-bottom:.25rem} .sub{color:#666;margin-top:0}
 .card{border:1px solid #e2e2e2;border-radius:10px;padding:1.25rem;margin:1.5rem 0}
 button{background:#0F6E56;color:#fff;border:0;border-radius:6px;
        padding:.6rem 1.1rem;font-size:1rem;cursor:pointer}
 button:disabled{opacity:.5;cursor:default}
 pre{background:#f6f6f6;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}
 .verdict{font-size:1.5rem;font-weight:600;margin:.5rem 0}
 .positive{color:#c0392b}.negative{color:#0F6E56}.invalid{color:#b8860b}
 code{background:#f6f6f6;padding:.15rem .35rem;border-radius:3px}
 .note{font-size:.85rem;color:#666}
</style></head><body>
<h1>LFT Reader API</h1>
<p class="sub">COVID-19 lateral flow test reader &mdash; YOLOv8-OBB + MobileNetV3, on ONNX Runtime.</p>

<div class="card">
  <input type="file" id="f" accept="image/*">
  <button id="go" onclick="send()">Analyse</button>
  <div id="out"></div>
</div>

<p class="note">Research and demonstration use only. Not a medical device.</p>
<p>API docs: <a href="/docs">/docs</a> &middot; Health: <a href="/health">/health</a></p>
<pre>curl -X POST -F "file=@test.jpg" $(location.origin)/predict</pre>

<script>
document.querySelector('pre').textContent =
  'curl -X POST -F "file=@test.jpg" ' + location.origin + '/predict';

async function send(){
  const f = document.getElementById('f').files[0];
  const out = document.getElementById('out');
  const btn = document.getElementById('go');
  if(!f){ out.innerHTML = '<p>Choose an image first.</p>'; return; }
  btn.disabled = true;
  out.innerHTML = '<p>Analysing&hellip; (first request after idle may take ~15s)</p>';
  const fd = new FormData(); fd.append('file', f);
  try{
    const r = await fetch('/predict', {method:'POST', body:fd});
    const j = await r.json();
    if(!r.ok){ out.innerHTML = '<p><b>Error ' + r.status + ':</b> ' + (j.detail||'') + '</p>'; }
    else if(!j.detected){ out.innerHTML = '<p>' + j.message + '</p>'; }
    else {
      out.innerHTML = '<div class="verdict ' + j.result + '">' + j.result.toUpperCase() +
        '</div><p>Confidence ' + (j.confidence*100).toFixed(1) + '% &middot; ' +
        j.processing_ms + ' ms<br><span class="note">' + j.decision_source + '</span></p>' +
        '<pre>' + JSON.stringify(j, null, 2) + '</pre>';
    }
  }catch(e){ out.innerHTML = '<p><b>Request failed:</b> ' + e + '</p>'; }
  btn.disabled = false;
}
</script></body></html>"""
