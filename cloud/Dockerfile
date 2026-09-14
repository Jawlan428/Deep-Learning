# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# LFT Reader API — slim container, no PyTorch.
#
# MULTI-STAGE BUILD, AND WHY IT MATTERS HERE
#   Every RUN, COPY and ADD adds a layer, and layers are cumulative: deleting a
#   file in a later layer does NOT reclaim its space, it just hides it. So
#   "pip install && apt-get purge build-essential" in one image saves nothing —
#   the compiler is still in the history, still in the pushed image.
#
#   A multi-stage build sidesteps that. The `builder` stage installs everything
#   into a virtualenv. The final stage starts from a clean base and copies ONLY
#   that virtualenv across. Whatever the builder needed to compile the wheels
#   never reaches the shipped image, because that image has no ancestry
#   containing it.
#
# EXPECTED SIZE
#   python:3.11-slim base    ~130 MB
#   onnxruntime               ~50 MB
#   opencv-python-headless    ~90 MB
#   numpy / Pillow / fastapi  ~55 MB
#   models (45.8 + 6.1)       ~52 MB
#   -------------------------------
#   roughly 380 MB, inside the 500 MB target.
#
#   For contrast, a torch + ultralytics image for the same job lands around
#   2 GB and will not fit in 1 GiB of RAM.
# ---------------------------------------------------------------------------

# =========================== STAGE 1: builder ==============================
FROM python:3.11-slim AS builder

# A venv is the cleanest unit to copy between stages: one self-contained
# directory holding every dependency, with no system packages entangled.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# requirements.txt is copied ALONE, before the application code.
# Docker caches each layer and invalidates everything after the first change.
# Copying the whole directory first would mean every edit to main.py triggers
# a full reinstall of onnxruntime and opencv. This way, dependencies are only
# reinstalled when requirements.txt itself changes.
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================ STAGE 2: runtime =============================
FROM python:3.11-slim

# opencv-python-headless drops the GUI stack but still links glib. Without
# this the container builds fine and then dies at import with
# "ImportError: libgthread-2.0.so.0: cannot open shared object file" —
# a runtime failure from a missing build-time dependency, which is exactly
# the kind of bug that only shows up after deployment.
#
# rm -rf /var/lib/apt/lists/* in the SAME RUN matters: a separate RUN would
# leave the package index in the previous layer, still occupying space.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Run as a non-root user. If the process is ever compromised, the attacker
# lands as an unprivileged account rather than root. Cloud Run is sandboxed
# anyway, but defence in depth costs one line here.
RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/app/models \
    ORT_THREADS=2 \
    PORT=8080

# PYTHONUNBUFFERED is not cosmetic. Without it Python buffers stdout, and when
# a container is killed the last buffered logs are lost — so the output you
# most need after a crash is the output you never see.

WORKDIR /app

COPY --chown=appuser:appuser main.py pipeline.py obb_postprocess.py ./
COPY --chown=appuser:appuser models/detector.onnx \
                             models/classifier.onnx \
                             models/classifier_meta.json \
                             ./models/

USER appuser

EXPOSE 8080

# CMD in shell form so ${PORT} is expanded at runtime — Cloud Run injects PORT
# and it is not guaranteed to be 8080. `exec` replaces the shell with uvicorn
# so the server becomes PID 1 and receives SIGTERM directly; without it, the
# shell holds PID 1, swallows the signal, and the container is SIGKILLed after
# the grace period instead of shutting down cleanly.
#
# --workers 1 is deliberate. Cloud Run scales by adding CONTAINERS, not
# processes, and each replica has 2 vCPUs. A second worker would load a second
# copy of both models into the same 1 GiB and compete for the same cores.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1
