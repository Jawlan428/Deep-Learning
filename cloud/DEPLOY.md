# Deploying the LFT Reader API to Google Cloud Run

Follow these in order. **Step 3 comes before any resource exists** — that is the point of it.

Nothing here should cost money. The always-free Cloud Run tier covers roughly
30,000 requests/month for this workload, and it does not expire.

---

## Step 1 — Install the Google Cloud CLI

Download and run the Windows installer:

<https://cloud.google.com/sdk/docs/install>

Tick **"Run gcloud init"** at the end. When the installer finishes, **open a NEW
terminal** — the installer edits `PATH`, and windows opened beforehand still
have the old one. (This is the same class of problem as the pip install that
went to the wrong Python earlier.)

Check it worked:

```
gcloud --version
```

---

## Step 2 — Create the account and a project

1. Go to <https://console.cloud.google.com> and sign in with your Google account.
2. Accept the free trial when offered. A **card is required for identity
   verification** — Google places a temporary authorisation hold, not a charge,
   and does **not** auto-charge when the trial ends. Billing simply stops and
   the account enters a grace period.
3. Create a project. Name it something like `lft-reader`. Note the **Project ID**
   (it may have digits appended, e.g. `lft-reader-472913`) — you need the ID, not
   the display name.

> **I cannot do this step for you and would not if I could.** Entering payment
> details is something you do yourself, in Google's own interface.

---

## Step 3 — Set a budget alert BEFORE creating anything

Do this while the project is still empty. A budget alert on a project with no
resources is a safety net you never notice; a budget alert set up after a
surprise is just a receipt.

1. Console → **Billing** → **Budgets & alerts** → **Create budget**
2. Scope: your project
3. Amount: **$1**
4. Thresholds: 50%, 90%, 100% — all "actual spend"
5. Tick **Email alerts to billing admins**

$1 is deliberately absurd. You should never reach it. That is exactly why it is
a good alarm: if it fires, something is wrong and you want to know immediately.

---

## Step 4 — Enable the two APIs and create a repository

Google Cloud services are off by default. Enabling is free; it just switches on
the API surface.

```
gcloud config set project YOUR_PROJECT_ID

gcloud services enable artifactregistry.googleapis.com run.googleapis.com

gcloud artifacts repositories create lft-repo ^
    --repository-format=docker ^
    --location=us-central1 ^
    --description="LFT reader container images"
```

### Why `us-central1` and not somewhere near Israel

Cloud Run's **always-free** compute allowance only applies in selected US
regions. `us-central1` qualifies; `me-west1` (Tel Aviv) and the European
regions do **not** — deploy there and you pay from the first request.

The cost is latency: roughly 150–200 ms extra round-trip from Israel. For a
demo where the model itself takes 1–2 seconds, that is noise. Choose free.

---

## Step 5 — Push the image you already tested

You built and verified `lft-reader:v1` locally. Push **that exact image** rather
than rebuilding from source in the cloud — a rebuild is a different artifact and
you have not tested it. Deploy what you verified.

Run `deploy_gcp.bat`, or do it by hand:

```
gcloud auth configure-docker us-central1-docker.pkg.dev

docker tag lft-reader:v1 us-central1-docker.pkg.dev/YOUR_PROJECT_ID/lft-repo/lft-reader:v1

docker push us-central1-docker.pkg.dev/YOUR_PROJECT_ID/lft-repo/lft-reader:v1
```

`configure-docker` writes a credential helper into your Docker config so
`docker push` can authenticate as you. It stores no password — it shells out to
`gcloud` for a short-lived token each time.

---

## Step 6 — Deploy

```
gcloud run deploy lft-reader ^
    --image=us-central1-docker.pkg.dev/YOUR_PROJECT_ID/lft-repo/lft-reader:v1 ^
    --region=us-central1 ^
    --platform=managed ^
    --memory=1Gi ^
    --cpu=2 ^
    --min-instances=0 ^
    --max-instances=3 ^
    --concurrency=4 ^
    --timeout=60 ^
    --allow-unauthenticated
```

### Every flag, and why

| Flag | Reason |
|---|---|
| `--memory=1Gi` | The default 512 MiB is not enough for ONNX Runtime plus 52 MB of models. Too low and the container is OOM-killed on first request. |
| `--cpu=2` | Matches what you benchmarked. The detector is 29.4 GFLOPs; one vCPU roughly doubles latency. |
| `--min-instances=0` | **This is what makes it free.** No container runs when nobody is using it. It is also what causes cold starts. |
| `--max-instances=3` | A ceiling on how far it can scale. Without it, a traffic spike (or a bot) could scale out and burn the free quota. This is the cost-safety flag. |
| `--concurrency=4` | How many requests one container handles at once. CPU-bound inference does not benefit from high concurrency — they would just queue while competing for 2 vCPUs. |
| `--timeout=60` | Fail fast. A request taking over a minute is stuck, not slow. |
| `--allow-unauthenticated` | Makes it genuinely public. Without it an interviewer clicking your link gets **403**. |

Deployment prints a URL like
`https://lft-reader-xxxxxxxxxx-uc.a.run.app`. That is the live URL.

---

## Step 7 — Verify

```
curl https://YOUR_URL/health

curl -X POST -F "file=@C:\path\to\positive_01.jpg" https://YOUR_URL/predict
```

Then open the URL in a browser. **Time the first request after a few minutes of
idle** — that is your real cold start, and it is the number to quote, not the
373 ms measured locally.

Expect roughly **1–2 s warm**, **3–6 s cold**.

---

## Staying inside the free tier

| Resource | Free allowance | Your usage |
|---|---|---|
| Cloud Run vCPU | 180,000 vCPU-s/month | ~6 vCPU-s per request → ~30,000 requests |
| Cloud Run memory | 360,000 GiB-s/month | ~3 GiB-s per request |
| Cloud Run requests | 2,000,000/month | not the binding limit |
| Artifact Registry | **0.5 GB** | one image ≈ 380 MB — **this is the tight one** |

**Artifact Registry is the constraint to watch.** Every redeploy pushes a new
image. Two versions will exceed 0.5 GB. Delete old ones:

```
gcloud artifacts docker images list us-central1-docker.pkg.dev/YOUR_PROJECT_ID/lft-repo

gcloud artifacts docker images delete us-central1-docker.pkg.dev/YOUR_PROJECT_ID/lft-repo/lft-reader:OLD_TAG
```

---

## What you should be able to explain afterwards

- **Why cold starts exist** — `--min-instances=0` means no container runs when
  idle; the first request pays for container start, Python import, and ~370 ms
  of ONNX session construction.
- **Why the container listens on `$PORT`** — Cloud Run injects it and does not
  guarantee 8080.
- **Why `exec` in the CMD** — so uvicorn is PID 1 and receives `SIGTERM`
  directly. Without it the shell holds PID 1, swallows the signal, and the
  container is SIGKILLed after the grace period.
- **Why `--max-instances`** — an unbounded service cannot exceed a budget you
  did not set. This one can't run away.
- **What you did NOT learn here** — this is serverless containers, not Linux
  server administration. You did not configure a security group, an SSH key, or
  a reboot policy, because Cloud Run has none of those. Say so if asked about
  EC2; do not imply otherwise.
