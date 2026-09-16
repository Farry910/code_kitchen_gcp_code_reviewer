# 24/7 Intelligent Code Reviewer

An always-on, multi-language code reviewer on Google Cloud. Authenticated users
submit source code and get a bug report, architecture guidance, optimization
insights, and a standardized **1–10 rating** — grounded in a historical-review
corpus and stored as per-user history.

> **Operating the live service** (logs, rollback, scaling, settings, costs,
> troubleshooting) is covered in `../HOW_IT_WORKS.md`, Part B.

## Architecture (all GCP)

| Concern | Service |
|---|---|
| API / hosting (scale-to-zero, 24/7) | **Cloud Run** |
| Review engine | **Vertex AI — Gemini** |
| Historical learning (RAG) | **Vertex AI text embeddings** (→ Vector Search in prod) |
| Per-user history | **Firestore** |
| Privacy (secret/PII redaction) | **Cloud DLP** |
| Auth (production) | **Identity Platform** |

## Files
- `main.py` — FastAPI app: redact → ground → review → persist. Result cache, gzip, ETag.
- `reviewer.py` — Gemini call + fixed rubric + strict JSON schema (bounded thinking, timeout).
- `grounding.py` — embeds the CSV (batched, concurrent) and retrieves relevant rules.
- `privacy.py` — Cloud DLP redaction (async client).
- `history.py` — Firestore read/write; writes run off the response path.
- `gcp.py` — one shared GenAI client with retry policy.
- `cache.py` — small in-memory TTL/LRU cache.
- `rubric.py` — rubric text + output schema.
- `static/index.html` — demo UI (good for the video screentest).
- `sample_history.csv` — historical rules in the `<id,type,description>` schema.
- `tests/` — unit tests with every cloud client faked (`pytest`).

## How it stays fast

| Technique | Effect |
|---|---|
| Fully async request path (Gemini, embeddings, DLP, Firestore all awaited) | One instance serves ~80 concurrent reviews; no thread-pool cap |
| Gemini thinking disabled by default (`GEMINI_THINKING_BUDGET=0`) | Model call ~6 s instead of ~17 s with the dynamic default; same critical findings, equally consistent scores |
| Startup warm-up of every client in parallel | First request skips auth handshakes and channel setup |
| Firestore write scheduled in the background | The response never waits on the history round trip; `/history` still awaits in-flight writes for that user |
| Result cache keyed on SHA-256(code, language, model, rubric) | Resubmitting identical code returns in milliseconds; each submission is still recorded in history |
| Batched + concurrent embedding at startup | Thousands of historical rules index in seconds |
| gzip responses, in-memory UI with ETag/304 | Smaller payloads; reloads cost nothing |
| Cloud Run `--cpu-boost`, `--min-instances 1` | Faster cold starts, or none at all for the demo |

Every review response carries a `timings` object (`redact_ms`, `ground_ms`,
`review_ms`, `total_ms`) and a `cached` flag, and the UI shows them.

### Tuning (environment variables)

| Variable | Default | Notes |
|---|---|---|
| `GEMINI_MODEL` | `gemini-2.5-flash` | |
| `GEMINI_THINKING_BUDGET` | `0` | `0` = no thinking (fastest); e.g. `1024` caps it (~10 s/review); `-1` = send no thinking config (model default, ~17 s) |
| `GEMINI_TIMEOUT_MS` | `120000` | hard cap per model call; transient errors retry up to 3× |
| `EMBED_MODEL` | `text-embedding-005` | |
| `EMBED_BATCH` / `EMBED_CONCURRENCY` | `100` / `8` | indexing throughput for large CSVs |
| `GROUNDING_TOP_K` | `5` | rules injected into the prompt |
| `REVIEW_CACHE_SIZE` / `REVIEW_CACHE_TTL_S` | `512` / `3600` | set size to `0` to disable caching |
| `DLP_TIMEOUT_S` | `15` | redaction falls back to pass-through on failure |
| `MAX_CHARS` | `100000` | oversized submissions get 413 |
| `HISTORY_CSV` | `sample_history.csv` | path to the historical rules |

## Deploy from Cloud Shell

Run these in **Cloud Shell** (already authed to your lab project).

```bash
# 0. From the folder that contains this README:
export PROJECT_ID=$(gcloud config get-value project)
export REGION=us-central1

# 1. Enable APIs
gcloud services enable run.googleapis.com aiplatform.googleapis.com \
  firestore.googleapis.com dlp.googleapis.com cloudbuild.googleapis.com

# 2. Create the Firestore database (Native mode). Skip if it already exists.
gcloud firestore databases create --location=$REGION 2>/dev/null || true

# 3. Deploy to Cloud Run (builds the container for you via Cloud Build)
gcloud run deploy code-reviewer \
  --source . \
  --region $REGION \
  --allow-unauthenticated \
  --cpu-boost \
  --min-instances 1 \
  --concurrency 80 \
  --timeout 180 \
  --set-env-vars GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GCP_LOCATION=$REGION
```

`--min-instances 1` keeps one warm instance so the demo never cold-starts; use
`0` to scale to zero (and pay nothing while idle) at the cost of a few seconds
on the first request after a quiet spell.

Cloud Run prints a **Service URL** — open it in the browser to use the UI.

### Required roles
The Cloud Run runtime service account needs: **Vertex AI User**,
**Cloud Datastore User** (Firestore), and **DLP User**. In a Qwiklabs lab the
default service account is usually already broad enough.

## Try the API directly
```bash
URL=$(gcloud run services describe code-reviewer --region $REGION --format='value(status.url)')

curl -s -X POST "$URL/review" -H "Content-Type: application/json" -d '{
  "language": "python",
  "user_id": "demo-user",
  "code": "def f(d):\n  for i in d:\n    q = \"SELECT * FROM t WHERE x=\"+i\n    run(q)"
}' | python3 -m json.tool

# per-user history (newest first, ?limit=1..200)
curl -s "$URL/history/demo-user" | python3 -m json.tool
```

## Run locally (optional)
```bash
pip install -r requirements.txt
export GOOGLE_CLOUD_PROJECT=$(gcloud config get-value project)
gcloud auth application-default login   # for local ADC
uvicorn main:app --reload --port 8080
```

## Tests
```bash
pip install -r requirements-dev.txt
pytest -q
```
No cloud credentials needed: DLP, Vertex AI and Firestore are faked.

## Notes
- Grounding is in-memory for demo speed; it maps 1:1 to **Vertex AI Vector
  Search** in production (same embeddings, externalized ANN index).
- Submitted code is treated strictly as untrusted data (prompt-injection safe).
- DLP redaction, oversized/empty input handling, and rating clamping are built in.
- Only a hash of submitted code is kept in the result cache; the code itself is
  never retained in memory beyond the request.
