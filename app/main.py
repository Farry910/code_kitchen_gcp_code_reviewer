"""24/7 Intelligent Code Reviewer: FastAPI entrypoint.

Flow per submission:
  1. Redact secrets/PII with Cloud DLP (privacy).
  2. Retrieve the most relevant historical rules (RAG grounding).
  3. Review with Vertex AI Gemini using a fixed 1-10 rubric.
  4. Persist the result to Firestore (per-user history).

Performance design:
  * Fully async request path; the single event loop handles many concurrent
    reviews (no thread pool cap) while each waits on Vertex / DLP / Firestore.
  * Startup warms every downstream client concurrently, so the first real
    request does not pay for auth handshakes and channel setup.
  * The Firestore write (step 4) is scheduled in the background; it is never
    on the response's critical path, yet /history still sees it immediately.
  * Identical submissions (same code, language, model, rubric) are served from
    an in-memory cache: resubmitting the same file costs milliseconds, not a
    fresh model call. Only the SHA-256 of the code is used as the key.
  * Responses are gzip-compressed and the UI is served with an ETag.
"""
import asyncio
import hashlib
import logging
import os
import pathlib
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import history
import privacy
import reviewer
from cache import TTLCache
from grounding import Grounding
from rubric import RUBRIC_VERSION

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("main")

MAX_CHARS = int(os.environ.get("MAX_CHARS", "100000"))
HISTORY_CSV = os.environ.get("HISTORY_CSV", "sample_history.csv")
TOP_K = int(os.environ.get("GROUNDING_TOP_K", "5"))
CACHE_SIZE = int(os.environ.get("REVIEW_CACHE_SIZE", "512"))
CACHE_TTL_S = float(os.environ.get("REVIEW_CACHE_TTL_S", "3600"))

grounding = Grounding(csv_path=HISTORY_CSV)
review_cache = TTLCache(maxsize=CACHE_SIZE, ttl=CACHE_TTL_S)

# The UI is a single static page: read it once, serve it from memory with an
# ETag so a reload after the first visit is a 304 with no body.
_INDEX_HTML = (pathlib.Path(__file__).parent / "static" / "index.html").read_bytes()
_INDEX_ETAG = '"' + hashlib.sha256(_INDEX_HTML).hexdigest()[:16] + '"'


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the grounding index and open the DLP / Firestore channels
    # concurrently, off the request path.
    t0 = time.perf_counter()
    await asyncio.gather(grounding.load(), privacy.warmup(), history.warmup())
    log.info("startup complete in %.0f ms", (time.perf_counter() - t0) * 1000)
    yield
    await history.flush(timeout=8.0)  # don't drop in-flight history writes


app = FastAPI(title="24/7 Intelligent Code Reviewer", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024)


class ReviewRequest(BaseModel):
    code: str = Field(..., min_length=1, description="Source code to review.")
    language: str | None = Field(default=None, max_length=64, description="Optional language hint.")
    user_id: str = Field(
        default="anonymous",
        min_length=1,
        max_length=128,
        pattern=r"^[^/]+$",
        description="Authenticated user id.",
    )


def _cache_key(code: str, language: str | None) -> str:
    h = hashlib.sha256()
    h.update(f"{RUBRIC_VERSION}|{reviewer.MODEL}|{language or ''}|".encode())
    h.update(code.encode())
    return h.hexdigest()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if request.headers.get("if-none-match") == _INDEX_ETAG:
        return Response(status_code=304, headers={"ETag": _INDEX_ETAG})
    return HTMLResponse(
        _INDEX_HTML, headers={"ETag": _INDEX_ETAG, "Cache-Control": "no-cache"}
    )


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "rules_loaded": len(grounding.rules),
        "model": reviewer.MODEL,
        "rubric_version": RUBRIC_VERSION,
        "cache": review_cache.stats(),
    }


@app.post("/review")
async def review(req: ReviewRequest) -> dict:
    code = req.code
    # Edge case: empty / whitespace-only submission.
    if not code.strip():
        raise HTTPException(status_code=400, detail="Nothing to review.")
    # Edge case: oversized submission.
    if len(code) > MAX_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Submission too large (max {MAX_CHARS} chars).",
        )

    t0 = time.perf_counter()
    timings: dict[str, int] = {}

    def lap(name: str, since: float) -> float:
        now = time.perf_counter()
        timings[name] = round((now - since) * 1000)
        return now

    key = _cache_key(code, req.language)
    cached = review_cache.get(key)
    if cached is not None:
        result = {**cached, "cached": True}
    else:
        # 1. Privacy: strip secrets/PII before the code is stored or sent onward.
        safe_code, redaction_summary = await privacy.redact(code)
        t = lap("redact_ms", t0)

        # 2. Grounding: pull the most relevant historical rules for this code.
        rules = await grounding.retrieve(safe_code, k=TOP_K)
        t = lap("ground_ms", t)

        # 3. Review with Gemini against the fixed rubric.
        result = await reviewer.review_code(safe_code, language=req.language, rules=rules)
        lap("review_ms", t)

        result["grounded_on"] = [r["id"] for r in rules]
        result["redactions"] = redaction_summary
        if result.get("rating") is not None:  # never cache a failed review
            review_cache.set(key, result)
        result = {**result, "cached": False}

    # 4. Persist per-user history for growth tracking (off the critical path).
    history.save_review(
        req.user_id,
        {
            "language": req.language,
            "rating": result.get("rating"),
            "summary": result.get("summary"),
            "redactions": result.get("redactions"),
        },
    )

    timings["total_ms"] = round((time.perf_counter() - t0) * 1000)
    result["timings"] = timings
    return result


@app.get("/history/{user_id}")
async def get_user_history(
    user_id: str, limit: int = Query(default=50, ge=1, le=200)
) -> dict:
    return {"user_id": user_id, "reviews": await history.get_history(user_id, limit=limit)}
