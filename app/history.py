"""Persistent per-user session history in Firestore.

Documents live under users/{user_id}/reviews/{auto_id}, so each user's
development growth can be tracked over time and read back in the UI.

Performance notes:
  * Writes are scheduled as background tasks, so the review response is not
    held up by the Firestore round trip.
  * In-flight writes are tracked per user and awaited before that user's
    history is read (read-your-writes) and on shutdown (nothing is dropped).
"""
import asyncio
import logging
import os

from google.cloud import firestore

log = logging.getLogger("history")

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")

_db: firestore.AsyncClient | None = None
_pending: dict[str, set[asyncio.Task]] = {}


def _get_db() -> firestore.AsyncClient:
    global _db
    if _db is None:
        _db = firestore.AsyncClient(project=PROJECT)
    return _db


def _reviews(user_id: str):
    return _get_db().collection("users").document(user_id).collection("reviews")


async def _save(user_id: str, data: dict) -> None:
    try:
        doc = {**data, "created_at": firestore.SERVER_TIMESTAMP}
        await _reviews(user_id).add(doc)
    except Exception as e:  # history is best-effort; never fail the review
        log.warning("save failed: %s", e)


def save_review(user_id: str, data: dict) -> asyncio.Task:
    """Schedule the write without blocking the caller. Returns the task."""
    task = asyncio.create_task(_save(user_id, data))
    bucket = _pending.setdefault(user_id, set())
    bucket.add(task)

    def _done(t: asyncio.Task) -> None:
        bucket.discard(t)
        if not bucket:
            _pending.pop(user_id, None)

    task.add_done_callback(_done)
    return task


async def flush(user_id: str | None = None, timeout: float = 5.0) -> None:
    """Wait for in-flight writes (for one user, or for everyone)."""
    tasks = [
        t
        for uid, bucket in list(_pending.items())
        if user_id is None or uid == user_id
        for t in bucket
    ]
    if tasks:
        await asyncio.wait(tasks, timeout=timeout)


async def get_history(user_id: str, limit: int = 50) -> list[dict]:
    await flush(user_id)  # read-your-writes: include a review saved a moment ago
    try:
        q = (
            _reviews(user_id)
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
        out = []
        async for d in q.stream():
            rec = d.to_dict()
            ts = rec.get("created_at")
            out.append(
                {
                    "rating": rec.get("rating"),
                    "summary": rec.get("summary"),
                    "language": rec.get("language"),
                    "created_at": ts.isoformat() if hasattr(ts, "isoformat") else None,
                }
            )
        return out
    except Exception as e:
        log.warning("read failed: %s", e)
        return []


async def warmup() -> None:
    """Open the Firestore channel and fetch an auth token at startup."""
    if not PROJECT:
        return
    try:
        await _get_db().collection("users").document("startup-probe").get()  # any id not matching __.*__
        log.info("Firestore ready.")
    except Exception as e:
        log.warning("Firestore warmup failed: %s", e)
