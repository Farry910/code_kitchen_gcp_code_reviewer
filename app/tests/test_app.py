"""Unit tests for the reviewer API with every Google Cloud client faked.

Run from app/:  pip install -r requirements-dev.txt && pytest -q
"""
import asyncio
import os
import pathlib
import sys
from types import SimpleNamespace

import httpx
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")

import history  # noqa: E402
import main  # noqa: E402
import privacy  # noqa: E402
import reviewer  # noqa: E402
from cache import TTLCache  # noqa: E402
from grounding import Grounding  # noqa: E402
from rubric import ReviewResult  # noqa: E402

pytestmark = pytest.mark.asyncio

SQL_SNIPPET = 'def get_user(uid):\n    q = "SELECT * FROM users WHERE id=" + uid\n    return db.execute(q)'
RULES = [
    {"id": "3", "type": "security", "description": "Never interpolate raw user input into SQL"},
    {"id": "1", "type": "formatting", "description": "Avoid single-character variable names"},
]


@pytest.fixture
def api(monkeypatch):
    """ASGI client with DLP, grounding, Gemini and Firestore replaced by fakes."""
    calls = {"review": [], "saves": []}

    async def fake_redact(text):
        redacted = text.replace("me@example.com", "[EMAIL_ADDRESS]")
        return redacted, {"applied": True, "changed": redacted != text}

    async def fake_retrieve(code, k=5):
        return RULES[:k]

    async def fake_review(code, language=None, rules=None):
        calls["review"].append({"code": code, "language": language, "rules": rules})
        return {
            "rating": 3,
            "summary": f"reviewed {language}",
            "findings": [],
            "strengths": [],
            "optimizations": [],
        }

    async def fake_save(user_id, data):
        await asyncio.sleep(0.01)  # behave like a real round trip
        calls["saves"].append((user_id, data))

    monkeypatch.setattr(privacy, "redact", fake_redact)
    monkeypatch.setattr(main.grounding, "retrieve", fake_retrieve)
    monkeypatch.setattr(reviewer, "review_code", fake_review)
    monkeypatch.setattr(history, "_save", fake_save)
    main.review_cache.clear()

    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://t")
    return client, calls


# ---------------------------------------------------------------- /review

async def test_empty_submission_is_400(api):
    client, _ = api
    r = await client.post("/review", json={"code": "   \n"})
    assert r.status_code == 400


async def test_oversized_submission_is_413(api, monkeypatch):
    client, _ = api
    monkeypatch.setattr(main, "MAX_CHARS", 10)
    r = await client.post("/review", json={"code": "x" * 11})
    assert r.status_code == 413


async def test_user_id_with_slash_is_rejected(api):
    client, _ = api
    r = await client.post("/review", json={"code": "x = 1", "user_id": "a/b"})
    assert r.status_code == 422


async def test_full_pipeline(api):
    client, calls = api
    body = {"code": SQL_SNIPPET + "\n# contact me@example.com", "language": "python", "user_id": "u1"}
    r = await client.post("/review", json=body)
    assert r.status_code == 200
    d = r.json()
    assert d["rating"] == 3
    assert d["grounded_on"] == ["3", "1"]
    assert d["redactions"] == {"applied": True, "changed": True}
    assert d["cached"] is False
    assert {"redact_ms", "ground_ms", "review_ms", "total_ms"} <= set(d["timings"])

    # The model saw the redacted code and the retrieved rules.
    assert len(calls["review"]) == 1
    assert "me@example.com" not in calls["review"][0]["code"]
    assert "[EMAIL_ADDRESS]" in calls["review"][0]["code"]
    assert calls["review"][0]["rules"] == RULES

    # History write happened in the background, under the right user.
    await history.flush("u1")
    assert len(calls["saves"]) == 1
    uid, data = calls["saves"][0]
    assert uid == "u1"
    assert data["rating"] == 3 and data["language"] == "python"


async def test_identical_submission_is_served_from_cache(api):
    client, calls = api
    body = {"code": SQL_SNIPPET, "language": "python", "user_id": "u2"}
    first = (await client.post("/review", json=body)).json()
    second = (await client.post("/review", json=body)).json()
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["rating"] == first["rating"]
    assert second["grounded_on"] == first["grounded_on"]
    assert len(calls["review"]) == 1, "the model must only be called once"
    # Every submission is still a session event for that user.
    await history.flush("u2")
    assert len(calls["saves"]) == 2
    assert main.review_cache.stats()["hits"] == 1


async def test_cache_key_includes_language(api):
    client, calls = api
    await client.post("/review", json={"code": SQL_SNIPPET, "language": "python"})
    await client.post("/review", json={"code": SQL_SNIPPET, "language": "sql"})
    assert len(calls["review"]) == 2


async def test_failed_review_is_not_cached(api, monkeypatch):
    client, calls = api

    async def broken(code, language=None, rules=None):
        calls["review"].append(None)
        return {**reviewer._UNPARSEABLE}

    monkeypatch.setattr(reviewer, "review_code", broken)
    body = {"code": "x = 1"}
    await client.post("/review", json=body)
    await client.post("/review", json=body)
    assert len(calls["review"]) == 2
    assert len(main.review_cache) == 0


# ---------------------------------------------------------------- /history

async def test_history_waits_for_pending_writes(api, monkeypatch):
    client, calls = api
    rows = []

    class FakeQuery:
        def order_by(self, *a, **k):
            return self

        def limit(self, n):
            return self

        async def stream(self):
            for r in rows:
                yield SimpleNamespace(to_dict=lambda r=r: r)

    monkeypatch.setattr(history, "_reviews", lambda user_id: FakeQuery())

    async def slow_save(user_id, data):
        await asyncio.sleep(0.05)
        rows.append({"rating": data["rating"], "summary": "s", "language": "py"})

    monkeypatch.setattr(history, "_save", slow_save)

    await client.post("/review", json={"code": "x = 1", "user_id": "u3"})
    # Immediately reading must include the review whose write is still in flight.
    r = await client.get("/history/u3?limit=10")
    assert r.status_code == 200
    assert [x["rating"] for x in r.json()["reviews"]] == [3]


async def test_history_limit_is_validated(api):
    client, _ = api
    assert (await client.get("/history/u?limit=0")).status_code == 422
    assert (await client.get("/history/u?limit=201")).status_code == 422


# ---------------------------------------------------------------- static / health

async def test_index_is_gzipped_and_supports_etag(api):
    client, _ = api
    r = await client.get("/", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200
    assert r.headers.get("content-encoding") == "gzip"
    etag = r.headers["etag"]
    assert etag
    r2 = await client.get("/", headers={"If-None-Match": etag})
    assert r2.status_code == 304


async def test_health(api):
    client, _ = api
    d = (await client.get("/health")).json()
    assert d["status"] == "ok"
    assert d["model"] == reviewer.MODEL
    assert "cache" in d


# ---------------------------------------------------------------- grounding

class FakeGenAI:
    """Embeds text as [#a, #b, #c] so similarity is predictable; counts requests."""

    def __init__(self):
        self.requests: list[list[str]] = []
        parent = self

        async def embed_content(model, contents, config=None):
            parent.requests.append(list(contents))
            embs = [SimpleNamespace(values=[t.count("a"), t.count("b"), t.count("c")]) for t in contents]
            return SimpleNamespace(embeddings=embs)

        self.aio = SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))


async def test_grounding_batches_large_corpora(tmp_path, monkeypatch):
    import grounding as g

    monkeypatch.setattr(g, "EMBED_BATCH", 100)
    csv = tmp_path / "rules.csv"
    csv.write_text(
        "id,type,description\n" + "".join(f"{i},t,a{'b' * (i % 3)}\n" for i in range(250)),
        encoding="utf-8",
    )
    fake = FakeGenAI()
    gr = Grounding(str(csv), client=fake)
    await gr.load()
    assert len(gr.rules) == 250
    assert gr.vectors.shape == (250, 3)
    assert [len(b) for b in fake.requests] == [100, 100, 50]


async def test_grounding_retrieves_top_k_in_order(tmp_path):
    csv = tmp_path / "rules.csv"
    csv.write_text(
        "﻿id,type,description\n"          # BOM must be tolerated
        "1,x,a\n2,x,b\n3,x,c\n4,x,ab\n"
        "5,x,\n"                               # malformed / empty row is skipped
        ",,   \n",
        encoding="utf-8",
    )
    gr = Grounding(str(csv), client=FakeGenAI())
    await gr.load()
    assert [r["id"] for r in gr.rules] == ["1", "2", "3", "4"]
    top = await gr.retrieve("aab", k=2)
    assert [r["id"] for r in top] == ["4", "1"]     # cos(ab)=0.95 > cos(a)=0.89
    assert len(await gr.retrieve("aab", k=10)) == 4  # k is clamped to corpus size


async def test_grounding_survives_missing_csv():
    gr = Grounding("does-not-exist.csv", client=FakeGenAI())
    await gr.load()
    assert gr.rules == [] and await gr.retrieve("x") == []


# ---------------------------------------------------------------- reviewer

async def test_reviewer_clamps_rating_and_handles_garbage(monkeypatch):
    parsed = ReviewResult(rating=42, summary="s", findings=[], strengths=[], optimizations=[])
    responses = iter([SimpleNamespace(parsed=parsed, text=""), SimpleNamespace(parsed=None, text="not json")])
    seen = {}

    async def generate_content(model, contents, config):
        seen["config"] = config
        seen["contents"] = contents
        return next(responses)

    fake = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    monkeypatch.setattr(reviewer, "get_genai_client", lambda: fake)

    ok = await reviewer.review_code(SQL_SNIPPET, language="python", rules=RULES)
    assert ok["rating"] == 10
    assert "[security] Never interpolate" in seen["contents"]
    assert seen["config"].thinking_config.thinking_budget == reviewer.THINKING_BUDGET
    assert seen["config"].http_options.timeout == reviewer.GEMINI_TIMEOUT_MS

    bad = await reviewer.review_code("x")
    assert bad["rating"] is None and bad["raw"] == "not json"


# ---------------------------------------------------------------- cache

async def test_ttl_cache_evicts_and_expires(monkeypatch):
    c = TTLCache(maxsize=2, ttl=100)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)
    assert c.get("a") is None and c.get("c") == 3     # LRU eviction
    import cache as cache_mod
    now = cache_mod.time.monotonic()
    monkeypatch.setattr(cache_mod.time, "monotonic", lambda: now + 101)
    assert c.get("c") is None                          # TTL expiry
    assert c.stats()["misses"] == 2
