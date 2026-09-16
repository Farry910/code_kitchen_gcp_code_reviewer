"""Historical learning (RAG grounding).

The <id, type, description> CSV is embedded once at startup with Vertex AI
text embeddings and held in memory for fast cosine-similarity retrieval.

Performance notes:
  * Embedding requests are batched (Vertex caps one request at 250 texts) and
    the batches run concurrently, so a large evaluation CSV indexes in seconds.
  * Everything is async: a review in flight never blocks other requests.
  * Rules are embedded as RETRIEVAL_DOCUMENT and the submitted code as
    RETRIEVAL_QUERY, the embedding model's asymmetric retrieval mode.

NOTE: in-memory retrieval keeps this demo fast on a time-boxed lab account.
In production this maps 1:1 to Vertex AI Vector Search (the same embeddings,
externalized to a managed ANN index): swap `_embed` + the numpy search for
Vector Search upserts/queries without changing the interface below.
"""
import asyncio
import csv
import logging
import os

import numpy as np
from google.genai import types

from gcp import get_genai_client

log = logging.getLogger("grounding")

EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-005")
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", "100"))            # texts per request (API max 250)
EMBED_CONCURRENCY = int(os.environ.get("EMBED_CONCURRENCY", "8"))  # parallel requests while indexing
EMBED_TIMEOUT_MS = int(os.environ.get("EMBED_TIMEOUT_MS", "30000"))
QUERY_CHARS = 8000  # ~2k tokens, the embedding model's input window

DOC_TASK = "RETRIEVAL_DOCUMENT"
QUERY_TASK = "RETRIEVAL_QUERY"


class Grounding:
    def __init__(self, csv_path: str, client=None):
        self.csv_path = csv_path
        self.rules: list[dict] = []      # [{id, type, description}]
        self.vectors: np.ndarray | None = None
        self._client = client            # injectable for tests

    def _get_client(self):
        if self._client is None:
            self._client = get_genai_client()
        return self._client

    async def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        client = self._get_client()
        config = types.EmbedContentConfig(
            task_type=task_type,
            http_options=types.HttpOptions(timeout=EMBED_TIMEOUT_MS),
        )
        sem = asyncio.Semaphore(EMBED_CONCURRENCY)

        async def embed_batch(batch: list[str]) -> list[list[float]]:
            async with sem:
                resp = await client.aio.models.embed_content(
                    model=EMBED_MODEL, contents=batch, config=config
                )
                return [e.values for e in resp.embeddings]

        batches = [texts[i:i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)]
        chunks = await asyncio.gather(*(embed_batch(b) for b in batches))
        vecs = np.array([v for chunk in chunks for v in chunk], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-8, None)  # L2-normalize for cosine

    def _read_rules(self) -> list[dict]:
        """Parse the CSV, tolerating a BOM and malformed / empty rows."""
        rules = []
        with open(self.csv_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                desc = (row.get("description") or "").strip()
                if not desc:
                    continue
                rules.append(
                    {
                        "id": (row.get("id") or "").strip(),
                        "type": (row.get("type") or "general").strip(),
                        "description": desc,
                    }
                )
        return rules

    async def load(self) -> None:
        """Parse the CSV and build the in-memory index. Never raises."""
        if not os.path.exists(self.csv_path):
            log.warning("CSV not found at %s; running ungrounded.", self.csv_path)
            return
        try:
            rules = self._read_rules()
            vectors = (
                await self._embed([r["description"] for r in rules], DOC_TASK)
                if rules
                else None
            )
            self.rules, self.vectors = rules, vectors
            log.info("loaded %d historical rules.", len(rules))
        except Exception as e:  # never let grounding crash startup
            log.warning("load failed, continuing ungrounded: %s", e)
            self.rules, self.vectors = [], None

    async def retrieve(self, code: str, k: int = 5) -> list[dict]:
        if self.vectors is None or not self.rules:
            return []
        try:
            q = (await self._embed([code[:QUERY_CHARS]], QUERY_TASK))[0]
            sims = self.vectors @ q
            k = min(k, len(sims))
            if k < len(sims):
                top = np.argpartition(-sims, k - 1)[:k]   # O(n) partial select
                top = top[np.argsort(-sims[top])]         # then order just those k
            else:
                top = np.argsort(-sims)
            return [self.rules[i] for i in top]
        except Exception as e:
            log.warning("retrieve failed: %s", e)
            return []
