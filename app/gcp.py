"""Shared, lazily-created Google GenAI client (Vertex AI backend).

One client means one HTTP connection pool and one cached auth token shared by
the reviewer (Gemini) and grounding (embeddings) modules instead of two.
"""
import os

from google import genai
from google.genai import types

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

# Retry transient failures (rate limits, 5xx) with capped exponential backoff.
# Lab projects have small per-minute embedding quotas, so 429s are expected
# under bursty load; four attempts with backoff ride them out.
RETRY = types.HttpRetryOptions(
    attempts=4,
    initial_delay=1.0,
    max_delay=10.0,
    http_status_codes=[408, 429, 500, 502, 503, 504],
)

_client: genai.Client | None = None


def get_genai_client() -> genai.Client:
    global _client
    if _client is None:
        # Uses Application Default Credentials on Cloud Run / Cloud Shell.
        _client = genai.Client(
            vertexai=True,
            project=PROJECT,
            location=LOCATION,
            http_options=types.HttpOptions(retry_options=RETRY),
        )
    return _client
