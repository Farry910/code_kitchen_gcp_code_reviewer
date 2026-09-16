"""Privacy: Cloud DLP (Sensitive Data Protection) redaction.

Secrets and PII are replaced with their info-type name (e.g. [EMAIL_ADDRESS])
BEFORE the code is persisted or sent to the model.

Uses the async DLP client, so a slow redaction never blocks other requests.
"""
import logging
import os

from google.cloud import dlp_v2

log = logging.getLogger("privacy")

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
DLP_TIMEOUT_S = float(os.environ.get("DLP_TIMEOUT_S", "15"))

INFO_TYPES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD_NUMBER",
    "US_SOCIAL_SECURITY_NUMBER",
    "GCP_CREDENTIALS",
    "AWS_CREDENTIALS",
    "AUTH_TOKEN",
    "PASSWORD",
    "IP_ADDRESS",
]

# Every request is identical apart from the text, so build these once.
_INSPECT_CONFIG = {
    "info_types": [{"name": t} for t in INFO_TYPES],
    "min_likelihood": dlp_v2.Likelihood.POSSIBLE,
}
_DEIDENTIFY_CONFIG = {
    "info_type_transformations": {
        "transformations": [
            {"primitive_transformation": {"replace_with_info_type_config": {}}}
        ]
    }
}

_client = None


def _get_client() -> "dlp_v2.DlpServiceAsyncClient":
    global _client
    if _client is None:
        _client = dlp_v2.DlpServiceAsyncClient()
    return _client


async def redact(text: str) -> tuple[str, dict]:
    """Best-effort redaction. Returns (possibly-redacted text, summary).

    DLP failures never block a review; we log and return the original text.
    """
    if not PROJECT:
        return text, {"applied": False, "reason": "no project configured"}
    try:
        resp = await _get_client().deidentify_content(
            request={
                "parent": f"projects/{PROJECT}/locations/global",
                "item": {"value": text},
                "inspect_config": _INSPECT_CONFIG,
                "deidentify_config": _DEIDENTIFY_CONFIG,
            },
            timeout=DLP_TIMEOUT_S,
        )
        redacted = resp.item.value
        return redacted, {"applied": True, "changed": redacted != text}
    except Exception as e:
        log.warning("redaction failed, passing text through: %s", e)
        return text, {"applied": False, "reason": str(e)[:200]}


async def warmup() -> None:
    """Open the gRPC channel and fetch an auth token at startup, off the request path."""
    if not PROJECT:
        return
    _, summary = await redact("warmup")
    log.info("DLP ready: %s", summary)
