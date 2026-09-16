"""Regenerate deploy_from_cloudshell.sh from the canonical files in app/.

Usage (from this folder):  python gen_deploy.py
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
APP = ROOT / "app"
FILES = [
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "sample_history.csv",
    "rubric.py",
    "gcp.py",
    "cache.py",
    "reviewer.py",
    "grounding.py",
    "privacy.py",
    "history.py",
    "main.py",
    "static/index.html",
]

HEAD = r"""#!/usr/bin/env bash
# ============================================================================
# ONE-PASTE DEPLOY for the 24/7 Intelligent Code Reviewer.
#
# HOW TO USE:
#   1. Open Cloud Shell in the GCP console (the >_ icon, top-right).
#      It is already authenticated as your lab student account.
#   2. Copy the ENTIRE contents of this file.
#   3. Paste into the Cloud Shell terminal and press Enter.
#
# It recreates the app, enables APIs, creates Firestore, and deploys to
# Cloud Run. At the end it prints your live Service URL.
#
# Tunables (export before pasting, all optional):
#   MIN_INSTANCES=1   keep one warm instance so the demo never cold-starts
#                     (set 0 to scale to zero and pay nothing while idle)
#
# GENERATED from app/ by gen_deploy.py — edit the files there, then re-run it.
# ============================================================================
set -euo pipefail

PROJECT_ID="$(gcloud config get-value project 2>/dev/null)"
REGION="us-central1"
MIN_INSTANCES="${MIN_INSTANCES:-1}"
APP_DIR="$HOME/intelligent_code_reviewer/app"

echo ">> Project: $PROJECT_ID   Region: $REGION"
mkdir -p "$APP_DIR/static"
cd "$APP_DIR"
"""

TAIL = r"""
echo ">> Files written to $APP_DIR"

# ---------------------------------------------------------------------------
# Enable APIs, create Firestore, deploy.
# ---------------------------------------------------------------------------
echo ">> Enabling APIs (this can take a minute)..."
gcloud services enable run.googleapis.com aiplatform.googleapis.com \
  firestore.googleapis.com dlp.googleapis.com cloudbuild.googleapis.com

echo ">> Creating Firestore database (ignored if it already exists)..."
gcloud firestore databases create --location="$REGION" 2>/dev/null || true

echo ">> Deploying to Cloud Run..."
#   --cpu-boost       extra CPU while the container starts (faster cold start)
#   --min-instances   1 = always-on, no cold starts for the demo
#   --concurrency 80  one async worker comfortably serves 80 in-flight reviews
gcloud run deploy code-reviewer \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --cpu-boost \
  --min-instances "$MIN_INSTANCES" \
  --concurrency 80 \
  --timeout 180 \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GCP_LOCATION=$REGION"

URL="$(gcloud run services describe code-reviewer --region "$REGION" --format='value(status.url)')"
echo ""
echo "============================================================"
echo " DONE. Open your reviewer here:"
echo "   $URL"
echo "============================================================"
"""

DELIM = "EOF_FILE"


def main() -> None:
    out = [HEAD]
    for rel in FILES:
        body = (APP / rel).read_text(encoding="utf-8").rstrip("\n")
        assert f"\n{DELIM}\n" not in f"\n{body}\n", f"{rel} contains the heredoc delimiter"
        bar = "-" * 75
        out.append(f"\n# {bar}\n# {rel}\n# {bar}\ncat > {rel} <<'{DELIM}'\n{body}\n{DELIM}\n")
    out.append(TAIL)
    target = ROOT / "deploy_from_cloudshell.sh"
    target.write_text("".join(out), encoding="utf-8", newline="\n")
    print(f"wrote {target.name} ({''.join(out).count(chr(10))} lines)")


if __name__ == "__main__":
    main()
