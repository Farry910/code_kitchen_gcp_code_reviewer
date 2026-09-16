FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so this (slow) layer is cached across code-only changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# Pre-compile bytecode so cold starts skip it.
RUN python -m compileall -q .

# Cloud Run injects PORT (default 8080). uvicorn[standard] picks up uvloop +
# httptools automatically; one async worker per vCPU is the right shape here.
ENV PORT=8080
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips="*"
