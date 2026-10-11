# FlowForge — reproducible CPU-only image (author: 晨星)
FROM python:3.13-slim

LABEL maintainer="晨星"
LABEL org.opencontainers.image.description="FlowForge: deterministic normalizing-flow density estimation (RealNVP / MAF / FlowFuse)"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1

WORKDIR /app

# Pinned dependencies (Tier-1 path needs no GPU, no torch, no downloads).
COPY requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY pyproject.toml ./
COPY flowforge ./flowforge
COPY tests ./tests
COPY README.md LICENSE ./

# The package is importable from /app; no network needed at runtime.
ENV PYTHONPATH=/app

RUN python -m flowforge.cli selftest

CMD ["python", "flowforge/examples/run_demo.py", "--quick"]
