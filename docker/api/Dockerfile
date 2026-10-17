# The API (Phase 10): /ask and /health, on the runtime dependencies only (no
# torch; the `models` service embeds and reranks). It bakes in the index and a
# copy of the production retriever (D26), so it needs no DVC, tracking server
# or AWS at runtime. Build from the repo root after `dvc pull` and
# `python -m raglaw.retrieval.champion export` (README, "Build the images").

FROM python:3.14-slim@sha256:a2b82f3c48559aa0a8446d9af49826b6e2b2016f4cd2afabfe6013ec53729170 AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.20@sha256:100047e74f30778ab704942321a09750d6158739573ff58bf3924085cc6cd2d8 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /src
# Dependencies first, so a code change doesn't reinstall them.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-editable

FROM python:3.14-slim@sha256:a2b82f3c48559aa0a8446d9af49826b6e2b2016f4cd2afabfe6013ec53729170
RUN useradd --create-home --uid 10001 raglaw
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY params.yaml ./
# The query-drift baseline (Phase 14): numbers only, no question text.
COPY docs/metrics/drift_baseline.json ./docs/metrics/drift_baseline.json
# Qdrant's local mode writes a lock file beside the collection.
COPY --chown=raglaw:raglaw data/index ./data/index
COPY build/champion ./champion
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    RAGLAW_CHAMPION_DIR=/app/champion
USER raglaw
EXPOSE 8000
# Healthy only when every component is ready (/health's `status`).
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD ["python", "-c", "import json, sys, urllib.request; sys.exit(json.load(urllib.request.urlopen('http://localhost:8000/health'))['status'] != 'healthy')"]
CMD ["uvicorn", "raglaw.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
