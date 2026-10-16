# The `models` service (D18): the pinned embedder and reranker over HTTP, so
# the API needs no torch. CPU-only torch by default; `--build-arg
# TORCH_GROUP=torch-gpu` builds on CUDA 13 instead (docker/compose.gpu.yml).
# The weights are not in the image: `python -m raglaw.serving.fetch_models`
# (compose's `models-pull`) downloads them once, at their pinned revisions, into
# the `hf-cache` volume before the service starts.

FROM python:3.14-slim@sha256:a2b82f3c48559aa0a8446d9af49826b6e2b2016f4cd2afabfe6013ec53729170 AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.20@sha256:100047e74f30778ab704942321a09750d6158739573ff58bf3924085cc6cd2d8 /uv /bin/uv
ARG TORCH_GROUP=torch-cpu
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /src
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group embed --group "$TORCH_GROUP" \
        --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group embed --group "$TORCH_GROUP" \
        --no-editable

FROM python:3.14-slim@sha256:a2b82f3c48559aa0a8446d9af49826b6e2b2016f4cd2afabfe6013ec53729170
RUN useradd --create-home --uid 10001 raglaw \
    && mkdir -p /home/raglaw/.cache/huggingface \
    && chown -R raglaw:raglaw /home/raglaw/.cache
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY params.yaml ./
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/raglaw/.cache/huggingface \
    HF_HUB_DISABLE_XET=1 \
    RAGLAW_DEVICE=cpu
USER raglaw
EXPOSE 8001
# The weights are already in the volume (models-pull), so loading takes a minute.
HEALTHCHECK --interval=15s --timeout=5s --start-period=3m --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"]
CMD ["uvicorn", "raglaw.serving.models_app:app", "--host", "0.0.0.0", "--port", "8001"]
