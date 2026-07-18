# Single image for both processes; entrypoint selects api or bot (KB_PROCESS).
FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first for better layer caching. Includes the parse
# (docling), embed-local (onnxruntime) and mcp extras; only the api/bot runtime
# deps are needed but docling/ONNX load lazily at ingestion time.
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --extra parse --extra embed-local --extra mcp --no-install-project

COPY . .
RUN uv sync --no-dev --extra parse --extra embed-local --extra mcp

# Prefetch docling models so the ingestion process needs no network at runtime.
# The voyage-4-nano ONNX weights are provided at EMBED_MODEL_PATH (mounted or
# baked in a downstream image); set EMBED_BACKEND=openrouter to skip them.
RUN uv run python -c "from docling.utils.model_downloader import download_models; download_models()" \
    || echo "docling model prefetch skipped"

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV KB_PROCESS=api
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
