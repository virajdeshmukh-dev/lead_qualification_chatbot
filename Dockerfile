# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.12.5

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:${PYTHON_VERSION}-slim AS builder
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
# 1) dependencies only (cached until the lockfile changes)
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
# 2) the project itself, as a regular (non-editable) install
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM python:${PYTHON_VERSION}-slim AS runtime
RUN groupadd --system app && useradd --system --gid app --home-dir /app app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src/lead_qualification_chatbot/resources ./data
RUN mkdir -p /app/.chroma && chown app:app /app/.chroma
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    APP_ENV=prod \
    LOG_JSON=true \
    DATA_DIR=/app/data \
    CHROMA_DIR=/app/.chroma \
    LEADS_FILE=/app/leads/leads.jsonl
RUN mkdir -p /app/leads && chown app:app /app/leads
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
# One worker on purpose: sessions, locks and rate limits are in-memory (see README).
CMD ["uvicorn", "lead_qualification_chatbot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]