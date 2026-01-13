FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app
ENV UV_LINK_MODE=copy

COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY README.md README.md
COPY configs/ configs/
COPY src/ src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Kjør python direkte fra venv som uv sync lager
ENTRYPOINT ["/app/.venv/bin/python", "-u", "/app/src/clickbait_classifier/evaluate.py"]

