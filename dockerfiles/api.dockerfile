FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md

ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-install-project


COPY src/ src/
RUN uv sync --locked

ENTRYPOINT ["uv", "run", "uvicorn", "src.clickbait_classifier.api:app", "--host", "0.0.0.0", "--port", "8000"]

