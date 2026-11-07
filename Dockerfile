# LoanLens API image (Lesson 12). Build:  docker build -t loanlens .
FROM python:3.13-slim

# uv, copied from its official image (pinned version: builds are reproducible)
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

# 1. Dependencies first: this layer is cached until pyproject.toml or uv.lock change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 2. Then the code (changes often, so it comes last)
COPY src ./src

ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app/src PYTHONUNBUFFERED=1

# Don't run as root inside the container
RUN useradd --create-home loanlens
USER loanlens

EXPOSE 8000
CMD ["uvicorn", "loanlens.main:app", "--host", "0.0.0.0", "--port", "8000"]
