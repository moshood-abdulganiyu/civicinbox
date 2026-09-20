FROM python:3.14-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application code and the dataset needed to train the baseline
COPY app/ app/
COPY data/labeled_messages.jsonl data/labeled_messages.jsonl
COPY scripts/train_baseline.py scripts/train_baseline.py

# Train the baseline model at build time — same reasoning as the CI step:
# no stale local artifact, reproducible from source on every build
RUN uv run python scripts/train_baseline.py

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]