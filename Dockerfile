FROM python:3.14-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency + metadata files first for better layer caching
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code and the dataset needed to train the baseline
COPY app/ app/
COPY data/labeled_messages.jsonl data/labeled_messages.jsonl
COPY scripts/train_baseline.py scripts/train_baseline.py

# Now install the project itself (fast — deps already cached above)
RUN uv sync --frozen --no-dev

# Train the baseline model at build time — no stale local artifact,
# reproducible from source on every build
RUN uv run python scripts/train_baseline.py

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]