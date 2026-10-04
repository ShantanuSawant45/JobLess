FROM python:3.12-slim

# uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:0.4.10 /uv /bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300

WORKDIR /app

# Install dependencies first so this layer is cached until deps change
COPY pyproject.toml uv.lock ./

# Install CPU-only PyTorch BEFORE uv sync.
# sentence-transformers depends on torch, and by default uv would pull the
# full CUDA build (5 GB+ of nvidia-* wheels) which times out in Docker.
# Pointing to the cpu index forces the tiny CPU wheel instead.
RUN uv pip install --system \
    torch==2.7.0+cpu torchvision==0.22.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app

# Run as non-root
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]