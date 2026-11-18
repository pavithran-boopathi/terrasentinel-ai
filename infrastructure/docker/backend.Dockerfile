# syntax=docker/dockerfile:1.7
# Two targets: `api` (FastAPI, no torch) and `worker` (Celery + CPU PyTorch for inference).
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN useradd --create-home --uid 1000 app
COPY pyproject.toml README.md ./
COPY backend ./backend
COPY ml ./ml
COPY geospatial ./geospatial
COPY alembic.ini ./

FROM base AS api
RUN pip install ".[api,geo]"
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health').status==200 else 1)"
CMD ["sh", "-c", "alembic upgrade head && uvicorn backend.main:app --host 0.0.0.0 --port 8000"]

FROM base AS worker
# CPU wheels: containers on macOS cannot use the Apple GPU; CUDA images are a separate concern.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install ".[api,geo,ml,ml-research]"
ENV HF_HUB_OFFLINE=1
USER app
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD celery -A backend.workers.celery_app inspect ping -d celery@$HOSTNAME --timeout 5
CMD ["celery", "-A", "backend.workers.celery_app", "worker", "--loglevel=INFO", "--concurrency=1"]
