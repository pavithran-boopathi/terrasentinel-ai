# TerraSentinel AI

Satellite-powered disaster intelligence platform: deep-learning flood segmentation from Sentinel-1 SAR,
flood-exposure analysis of roads and buildings, and separately evaluated building-damage assessment —
served through FastAPI, PostGIS, Celery and a React/MapLibre web app.

> **Status:** Phase 1 (foundation) complete. No models are trained yet, so the app shows no ML results.
> See [PROGRESS.md](PROGRESS.md) and [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Stack
**ML:** PyTorch (MPS/CUDA/CPU), U-Net, SegFormer, TorchGeo, Albumentations, MLflow ·
**Geo:** Rasterio/GDAL, GeoPandas, Shapely, PyProj, STAC, COG, PostGIS ·
**Backend:** FastAPI, SQLAlchemy 2, Alembic, PostgreSQL, Redis, Celery ·
**Frontend:** React 19, TypeScript, Tailwind 4, React Query, MapLibre GL ·
**Ops:** Docker Compose, GitHub Actions, Terraform/AWS (optional, later)

## Quick start (macOS, no Docker)
```bash
# one-time: Python 3.12 + Node 22 toolchain
conda create -n terrasentinel -c conda-forge --override-channels python=3.12 nodejs=22
conda activate terrasentinel

make setup                 # .venv + backend deps + frontend deps
cp .env.example .env       # optional: set TS_SECRET_KEY so logins survive restarts
make migrate               # creates data/terrasentinel-dev.db (SQLite)
make dev                   # API http://127.0.0.1:8000/api/v1/docs · web http://127.0.0.1:5173
```

## Quick start (Docker)
```bash
echo "TS_SECRET_KEY=$(python -c 'import secrets;print(secrets.token_urlsafe(48))')" > .env
docker compose up --build
```
Runs PostGIS, Redis, API, Celery worker and web.

## Checks
```bash
make test     # pytest + vitest
make lint     # ruff + tsc + eslint
make check-env
```

## Layout
See [ARCHITECTURE.md](ARCHITECTURE.md). Datasets: [data/README.md](data/README.md).
