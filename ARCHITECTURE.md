# Architecture

## System overview

```
                ┌──────────────────────────── Browser ────────────────────────────┐
                │ React + TS + Tailwind · React Query · MapLibre/Deck.gl · Plotly │
                └───────────────┬─────────────────────────────────────────────────┘
                                │ HTTPS /api/v1 (JWT bearer)
                ┌───────────────▼───────────────┐        ┌───────────────────────┐
                │ FastAPI (uvicorn)             │ enqueue│ Celery workers        │
                │ auth · uploads · jobs ·       ├───────►│ tiled inference ·     │
                │ models · analysis · reports   │  Redis │ geospatial analysis · │
                └───────┬───────────────┬───────┘        │ report generation     │
                        │ SQLAlchemy    │                └───┬──────────┬────────┘
                ┌───────▼─────────┐ ┌───▼──────────────────────▼──┐  ┌─▼─────────────┐
                │ PostgreSQL +    │ │ Object storage               │  │ ML package    │
                │ PostGIS         │ │ local FS (dev) / S3 (cloud)  │  │ PyTorch, MPS/ │
                │ users, jobs,    │ │ imagery COGs, masks, model   │  │ CUDA/CPU      │
                │ models, results,│ │ weights, reports             │  └───────────────┘
                │ flood polygons  │ └──────────────────────────────┘
                └─────────────────┘
        Offline: ml/ training scripts → MLflow (local mlruns/) → registered model artifacts
```

## Repository layout

| Path | Purpose |
|---|---|
| `backend/` | FastAPI app (`main.py`), `api/routes`, `core` (config, logging, security), `db` (engine, Alembic migrations), `models` (ORM), `schemas` (Pydantic), `services`, `workers` (Celery) |
| `ml/` | Datasets, model definitions, training/evaluation loops, configs. Importable by workers for inference |
| `geospatial/` | Raster/vector utilities: I/O, CRS alignment, tiling, polygonisation, exposure analysis |
| `frontend/` | Vite + React app |
| `infrastructure/` | Dockerfiles, Terraform (Phase 9) |
| `scripts/` | Env check, dev runner, dataset download helpers |
| `tests/` | `backend/`, `ml/`, `geospatial/` — pytest |
| `data/` | Git-ignored. Raw datasets, processed chips, storage root |

## Key decisions

| Decision | Choice | Why |
|---|---|---|
| Python version | 3.12 (via conda env `terrasentinel` + project `.venv`) | System Python is 3.14 (Anaconda); geospatial + torch wheels are most mature on 3.12 |
| Packaging | One `pyproject.toml`, extras `api` / `ml` / `geo` / `dev` | API container stays slim (no torch); workers add `ml,geo` |
| DB | PostgreSQL 16 + PostGIS 3.4; SQLite fallback for zero-dependency dev/tests | Spatial joins for exposure analysis belong in PostGIS. The app also boots without Docker. Geometry-bearing features will require Postgres |
| Migrations | Alembic, hand-checked; a test asserts migrations == ORM models; autogenerate filtered to our tables | Prevents schema drift and accidental DROPs of PostGIS extension tables |
| Container runtime (dev) | Colima (VZ) on macOS | Open source, no admin rights; containers have no MPS, so training stays native |
| Auth | OAuth2 password flow → JWT (HS256, 60 min), Argon2id via `pwdlib` | Standard, stateless; first registered user becomes admin |
| Secrets | `TS_SECRET_KEY` required in production; dev generates an ephemeral key | Fail closed in prod, zero setup in dev |
| Background jobs | Celery + Redis, `acks_late`, `prefetch=1`, time limits | ML jobs are long and memory-heavy; redeliver on worker crash |
| Large files | Never in DB rows. Stored as COGs in storage root / S3; DB keeps URI + metadata | DB stays small; COGs allow range reads for tiling |
| Logging | structlog, request-ID middleware, JSON in prod | Correlate API ↔ worker logs |
| Frontend data | React Query against the API only; pages for unbuilt features show an explicit "Phase N" notice | No hardcoded metrics |
| Map | MapLibre GL v6 (lazy-loaded chunk); `optimizeDeps.exclude` because its worker resolves relative to its module | Open source, no token |
| Device | `cuda` → `mps` → `cpu` auto-selection | Apple Silicon has no CUDA |

## Data model (current + planned)
- `users` ✅
- `datasets`, `images` (footprint `geometry(Polygon, 4326)`, CRS, resolution, storage URI) — Phase 2/5
- `models` (name, version, task, metrics JSON, artifact URI, sha256) — Phase 5
- `analysis_jobs` (status, progress, error, retries, input/output URIs) — Phase 5
- `flood_polygons`, `exposed_assets` (PostGIS geometries) — Phase 6

## Data pipeline (Sen1Floods11)
```
gs://sen1floods11 ──HTTPS+MD5──► data/raw/sen1floods11 ──prepare──► data/processed/sen1floods11
                                   (MANIFEST.json)          │           splits/{official,geo_holdout_v1}.json (sha256)
                                                            │           splits/loeo/<event>.json
                                                            │           norm_stats_<scheme>.json (train only)
                                                            └─► docs/data/sen1floods11_report.md
Sen1Floods11Dataset: read S1 dB → clip [-50,5] → standardise → NaN→0 ; label -1 or S1 NaN → 255 (ignore)
```
Split files carry a hash of their chip lists; loading a modified file fails. Chips that fail validation
(e.g. all-NaN S1) are excluded from every scheme and listed in the split metadata.

## Evaluation protocol (ML)
- Report on both the official splits (comparability with literature; note 89 usable test chips) and `geo_holdout_v1` (entire events unseen during training and early stopping) to measure generalisation and avoid spatial leakage from neighbouring chips.
- Normalisation statistics computed on the training split only.
- Metrics on the flood class with nodata pixels ignored; report per-event results, not just pooled numbers.
