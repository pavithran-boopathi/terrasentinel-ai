# Architecture

## System overview

```
                 ┌───────────────────────────── Browser ─────────────────────────────┐
                 │ React 19 + TS + Tailwind · React Query · MapLibre (WGS84 previews) │
                 │ Plotly (lazy) · pages: Dashboard, Explorer, Analysis, Impact,      │
                 │ Model Lab, Reports                                                 │
                 └───────────────┬────────────────────────────────────────────────────┘
                                 │ /api/v1 (JWT bearer)
        ┌────────────────────────▼─────────────────────────┐   enqueue    ┌─────────────────────────────┐
        │ FastAPI (api image, no torch)                    ├────Redis────►│ Celery worker (CPU torch)   │
        │ auth · images (validated upload, footprint) ·    │              │ tiled inference → COGs,     │
        │ models (registry) · analyses (jobs, previews,    │◄──status─────┤ polygons, change detection, │
        │ exposure, CSV/PDF) · monitoring · experiments    │   via DB     │ retries / permanent failure │
        └───────┬───────────────────────────┬──────────────┘              └──────────────┬──────────────┘
                │ SQLAlchemy                │ Storage protocol                           │
        ┌───────▼────────────┐   ┌──────────▼──────────────────────────┐                 │
        │ PostgreSQL+PostGIS │   │ LocalStorage (dev) │ S3Storage (AWS) │◄────────────────┘
        │ users, model_      │   │ images/, outputs/<job>/, models/    │
        │ versions, images   │   └─────────────────────────────────────┘
        │ (footprint GiST),  │
        │ analysis_jobs      │   Offline (native macOS, MPS): ml/ training → data/checkpoints/<run>/
        │ (flood_extent GiST)│   results.json + MLflow (sqlite) → scripts/register_model.py → registry
        └────────────────────┘
```

## Repository layout

| Path | Purpose |
|---|---|
| `backend/` | FastAPI app: `api/routes` (auth, images, models, analyses, monitoring, experiments), `core` (settings, logging, JWT/Argon2), `db` (engine, portable PostGIS `Geometry` type, Alembic migrations 0001–0003, autogenerate filter), `models`, `schemas`, `services` (storage, reports, compute), `workers` (Celery app + analysis task) |
| `ml/data/` | Sen1Floods11 (splits, norm stats, Dataset), SpaceNet 8 (warp, rasterise, spatial split, object metrics), xBD (labels, xView2 metric) |
| `ml/models/` | Own U-Net; factory for ResNet-34 U-Net (smp) and SegFormer B0/B2 (HF) with 2-band input adaptation |
| `ml/training/` | Generic trainer (`train.py`), SpaceNet 8 and xBD trainers, LOEO cross-validation, losses, augmentation, engine |
| `ml/evaluation/` | Pooled/per-event metrics with chip-bootstrap CIs, calibration, MC-dropout/TTA uncertainty |
| `ml/inference/` | Hann-blended sliding-window inference, `FloodPredictor` (GeoTIFF in → COGs/GeoJSON out) |
| `ml/baselines/` | Otsu, global threshold, gradient-boosted pixel classifier |
| `geospatial/` | Raster validation/alignment/COG, S1 input contract, previews, exposure analysis, change detection, STAC search |
| `frontend/` | React app |
| `infrastructure/` | Two-target Dockerfile (api / worker), Terraform AWS stack (validated, not applied) |
| `scripts/` | Download, prepare, baselines, grid runner, model registration, experiments export, error analysis, uncertainty, latency |

## Key decisions

| Decision | Choice | Why |
|---|---|---|
| Python | 3.12 (conda env `terrasentinel` → `.venv`) | Mature wheels for torch + GDAL stack |
| Packaging | One `pyproject.toml`; extras `api`, `geo`, `ml`, `ml-research`, `torchgeo`, `dev` | API image has no torch; worker adds CPU torch; TorchGeo optional (its Lightning extras need `jsonnet`, no ARM64 wheel) |
| Geometry column | `backend.db.types.Geometry`: PostGIS `geometry(…, 4326)` on Postgres (GiST indexes), WKT on SQLite | Real spatial types in production; zero-dependency tests. A real-PostGIS round-trip test runs in CI |
| Large files | Never in DB rows. COGs/GeoJSON/weights in storage; DB has URIs + metadata | DB stays small; COGs allow range reads |
| Storage | Protocol with Local and S3 backends; S3 uses a local cache so GDAL/torch see paths; `commit_dir` uploads outputs | Same code locally and on AWS |
| Inference | 512² tiles, 64 px overlap, Hann blending with a zero-weight band at interior tile edges; memmap accumulators | Seamless output (matches whole-image inference to 1e-5 in tests); bounded memory for any scene size |
| Jobs | Celery, `acks_late`, prefetch 1, time limits; transient errors retried with backoff, bad input/missing artifacts fail immediately; admin endpoint requeues stale `running` jobs | ML jobs are long and memory-heavy |
| Upload security | Extension allow-list, server-side names, streamed size cap, GDAL open + band/dtype/CRS/size checks, traversal-safe storage keys, ownership checks on every resource | Untrusted GeoTIFFs |
| Exposure | Overlap of buildings/roads with predicted water in local UTM metres; labelled "potential exposure — not damage" everywhere | Prevents conflating flood extent with structural damage |
| Damage | Separate xBD model/task (pre+post optical, xView2 metric); never consumes flood masks | Different physics and labels |
| Change detection | Model run on pre (warped to post grid) and post; new/receded water; unsupervised log-ratio rule reported alongside | Model-based change is interpretable; the rule is an independent sanity check |
| Device | `cuda` → `mps` → `cpu`; training native on macOS (containers have no MPS) | Apple Silicon |
| Frontend data | React Query against the API only; Model Lab reads `docs/results/experiments.json` generated from `results.json` | No hand-typed metrics in the UI |
| Map | MapLibre + backend PNG previews warped to WGS84 (no tile server) | Simple, auth-compatible; TiTiler would be the next step for large scenes |
| Deck.gl | Not used | MapLibre's native GeoJSON layers handle the feature counts involved; Deck.gl adds bundle size without benefit here |
| Cloud | Terraform for ECS Fargate (ARM64), RDS, ElastiCache, S3, ALB, CloudWatch; **not applied** | Cost control; deploy is an explicit owner decision |

## Data model
- `users`
- `model_versions` — name, version, arch, artifact URI, sha256, threshold, metrics + training provenance (copied from `results.json`), `is_default`
- `satellite_images` — owner, storage URI, sha256, size, CRS, dimensions, `footprint geometry(POLYGON,4326)` + GiST
- `analysis_jobs` — owner, image, optional pre-event image, model version, status/progress/attempts/error, timings, summary JSON (incl. change and exposure), output prefix, `flood_extent geometry(MULTIPOLYGON,4326)` + GiST

## Evaluation protocol (ML)
- Sen1Floods11: official splits (literature comparability; Paraguay_34417 excluded → 89 test chips) **and** `geo_holdout_v1` (whole events held out) **and** leave-one-event-out CV with a fixed epoch budget (no held-out-event model selection).
- Normalisation statistics from training chips only (recomputed per LOEO fold). Ignore index for label −1 and S1 NaN.
- Flood-class IoU/F1/precision/recall pooled over pixels; 95% CIs by bootstrapping chips; 3 seeds per configuration; per-event breakdowns.
- SpaceNet 8: spatial block split by tile-grid column; per-class pixel IoU and object-level flood attribution (per building; length-weighted along true road centrelines) with oracle-geometry reference baselines.
- Calibration (ECE, Brier, NLL) before/after temperature scaling fitted on validation; uncertainty judged by its AUROC for detecting errors and by risk–coverage curves.
