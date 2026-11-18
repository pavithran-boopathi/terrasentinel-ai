# TerraSentinel AI — Project Plan

Satellite-powered disaster intelligence: flood segmentation from Sentinel-1 SAR, exposure of
roads/buildings to predicted flooding, and (separately) building damage classification from
pre/post imagery — served through a FastAPI + React application.

**Guiding rules**
- Every metric in this repo comes from a run that is logged in `EXPERIMENTS.md` / MLflow. No invented numbers.
- *Flood exposure* (a building intersects a predicted flood mask) is never reported as *damage*.
  Damage classification is a separate model (xBD) with its own evaluation.
- Develop on the Mac first (MPS, small subsets); design every training script to scale to a cloud GPU unchanged.
- Each phase ends with passing tests and an updated `PROGRESS.md`.

## Phase 1 — Project foundation ✅ (2026-11-18)
- Git repo, directory layout, `pyproject.toml` with optional extras (`api`, `ml`, `geo`, `dev`)
- FastAPI skeleton: settings, structured logging, request IDs, health/readiness/compute endpoints
- JWT auth (Argon2 hashes), users table, Alembic migrations (PostGIS extension on Postgres)
- Celery app (Redis broker) with a diagnostic task
- React + TypeScript + Tailwind + React Query + MapLibre skeleton with all six pages routed
- Docker Compose (PostGIS, Redis, API, worker, web), GitHub Actions CI, env checker script
- Docs: this plan, ARCHITECTURE, PROGRESS, EXPERIMENTS, AGENTS

## Phase 2 — Satellite data engineering
- Verify Sen1Floods11 license, layout and size; download the **hand-labeled** split only (approval required before download)
- `geospatial/`: raster I/O (rasterio), CRS/transform validation, nodata handling, reprojection & alignment, COG writing, footprint → GeoJSON
- `ml/data/`: Sen1Floods11 dataset class (VV/VH, dB scaling, per-band normalisation stats computed on train only), label handling (`-1` = no data → ignore index)
- Reproducible splits: use the official split **and** a geographic hold-out split (by flood event/country)
- Data validation report (shape, CRS, value ranges, class balance) + visualisation utilities
- STAC client helper (Planetary Computer / Earth Search) for pulling a Sentinel-1 scene by bbox/date
- Tests with tiny synthetic GeoTIFF fixtures (no network in CI)

## Phase 3 — First model: U-Net baseline
- Baselines first: Otsu threshold on VH dB, and a pixel-wise logistic regression / random forest
- U-Net (ResNet-34 encoder, 2-channel input), loss = BCE + Dice (or Lovász) with ignore-index
- Albumentations (flips, rot90, small affine, speckle-like noise), mixed precision where supported
- Early stopping on val IoU, checkpointing, seed control, MLflow logging (local `mlruns/`)
- Evaluation on held-out test + Bolivia hold-out: IoU, Dice, precision, recall, F1, latency (MPS/CPU)
- Results → `EXPERIMENTS.md`, model card in `docs/models/`

## Phase 4 — SegFormer
- Fine-tune SegFormer (MiT-b0/b1, Hugging Face) with adapted 2-channel stem; same splits, losses, metrics
- Paired comparison vs U-Net (per-chip IoU, bootstrap CIs), parameter count, latency, memory
- Decide serving default from evidence

## Phase 5 — Production inference
- Sliding-window tiled inference with overlap + cosine/Hann blending, arbitrary-size GeoTIFF in → georeferenced COG mask out
- Model registry table (name, version, metrics, artifact URI, checksum); TorchScript/ONNX export where it pays off
- Upload API (size/type/CRS validation, streaming to storage), `analysis_jobs` table, Celery task with retries, progress reporting, failure recovery
- Polygonised flood extent stored in PostGIS

## Phase 6 — Geospatial intelligence (SpaceNet 8)
- Ingest SpaceNet 8 building footprints + road network (verify license and S3 access first)
- Road/building segmentation model **or** use labelled/OSM vectors where models are not justified — document the choice
- Exposure analysis: spatial joins of buildings/roads with flood polygons in PostGIS (counts, km of road, % area), in a projected equal-area CRS
- Evaluate against SpaceNet 8 flooded/non-flooded labels

## Phase 7 — Interactive web application
- Explorer: imagery tiles (COG via TiTiler or pre-rendered), Deck.gl overlays, region selection, before/after swipe
- Analysis: upload → model choice → job progress → results; Impact page; Model Lab charts (Plotly) from API; Reports (PDF, GeoJSON, GeoTIFF, CSV)

## Phase 8 — Advanced research
- Geographic generalisation: leave-one-event-out CV
- Uncertainty: MC dropout / test-time augmentation; calibration (ECE, reliability diagrams, temperature scaling)
- Multitemporal change detection (pre/post SAR differencing vs learned model)
- xBD building damage model (Siamese / two-stream), evaluated separately (xBD license is non-commercial — check)
- Error analysis by land cover, event, and label quality

## Phase 9 — MLOps & cloud (optional, needs approval for any paid resource)
- MLflow model registry, CI for model smoke tests, Prometheus-style metrics, CloudWatch-ready logs
- Terraform for S3 + ECS (Fargate) + RDS PostGIS + ElastiCache; GPU training on a spot instance — **plan only until approved**

## Phase 10 — Final evaluation
- Benchmark table, failure-case gallery, architecture diagram, demo script, evidence-backed resume bullets

## Compute strategy
| Work | Where | Notes |
|---|---|---|
| Data prep, unit tests, small training runs | Mac (M5 Pro, 24 GB, MPS) | batch 4–8 at 512², subsets |
| Full U-Net / SegFormer training | Mac first; cloud GPU if too slow | same script, `--device cuda` |
| xBD damage model (large) | Cloud GPU likely | decide after Phase 4 timings |
