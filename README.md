# TerraSentinel AI

**Satellite-powered flood intelligence:** Sentinel-1 SAR flood segmentation (U-Net, ResNet-34 U-Net,
SegFormer-B0/B2), multitemporal change detection, calibrated uncertainty, building/road flood
**exposure** analysis, and a separate xBD building-**damage** pipeline — served through FastAPI,
PostGIS, Celery and a React/MapLibre web app, with reproducible experiments and an AWS Terraform stack.

> Every metric in this repository was produced by code in this repository and is traceable to a
> `results.json` (see [EXPERIMENTS.md](EXPERIMENTS.md)). Nothing is copied from papers.

## Results at a glance (Sen1Floods11, hand-labelled S1, pooled flood IoU, mean ± std over 3 seeds)

| Model | Params | Official test (89 chips) | Bolivia — unseen event (15) | Whole events held out (98) |
|---|---|---|---|---|
| Best classical baseline (gradient-boosted pixels) | — | 0.639 | 0.438 | 0.664 |
| U-Net, from scratch | 7.8 M | 0.645 ± 0.007 | 0.643 ± 0.083 | 0.676 ± 0.040 |
| U-Net, ImageNet ResNet-34 | 24.4 M | 0.648 ± 0.010 | 0.672 ± 0.043 | 0.670 ± 0.042 |
| SegFormer-B0 | 3.7 M | 0.652 ± 0.013 | **0.711 ± 0.022** | 0.652 ± 0.042 |
| SegFormer-B2 | 27.3 M | **0.658 ± 0.005** | 0.708 ± 0.028 | **0.700 ± 0.048** |

What the evidence says (details in EXPERIMENTS.md):
- Deep models beat simple SAR thresholds clearly (Otsu 0.23, global threshold 0.53) but only narrowly beat a strong pixel classifier in-distribution.
- Their real advantage is **generalisation to unseen floods**: on Bolivia, pretrained SegFormers reach ~0.71 vs 0.44 for the pixel classifier, with low seed variance.
- **Training-event diversity matters more than architecture**: the same unseen Bolivia event scores 0.64–0.71 when models train on 10 events, 0.30–0.50 when they train on 6.
- Errors are mostly **missed water** (precision ≈ 0.9, recall ≈ 0.7 on held-out events) — flooded vegetation and narrow channels that are not dark in SAR.
- **Leave-one-event-out CV** (11 folds, no model selection on the held-out event): U-Net ResNet-34 averages 0.576 ± 0.184 IoU vs 0.516 for the pixel baseline and wins on 8/11 events; per-event difficulty (0.27 Pakistan … 0.83 Mekong) dwarfs architecture differences.
- **Infrastructure (SpaceNet 8 Germany):** an end-to-end deep model failed to learn building flood status on 131 tiles (F1 0.00 — reported as a negative result); given known geometry, an object-level classifier detects flooded roads well (F1 0.60, AUROC 0.87) and buildings only weakly (AUROC 0.68).
- **Speed:** SegFormer-B0 runs a 512² chip in 9.3 ms on the M5 Pro GPU; tiled inference covers a 4096² scene (≈1,680 km² at 10 m) in 1.3 s.
- Predictive uncertainty is useful: entropy detects misclassified pixels with AUROC 0.82–0.88; deferring the 10 % least certain pixels lifts IoU on the rest from ≈ 0.65 to ≈ 0.78.
- Data issues found and handled: an all-NaN Sentinel-1 chip in the official test split (excluded), extreme backscatter outliers (clipped), and event leakage in the official splits (event-held-out and leave-one-event-out protocols added).

## Features

| Area | What exists |
|---|---|
| Data | MD5-verified downloaders (Sen1Floods11, SpaceNet 8), raster validation, reproducible hashed splits, train-only normalisation |
| Models | U-Net (own), ResNet-34 U-Net, SegFormer-B0/B2 with 2-band SAR stem adaptation; classical baselines |
| Evaluation | Pooled + per-event IoU/F1/P/R, chip-bootstrap CIs, multi-seed, val-tuned thresholds, LOEO CV, calibration (ECE/Brier/NLL, temperature scaling), MC dropout & TTA uncertainty, error analysis |
| Inference | Seamless tiled inference for any scene size (Hann blending, memmap), COG + GeoJSON outputs, geodesic areas, change detection against a pre-event scene |
| Geospatial | PostGIS geometries + GiST indexes, building/road exposure in local UTM, WGS84 map previews, STAC search |
| Backend | FastAPI, JWT/Argon2 auth, validated uploads, model registry, Celery jobs with retries and failure recovery, monitoring, PDF/CSV reports, Local + S3 storage |
| Frontend | React + TypeScript + Tailwind + MapLibre + Plotly: Dashboard, Explorer (before/after), AI Analysis, Disaster Impact, Model Lab, Reports |
| Damage (separate task) | xBD pipeline + xView2 metric, tested on synthetic xBD-format data — **not trained** (dataset requires registration) |
| MLOps | MLflow tracking, 6-job GitHub Actions CI (incl. real PostGIS, Docker builds, Terraform validate), two-target Docker images, Terraform for AWS ECS/RDS/ElastiCache/S3/CloudWatch (**validated, not applied**) |

## Quick start (macOS, Apple Silicon)

```bash
conda create -n terrasentinel -c conda-forge --override-channels python=3.12 nodejs=22 colima lima-vm docker-cli docker-compose
conda activate terrasentinel
make setup && cp .env.example .env   # set TS_SECRET_KEY
```

**Full stack in Docker** (PostGIS, Redis, API, Celery worker, web) — see [docs/docker-macos.md](docs/docker-macos.md):
```bash
colima start && docker compose up -d --build     # http://127.0.0.1:5173
```

**Data and models** (training runs natively on MPS; containers have no Apple GPU):
```bash
python scripts/download_sen1floods11.py && python scripts/prepare_sen1floods11.py
python scripts/run_baselines.py
python -m ml.training.train --config ml/configs/segformer_b2_official.yaml
python scripts/register_model.py --run data/checkpoints/<run> --name flood-segformer-b2 --version 1 --default
```

**Tests:** `make test` (Python + frontend) · `make lint`.

## Documentation
[PROJECT_PLAN.md](PROJECT_PLAN.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [EXPERIMENTS.md](EXPERIMENTS.md) ·
[PROGRESS.md](PROGRESS.md) · [docs/data/sen1floods11_report.md](docs/data/sen1floods11_report.md) ·
[infrastructure/terraform/README.md](infrastructure/terraform/README.md) · [AGENTS.md](AGENTS.md)

## Limitations
- Flood maps are model predictions from SAR; flooded vegetation, urban flooding and narrow channels are often missed.
- "Potentially exposed" = overlap with predicted water — **not** damage. Damage assessment is a separate (untrained) xBD model.
- Test sets are small (15–98 chips); differences of ~0.01–0.03 IoU between deep models are within noise.
- Sen1Floods11 declares no license and SpaceNet 8 imagery is reported as non-commercial; both used for research only, not redistributed.

## Data credits
Bonafilia et al., *Sen1Floods11*, CVPRW 2020 · Hänsch et al., *SpaceNet 8*, CVPRW 2022 · Gupta et al., *xBD*, CVPRW 2019 ·
Copernicus Sentinel-1 data · JRC Global Surface Water.
