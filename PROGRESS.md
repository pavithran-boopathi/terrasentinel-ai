# Progress

## 2026-11-18 — Published to GitHub, CI green
- Project moved to `~/Desktop/terrasentinel/terrasentinel-ai`; `.venv`, `node_modules` rebuilt; all tests re-run there
- Public repo: https://github.com/pavithran-boopathi/terrasentinel-ai (pushed with approval)
- First CI run failed: `backend` job lacked numpy for the Phase 2 tests → added a `data-ml` job (geo extras + CPU torch)
- CI run 44393784983: **backend, backend-postgres (migrations on PostGIS), data-ml, frontend — all success**

## 2026-11-18 — Phase 2 (satellite data engineering) complete
### Data
- Sen1Floods11 v1.1 hand-labeled subset downloaded with approval: S1Hand, LabelHand, JRCWaterHand, S1OtsuLabelHand, split CSVs, metadata — 1,789 files, 706 MB, all MD5-verified (`scripts/download_sen1floods11.py`, manifest in `data/raw/sen1floods11/MANIFEST.json`)
- License check: **none declared** (GitHub repo has no LICENSE; bucket and STAC catalog have no license field). Only a third-party HF mirror says CC-BY-4.0. Treated as research use with citation; data not redistributed
- Sentinel-2 chips deliberately not downloaded (user choice)

### Code
- `geospatial/raster.py`: raster info, validation (structure, CRS, values, nodata), grid-alignment check, reprojection onto a reference grid, COG writer, footprint, pixel area (geodesic for EPSG:4326)
- `geospatial/viz.py`: SAR false colour, mask overlay, nodata hatching
- `geospatial/stac.py`: Sentinel-1 GRD search on Earth Search (live query verified: 3 scenes over Bolivia, Feb 2018)
- `ml/data/sen1floods11.py`: chip index, official + event-holdout + leave-one-event-out splits, hashed split files with tamper detection, leakage checks, train-only normalisation, ignore-index encoding, PyTorch Dataset
- `scripts/prepare_sen1floods11.py` → splits, norm stats, validation, quick-looks, [docs/data/sen1floods11_report.md](docs/data/sen1floods11_report.md)

### Findings from the real data
- **Paraguay_34417** (official *test*) has an all-NaN S1 chip while its label has ~50k water px → excluded everywhere; official test = 89 usable chips (not 90)
- **Bolivia_233925** has 387 px below −50 dB (min −103.9) → handled by clipping to [−50, 5] dB
- Official splits share all 10 non-Bolivia events between train/val/test → added `geo_holdout_v1` (test: Bolivia, Mekong, Ghana · val: Pakistan, Spain · train: 289 chips)
- Water is 4.6–24.1% of labelled px per event; ignored px up to 33.9% (Ghana)

### Tests (run 2026-11-18)
- `pytest`: **49 passed** (backend 21, geospatial 18, ml 10 — all on synthetic rasters, no network); ruff clean
- Real-data smoke test: geo_holdout_v1 train (289 chips) through DataLoader (4 workers) onto MPS in 22 s; normalised mean −0.005 / −0.005

### Bugs found & fixed
- `reproject_to_match` treated 0 as nodata for integer rasters without nodata, turning every "not water" pixel into water → never invent nodata for ints; explicit `dst_nodata` param; regression tests
- DataLoader workers on macOS use `spawn` → training entry points need `if __name__ == "__main__":`

### Next (Phase 3)
Otsu + pixel-classifier baselines, then U-Net (BCE+Dice, ignore index 255) on `geo_holdout_v1` and `official`, MLflow tracking, real metrics into EXPERIMENTS.md. Consider caching chips as `.npy` if loading (≈22 s/epoch) becomes the bottleneck.

## 2026-11-18 — Docker runtime installed & full stack verified
- Installed Colima 0.9.1 + Docker CLI 29.9 + Compose from conda-forge (no Homebrew/admin needed); VM 4 CPU / 8 GB / 60 GB
- Fixed: conda-forge `limactl` lacked the macOS virtualization entitlement → ad-hoc signed (docs/docker-macos.md)
- `docker compose up --build`: db, redis, api, worker, web all **healthy**
- Verified on PostgreSQL 16 + PostGIS 3.4: `alembic upgrade head`, full `downgrade base` → `upgrade head` cycle, PostGIS extension present, ORM ↔ schema diff empty
- Verified Celery task round-trip through Redis to the worker container
- Verified in browser: containerised web → API → Postgres; register / login / me
- **Bugs found & fixed**
  - Postgres healthcheck passed against the image's socket-only init server → API raced and crashed. Now probes TCP (compose + CI)
  - Alembic autogenerate would have proposed dropping 37 PostGIS/TIGER tables → `include_object` filter + regression test
  - Worker inherited the API's HTTP healthcheck → now `celery inspect ping`
- Tests: backend **21 passed**, frontend 7 passed, ruff clean
- Note: containers report device = CPU (no MPS in Linux VMs on macOS) — training runs natively

## 2026-11-18 — Phase 1 complete

### Environment (measured by `scripts/check_env.py`)
| Item | Value |
|---|---|
| OS / arch | macOS 27.2.1, arm64 (Apple M5 Pro, 18 cores) |
| Memory / free disk | 24 GB / ~828 GB |
| Python | 3.12.16 (conda env `terrasentinel`) → project `.venv` |
| Node | 22.23.5 (same conda env) |
| PyTorch | 2.15.0, **MPS available** (2048² matmul ×20 in 0.198 s) |
| Docker / Postgres / Redis / Homebrew | **not installed** |

### Done
- Repo skeleton, `pyproject.toml` with extras, `.gitignore`, `.env.example`, `Makefile`
- Backend: settings, structlog + request IDs, `/health`, `/health/ready`, `/system/compute`, register / login / me (JWT + Argon2), users table + Alembic migration, Celery app + diagnostic task
- Frontend: Vite 8 / React 19 / TS 5.9 / Tailwind 4, layout + six routed pages, live dashboard (API, DB, compute device), MapLibre basemap on Explorer, sign-in / register, protected route
- Docker Compose (PostGIS, Redis, API, worker, web), Dockerfiles, GitHub Actions CI (backend lint+tests, migrations on real PostGIS, frontend typecheck/lint/tests/build)

### Test results (run locally, 2026-11-18)
- Backend `pytest`: **20 passed**
- `ruff check` + `ruff format --check`: clean
- Frontend `vitest`: **7 passed**; `tsc` typecheck, ESLint, `vite build`: clean
- Manual end-to-end in browser: dashboard shows live API data (device = MPS); register → login → protected page works; Explorer map renders

### Not verified
- (CI verified later the same day — see top entry)

### Issues found & fixed
- MapLibre v6 worker failed to load under Vite's dependency pre-bundling → excluded from `optimizeDeps`
- `.test` email domains are rejected by `email-validator` (reserved TLD) — intended; UI now shows readable validation messages
- `vite.config.ts` needed `@types/node`

### Next (Phase 2)
1. Check Sen1Floods11 license, structure and size; ask before downloading
2. Install `.[geo]` extras and build the raster validation / dataset pipeline
