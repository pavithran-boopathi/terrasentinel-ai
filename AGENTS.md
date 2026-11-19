# Instructions for AI coding agents (Claude Code etc.)

Read `PROJECT_PLAN.md`, `ARCHITECTURE.md` and the latest entry in `PROGRESS.md` before starting work.

## Ground rules
- **Never invent** metrics, accuracies, test outcomes or deployment status. Only report what was run, with the command.
- Flood **exposure** ≠ building **damage**. Keep them as separate models, endpoints, and UI labels.
- No fake/hardcoded data in product features. Unbuilt UI shows the `NotYetBuilt` notice.
- Work phase by phase (see plan). Finish and test a phase before starting the next.
- Ask the user before: downloading large datasets (check `df -h` first), pushing to GitHub, creating any cloud/paid resource, deleting data.
- Update `PROGRESS.md` (and `EXPERIMENTS.md` for ML runs) at the end of each milestone.

## Environment
- Python 3.12 lives in conda env `terrasentinel`; the project uses `.venv` built from it.
  `conda activate terrasentinel && source .venv/bin/activate`
- Node 22 is also in the `terrasentinel` conda env (no Homebrew on this machine).
- Apple Silicon: use `mps`, never assume CUDA. Device selection: `backend/services/compute.py`.
- Docker = Colima in the `terrasentinel` conda env (`colima start`), see `docs/docker-macos.md`. Without Docker the API falls back to SQLite at `data/terrasentinel-dev.db`.
- Containers have **no MPS**: run training natively in `.venv`.
- New Alembic migrations: autogenerate is filtered by `backend/db/migration_filters.py` so PostGIS/TIGER tables are never dropped.

## Commands
| Task | Command |
|---|---|
| Backend tests | `pytest` |
| Lint / format | `ruff check . && ruff format --check .` |
| Frontend checks | `cd frontend && npm run typecheck && npm run lint && npm test && npm run build` |
| Migrate | `alembic upgrade head` |
| New migration | `alembic revision --autogenerate -m "..."` then review it; `tests/backend/test_migrations.py` must pass |
| Run locally | `./scripts/dev.sh` (API :8000, web :5173) |
| Env report | `python scripts/check_env.py` |
| Download Sen1Floods11 (hand-labeled S1) | `python scripts/download_sen1floods11.py` (ask first) |
| Build splits/stats/report | `python scripts/prepare_sen1floods11.py` |
| Train a config | `python -m ml.training.train --config ml/configs/<cfg>.yaml [--seed N]` |
| Grid of configs × seeds | `scripts/run_grid.sh "cfgA cfgB" "42 1 2"` |
| Baselines / LOEO / SN8 | `python scripts/run_baselines.py` · `python -m ml.training.loeo` · `python -m ml.training.train_spacenet8` |
| Results for docs + web app | `python scripts/export_experiments_json.py` (then rebuild the api image) |
| Register serving model | `python scripts/register_model.py --run data/checkpoints/<run> --name … --version … --default` |
| Uncertainty / latency | `python scripts/evaluate_uncertainty.py --runs …` · `python scripts/benchmark_latency.py` (quiet machine) |

## Conventions
- Settings via `TS_*` env vars (`backend/core/config.py`). Secrets only in `.env` (git-ignored).
- Large files (imagery, masks, weights) go to `TS_STORAGE_ROOT` / S3; DB stores URIs + metadata.
- Every new endpoint gets tests (happy path + auth + validation). Every geospatial util gets tests with tiny synthetic rasters — no network in tests.
- ML: always load splits with `load_splits_json` and stats from `norm_stats_<scheme>.json`; never recompute stats on val/test. Training scripts need `if __name__ == "__main__":` (macOS DataLoader workers use spawn).
- ML: fixed seeds, configs in `ml/configs/`, log to MLflow, splits written to disk and versioned by hash.
- Commits: small, descriptive, conventional style (`feat(backend): ...`). Do not push without approval.
- Commits and PRs are authored by the project owner only: **no `Co-Authored-By` trailers, no "Generated with" lines, no AI attribution** in commit messages or PR descriptions.
