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
- Docker is not installed (as of 2026-11-18); the API falls back to SQLite at `data/terrasentinel-dev.db`.

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

## Conventions
- Settings via `TS_*` env vars (`backend/core/config.py`). Secrets only in `.env` (git-ignored).
- Large files (imagery, masks, weights) go to `TS_STORAGE_ROOT` / S3; DB stores URIs + metadata.
- Every new endpoint gets tests (happy path + auth + validation). Every geospatial util gets tests with tiny synthetic rasters — no network in tests.
- ML: fixed seeds, configs in `ml/configs/`, log to MLflow, splits written to disk and versioned by hash.
- Commits: small, descriptive, conventional style (`feat(backend): ...`). Do not push without approval.
