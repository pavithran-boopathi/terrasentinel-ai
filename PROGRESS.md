# Progress

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
- `docker compose up` — Docker is not installed on this Mac
- Alembic migration against real PostgreSQL/PostGIS — only SQLite locally; covered by the CI job, which has not run yet (nothing pushed)
- Celery with a real Redis broker — only eager mode tested

### Issues found & fixed
- MapLibre v6 worker failed to load under Vite's dependency pre-bundling → excluded from `optimizeDeps`
- `.test` email domains are rejected by `email-validator` (reserved TLD) — intended; UI now shows readable validation messages
- `vite.config.ts` needed `@types/node`

### Next (Phase 2)
1. Decide on Docker runtime (Docker Desktop, OrbStack or Colima) — optional, the app runs without it
2. Check Sen1Floods11 license, structure and size; ask before downloading
3. Install `.[geo]` extras and build the raster validation / dataset pipeline
