# Progress

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
- GitHub Actions CI has not run yet (nothing pushed)

### Issues found & fixed
- MapLibre v6 worker failed to load under Vite's dependency pre-bundling → excluded from `optimizeDeps`
- `.test` email domains are rejected by `email-validator` (reserved TLD) — intended; UI now shows readable validation messages
- `vite.config.ts` needed `@types/node`

### Next (Phase 2)
1. Check Sen1Floods11 license, structure and size; ask before downloading
2. Install `.[geo]` extras and build the raster validation / dataset pipeline
