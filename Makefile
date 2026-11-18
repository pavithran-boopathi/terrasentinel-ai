.PHONY: setup dev test lint fmt check-env migrate

PYTHON ?= python3.12

setup:            ## Create venv and install backend + frontend deps (needs Python 3.12 + Node 22 on PATH)
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -e ".[api,dev]"
	cd frontend && npm ci

dev:              ## Run API + web locally (no Docker)
	./scripts/dev.sh

migrate:
	.venv/bin/alembic upgrade head

test:
	.venv/bin/pytest
	cd frontend && npm test

lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .
	cd frontend && npm run typecheck && npm run lint

fmt:
	.venv/bin/ruff format .
	.venv/bin/ruff check --fix .

check-env:
	.venv/bin/python scripts/check_env.py
