"""Alembic migrations must apply cleanly and match the ORM models."""

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect

import backend.models  # noqa: F401
from backend.core.config import REPO_ROOT, get_settings
from backend.db.base import Base


def test_upgrade_head_matches_models(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    monkeypatch.setenv("TS_DATABASE_URL", url)
    get_settings.cache_clear()
    try:
        command.upgrade(Config(str(REPO_ROOT / "alembic.ini")), "head")
    finally:
        get_settings.cache_clear()
    engine = create_engine(url)
    assert "users" in inspect(engine).get_table_names()
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == [], f"models and migrations differ: {diff}"
