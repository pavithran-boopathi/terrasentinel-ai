"""Alembic migrations must apply cleanly and match the ORM models."""

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect

import backend.models  # noqa: F401
from backend.core.config import REPO_ROOT, get_settings
from backend.db.base import Base
from backend.db.migration_filters import include_object


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
        diff = compare_metadata(
            MigrationContext.configure(conn, opts={"include_object": include_object}), Base.metadata
        )
    assert diff == [], f"models and migrations differ: {diff}"


def test_filter_ignores_foreign_extension_tables(tmp_path):
    """Tables we don't own (e.g. PostGIS/TIGER) must never appear in autogenerate diffs."""
    engine = create_engine(f"sqlite:///{tmp_path / 'f.db'}")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE spatial_ref_sys (srid INTEGER PRIMARY KEY)")
        conn.exec_driver_sql("CREATE INDEX ix_srs ON spatial_ref_sys (srid)")
    with engine.connect() as conn:
        ctx = MigrationContext.configure(conn, opts={"include_object": include_object})
        assert compare_metadata(ctx, Base.metadata) == []
        unfiltered = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert any(d[0] == "remove_table" for d in unfiltered)  # proves the filter does the work
