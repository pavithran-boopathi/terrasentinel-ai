"""Restrict Alembic autogenerate to tables owned by our ORM models.

The PostGIS image installs extension tables (spatial_ref_sys, TIGER geocoder, topology). Without
this filter, ``alembic revision --autogenerate`` would emit DROP statements for them.
"""

from __future__ import annotations

from typing import Any

from backend.db.base import Base


def include_object(
    obj: Any, name: str | None, type_: str, reflected: bool, compare_to: Any
) -> bool:
    if type_ == "table":
        return name in Base.metadata.tables
    if type_ == "index" and name and name.endswith("_gist"):
        return False  # spatial indexes are created by hand in migrations (PostgreSQL only)
    table = getattr(obj, "table", None)
    if table is not None and type_ in {
        "index",
        "column",
        "unique_constraint",
        "foreign_key_constraint",
    }:
        return table.name in Base.metadata.tables
    return True
