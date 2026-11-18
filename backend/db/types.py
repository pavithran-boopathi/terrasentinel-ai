"""Portable geometry column: PostGIS ``geometry(…, 4326)`` on PostgreSQL, WKT text elsewhere.

Python values are Shapely geometries in EPSG:4326. On PostgreSQL the column is a real PostGIS
geometry (GiST-indexable, usable in ST_Intersects etc.); on SQLite (dev/tests) it is stored as
WKT so the same models work without SpatiaLite.
"""

from __future__ import annotations

import re
from typing import Any

from shapely import wkb, wkt
from shapely.geometry.base import BaseGeometry
from sqlalchemy import Text
from sqlalchemy.types import TypeDecorator, UserDefinedType

_HEX = re.compile(r"[0-9A-Fa-f]+")


class _PostGISGeometry(UserDefinedType):
    cache_ok = True

    def __init__(self, geometry_type: str = "GEOMETRY", srid: int = 4326) -> None:
        self.geometry_type = geometry_type
        self.srid = srid

    def get_col_spec(self, **kw: Any) -> str:
        return f"geometry({self.geometry_type}, {self.srid})"

    def bind_expression(self, bindvalue):
        from sqlalchemy import func

        return func.ST_GeomFromEWKT(bindvalue)


class Geometry(TypeDecorator):
    impl = Text
    cache_ok = True

    def __init__(self, geometry_type: str = "GEOMETRY", srid: int = 4326) -> None:
        super().__init__()
        self.geometry_type = geometry_type
        self.srid = srid

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(_PostGISGeometry(self.geometry_type, self.srid))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value: BaseGeometry | None, dialect) -> str | None:
        if value is None:
            return None
        if not isinstance(value, BaseGeometry):
            raise TypeError(f"expected a shapely geometry, got {type(value).__name__}")
        if dialect.name == "postgresql":
            return f"SRID={self.srid};{value.wkt}"
        return value.wkt

    def process_result_value(self, value: Any, dialect) -> BaseGeometry | None:
        if value is None:
            return None
        if isinstance(value, bytes | memoryview):
            return wkb.loads(bytes(value))
        if isinstance(value, str):
            if _HEX.fullmatch(value):  # PostGIS default output: hex-encoded EWKB
                return wkb.loads(value, hex=True)
            text = value.split(";", 1)[1] if value.startswith("SRID=") else value
            return wkt.loads(text)
        raise TypeError(f"unexpected geometry value {type(value).__name__}")
