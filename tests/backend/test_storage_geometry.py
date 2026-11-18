import io

import pytest
from shapely.geometry import Polygon, box
from sqlalchemy import create_engine, text
from sqlalchemy.dialects import postgresql, sqlite

from backend.db.types import Geometry
from backend.services.storage import LocalStorage, StorageError


def test_local_storage_blocks_path_traversal(tmp_path):
    s = LocalStorage(tmp_path / "root")
    with pytest.raises(StorageError):
        s.save_stream("../escape.txt", io.BytesIO(b"x"), 10)
    with pytest.raises(StorageError):
        s.local_path("local://../../etc/passwd")
    with pytest.raises(StorageError):
        s.local_path("s3://bucket/key")


def test_local_storage_size_cap_removes_partial(tmp_path):
    s = LocalStorage(tmp_path)
    with pytest.raises(StorageError):
        s.save_stream("a/b.bin", io.BytesIO(b"x" * 100), 10)
    assert not list(tmp_path.rglob("*.bin*"))
    uri, sha, n = s.save_stream("a/c.bin", io.BytesIO(b"hello"), 10)
    assert n == 5 and s.local_path(uri).read_bytes() == b"hello" and len(sha) == 64


def test_geometry_type_roundtrip_sqlite():
    g = Geometry("POLYGON")
    poly = box(0, 0, 1, 1)
    d = sqlite.dialect()
    assert g.process_bind_param(poly, d).startswith("POLYGON")
    assert g.process_result_value(poly.wkt, d).equals(poly)
    with pytest.raises(TypeError):
        g.process_bind_param("POLYGON((0 0,1 0,1 1,0 0))", d)


def test_geometry_type_postgres_binds_ewkt_and_reads_wkb():
    g = Geometry("POLYGON", 4326)
    d = postgresql.dialect()
    poly = Polygon([(0, 0), (1, 0), (1, 1)])
    assert g.process_bind_param(poly, d) == f"SRID=4326;{poly.wkt}"
    assert g.process_result_value(poly.wkb, d).equals(poly)
    assert "geometry(POLYGON, 4326)" in str(g.load_dialect_impl(d).compile(dialect=d))


def test_geometry_column_persists_on_sqlite(tmp_path):
    from sqlalchemy import Column, Integer, MetaData, Table

    md = MetaData()
    t = Table("g", md, Column("id", Integer, primary_key=True), Column("geom", Geometry()))
    e = create_engine(f"sqlite:///{tmp_path}/g.db")
    md.create_all(e)
    with e.begin() as c:
        c.execute(t.insert(), {"id": 1, "geom": box(1, 2, 3, 4)})
        assert c.execute(text("select geom from g")).scalar().startswith("POLYGON")
        assert c.execute(t.select()).first().geom.bounds == (1, 2, 3, 4)
