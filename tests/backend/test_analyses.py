"""End-to-end: upload → register model → analysis job (Celery eager) → outputs."""

import io
import json

import numpy as np
import pytest
import torch
from rasterio.transform import from_origin

from backend.core.config import get_settings
from backend.models.registry import AnalysisJob, JobStatus, ModelVersion
from backend.workers import tasks
from ml.models.unet import UNet
from tests.backend.conftest import API
from tests.geospatial.conftest import write_tif

PASSWORD = "correct-horse-battery"


@pytest.fixture
def env(client, db_sessionmaker, tmp_path, monkeypatch):
    monkeypatch.setenv("TS_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("TS_INFERENCE_DEVICE", "cpu")
    monkeypatch.setenv("TS_INFERENCE_TILE", "64")
    monkeypatch.setenv("TS_INFERENCE_OVERLAP", "8")
    get_settings.cache_clear()
    monkeypatch.setattr(tasks, "session_factory", db_sessionmaker)
    tasks._predictor.cache_clear()
    yield client, db_sessionmaker, tmp_path
    get_settings.cache_clear()


def token(client, email="ana@example.com"):
    client.post(f"{API}/auth/register", json={"email": email, "password": PASSWORD})
    t = client.post(f"{API}/auth/login", data={"username": email, "password": PASSWORD})
    return {"Authorization": f"Bearer {t.json()['access_token']}"}


def s1_bytes(tmp_path, bands=2, dtype=np.float32, name="s1.tif"):
    img = np.full((bands, 96, 128), -12.0, dtype)
    if np.dtype(dtype).kind == "f":
        img[:, :8] = np.nan
    p = write_tif(
        tmp_path / name, img, crs="EPSG:32633", transform=from_origin(500000, 4600000, 10, 10)
    )
    return p.read_bytes()


def register_tiny_model(sm, tmp_path, default=True):
    from backend.services.storage import get_storage, sha256_file

    m = UNet(2, 1, base_channels=4, depth=2)
    ckpt = tmp_path / "tiny.pt"
    torch.save(
        {
            "model": m.state_dict(),
            "config": {"arch": "unet", "base_channels": 4, "depth": 2, "dropout": 0.0},
            "meta": {
                "norm_stats": {
                    "mean": [-10, -17],
                    "std": [4, 4.5],
                    "clip": [-50, 5],
                    "n_pixels": 1,
                    "source": "t",
                },
                "run_name": "tiny",
            },
        },
        ckpt,
    )
    uri = get_storage().put_file("models/tiny/1/model.pt", ckpt)
    with sm() as db:
        mv = ModelVersion(
            name="tiny",
            version="1",
            arch="unet",
            artifact_uri=uri,
            sha256=sha256_file(ckpt),
            size_bytes=ckpt.stat().st_size,
            threshold=0.5,
            metrics={"test": {"iou": 0.1}},
            training={},
            is_default=default,
        )
        db.add(mv)
        db.commit()
        return str(mv.id)


def upload(client, h, data, name="scene.tif"):
    return client.post(
        f"{API}/images", headers=h, files={"file": (name, io.BytesIO(data), "image/tiff")}
    )


def test_full_analysis_flow(env):
    client, sm, tmp = env
    h = token(client)
    model_id = register_tiny_model(sm, tmp)
    assert client.get(f"{API}/models", headers=h).json()[0]["id"] == model_id

    r = upload(client, h, s1_bytes(tmp))
    assert r.status_code == 201, r.text
    img = r.json()
    assert img["footprint"]["type"] == "Polygon" and img["crs"] == "EPSG:32633"

    r = client.post(f"{API}/analyses", headers=h, json={"image_id": img["id"]})
    assert r.status_code == 202, r.text
    job = client.get(f"{API}/analyses/{r.json()['id']}", headers=h).json()
    assert job["status"] == "succeeded", job["error"]
    assert job["progress"] == 1.0 and job["attempts"] == 1
    assert set(job["files"]) == {
        "flood_mask.tif",
        "probability.tif",
        "flood_polygons.geojson",
        "summary.json",
    }
    s = job["summary"]
    assert s["pixels_valid"] == (96 - 8) * 128 and s["model"] == "tiny:1"

    f = client.get(f"{API}/analyses/{job['id']}/files/flood_polygons.geojson", headers=h)
    assert f.status_code == 200 and json.loads(f.content)["type"] == "FeatureCollection"
    assert (
        client.get(f"{API}/analyses/{job['id']}/files/../../etc/passwd", headers=h).status_code
        == 404
    )
    assert client.get(f"{API}/analyses/{job['id']}/files/secret.txt", headers=h).status_code == 404

    mon = client.get(f"{API}/monitoring/summary", headers=h).json()
    assert mon["jobs_by_status"] == {"succeeded": 1} and mon["succeeded_last_24h"] == 1


def test_other_users_cannot_see_images_or_jobs(env):
    client, sm, tmp = env
    h1, h2 = token(client), token(client, "bo@example.com")
    register_tiny_model(sm, tmp)
    img = upload(client, h1, s1_bytes(tmp)).json()
    job = client.post(f"{API}/analyses", headers=h1, json={"image_id": img["id"]}).json()
    assert client.get(f"{API}/images/{img['id']}", headers=h2).status_code == 404
    assert client.get(f"{API}/analyses/{job['id']}", headers=h2).status_code == 404
    assert (
        client.post(f"{API}/analyses", headers=h2, json={"image_id": img["id"]}).status_code == 404
    )
    assert client.get(f"{API}/analyses", headers=h2).json() == []
    assert client.get(f"{API}/images", headers=h1).json()[0]["id"] == img["id"]


def test_upload_requires_auth_and_rejects_bad_files(env, monkeypatch):
    client, _, tmp = env
    assert upload(client, {}, b"x").status_code == 401
    h = token(client)
    assert upload(client, h, b"x", name="evil.exe").status_code == 415
    assert upload(client, h, b"not a tiff").status_code == 422
    assert upload(client, h, s1_bytes(tmp, bands=3, name="b3.tif")).status_code == 422
    assert upload(client, h, s1_bytes(tmp, dtype=np.uint8, name="u8.tif")).status_code == 422
    monkeypatch.setenv("TS_MAX_UPLOAD_MB", "0")
    get_settings.cache_clear()
    assert upload(client, h, s1_bytes(tmp)).status_code == 413
    # rejected uploads leave nothing behind
    images_dir = tmp / "storage" / "images"
    assert not images_dir.exists() or not any(images_dir.rglob("*.tif"))


def test_analysis_without_default_model_is_404(env):
    client, _, tmp = env
    h = token(client)
    img = upload(client, h, s1_bytes(tmp)).json()
    assert (
        client.post(f"{API}/analyses", headers=h, json={"image_id": img["id"]}).status_code == 404
    )


def test_transient_failures_are_retried_then_marked_failed(env, monkeypatch):
    client, sm, tmp = env
    h = token(client)
    register_tiny_model(sm, tmp)
    img = upload(client, h, s1_bytes(tmp)).json()
    calls = []

    def boom(job_id):
        calls.append(job_id)
        raise RuntimeError("MPS out of memory")

    monkeypatch.setattr(tasks, "execute_analysis", boom)
    job = client.post(f"{API}/analyses", headers=h, json={"image_id": img["id"]}).json()
    assert len(calls) == 1 + get_settings().job_max_retries
    with sm() as db:
        j = db.get(AnalysisJob, __import__("uuid").UUID(job["id"]))
        assert j.status == JobStatus.failed and "MPS out of memory" in j.error


def test_permanent_failure_is_not_retried(env, monkeypatch):
    client, sm, tmp = env
    h = token(client)
    register_tiny_model(sm, tmp)
    img = upload(client, h, s1_bytes(tmp)).json()
    calls = []

    def bad(job_id):
        calls.append(job_id)
        raise tasks.PermanentJobError("model artifact missing")

    monkeypatch.setattr(tasks, "execute_analysis", bad)
    job = client.post(f"{API}/analyses", headers=h, json={"image_id": img["id"]}).json()
    assert len(calls) == 1
    assert client.get(f"{API}/analyses/{job['id']}", headers=h).json()["status"] == "failed"


def test_recover_stale_jobs_is_admin_only(env):
    client, _, _ = env
    admin = token(client)  # first user = admin
    user = token(client, "bo@example.com")
    assert client.post(f"{API}/monitoring/recover-stale-jobs", headers=user).status_code == 403
    assert client.post(f"{API}/monitoring/recover-stale-jobs", headers=admin).json() == {
        "requeued": 0
    }


def test_exposure_endpoint_on_finished_analysis(env, monkeypatch):
    client, sm, tmp = env
    h = token(client)
    register_tiny_model(sm, tmp)
    img = upload(client, h, s1_bytes(tmp)).json()
    job = client.post(f"{API}/analyses", headers=h, json={"image_id": img["id"]}).json()
    # Force a known flood extent: a 100 m square in the scene (UTM 33N → WGS84).
    import geopandas as gpd
    from shapely.geometry import LineString, MultiPolygon, box

    x0, y0 = 500100, 4599500
    flood = gpd.GeoSeries([box(x0, y0, x0 + 100, y0 + 100)], crs="EPSG:32633").to_crs(4326)
    with sm() as db:
        j = db.get(AnalysisJob, __import__("uuid").UUID(job["id"]))
        j.flood_extent = MultiPolygon([flood.iloc[0]])
        db.commit()
    bld = gpd.GeoDataFrame(
        geometry=[box(x0 + 10, y0 + 10, x0 + 20, y0 + 20), box(x0 + 500, y0, x0 + 510, y0 + 10)],
        crs="EPSG:32633",
    )
    rd = gpd.GeoDataFrame(
        geometry=[LineString([(x0 - 50, y0 + 50), (x0 + 150, y0 + 50)])], crs="EPSG:32633"
    )
    files = {
        "buildings": ("b.geojson", bld.to_crs(4326).to_json(), "application/geo+json"),
        "roads": ("r.geojson", rd.to_crs(4326).to_json(), "application/geo+json"),
    }
    r = client.post(f"{API}/analyses/{job['id']}/exposure", headers=h, files=files)
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["buildings"]["total"] == 2 and rep["buildings"]["potentially_exposed"] == 1
    assert rep["roads"]["flooded_km"] == pytest.approx(0.1, rel=0.01)
    assert "not damage" in rep["label"]
    got = client.get(f"{API}/analyses/{job['id']}", headers=h).json()["files"]
    assert {"exposure.json", "exposed_buildings.geojson", "exposed_roads.geojson"} <= set(got)
    bad = {"buildings": ("b.geojson", rd.to_crs(4326).to_json(), "application/geo+json")}
    assert (
        client.post(f"{API}/analyses/{job['id']}/exposure", headers=h, files=bad).status_code == 422
    )
    assert client.post(f"{API}/analyses/{job['id']}/exposure", headers=h).status_code == 422


def test_previews_and_reports(env):
    client, sm, tmp = env
    h = token(client)
    register_tiny_model(sm, tmp)
    img = upload(client, h, s1_bytes(tmp)).json()
    assert len(img["bbox"]) == 4 and img["bbox"][0] < img["bbox"][2]
    p = client.get(f"{API}/images/{img['id']}/preview.png", headers=h)
    assert p.status_code == 200 and p.content[:8] == b"\x89PNG\r\n\x1a\n"
    job = client.post(f"{API}/analyses", headers=h, json={"image_id": img["id"]}).json()
    assert job["image_filename"] == "scene.tif" and job["model_name"] == "tiny:1"
    m = client.get(f"{API}/analyses/{job['id']}/preview.png", headers=h)
    assert m.status_code == 200 and m.content[:4] == b"\x89PNG"
    csv_r = client.get(f"{API}/analyses/{job['id']}/report.csv", headers=h)
    assert csv_r.status_code == 200 and "flooded_area_km2" in csv_r.text
    pdf = client.get(f"{API}/analyses/{job['id']}/report.pdf", headers=h)
    assert pdf.status_code == 200 and pdf.content[:5] == b"%PDF-"
    other = token(client, "bo@example.com")
    assert client.get(f"{API}/analyses/{job['id']}/report.pdf", headers=other).status_code == 404
    assert client.get(f"{API}/images/{img['id']}/preview.png", headers=other).status_code == 404


def test_experiments_endpoint(env, monkeypatch, tmp_path):
    client, _, _ = env
    h = token(client)
    from backend.api.routes import experiments

    monkeypatch.setattr(experiments, "SUMMARY", tmp_path / "missing.json")
    assert client.get(f"{API}/experiments", headers=h).status_code == 404
    f = tmp_path / "e.json"
    f.write_text('{"sen1floods11": []}')
    monkeypatch.setattr(experiments, "SUMMARY", f)
    assert client.get(f"{API}/experiments", headers=h).json() == {"sen1floods11": []}
    assert client.get(f"{API}/experiments").status_code == 401
