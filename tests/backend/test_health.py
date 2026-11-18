from tests.backend.conftest import API


def test_health(client):
    r = client.get(f"{API}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "x-request-id" in r.headers


def test_request_id_is_propagated(client):
    r = client.get(f"{API}/health", headers={"x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123"


def test_ready_reports_database(client):
    r = client.get(f"{API}/health/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "checks": {"database": "ok"}}


def test_compute_info_shape(client):
    r = client.get(f"{API}/system/compute")
    assert r.status_code == 200
    body = r.json()
    assert body["recommended_device"] in {"cpu", "mps", "cuda"}
    assert body["machine"]
