import jwt
import pytest

from backend.core.security import create_access_token
from tests.backend.conftest import API

PASSWORD = "correct-horse-battery"


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def register(client, email="ana@example.com", password=PASSWORD):
    return client.post(f"{API}/auth/register", json={"email": email, "password": password})


def login(client, email="ana@example.com", password=PASSWORD):
    return client.post(f"{API}/auth/login", data={"username": email, "password": password})


def test_register_login_me_flow(client):
    r = register(client)
    assert r.status_code == 201
    user = r.json()
    assert user["email"] == "ana@example.com"
    assert "hashed_password" not in user
    assert user["is_admin"] is True  # first account

    token = login(client).json()["access_token"]
    me = client.get(f"{API}/auth/me", headers=auth(token))
    assert me.status_code == 200
    assert me.json()["id"] == user["id"]


def test_second_user_is_not_admin(client):
    register(client)
    r = register(client, email="bo@example.com")
    assert r.json()["is_admin"] is False


def test_duplicate_email_case_insensitive(client):
    register(client)
    assert register(client, email="ANA@example.com").status_code == 409


@pytest.mark.parametrize("password", ["short", "x" * 129])
def test_password_length_validation(client, password):
    assert register(client, password=password).status_code == 422


def test_wrong_password_rejected(client):
    register(client)
    assert login(client, password="wrong-password-123").status_code == 401


def test_unknown_email_rejected(client):
    assert login(client, email="nobody@example.com").status_code == 401


def test_me_requires_token(client):
    assert client.get(f"{API}/auth/me").status_code == 401


def test_me_rejects_tampered_token(client):
    register(client)
    token = login(client).json()["access_token"]
    bad = token[:-2] + ("AA" if token[-2:] != "AA" else "BB")
    assert client.get(f"{API}/auth/me", headers=auth(bad)).status_code == 401


def test_me_rejects_token_for_unknown_user(client):
    token = create_access_token("00000000-0000-0000-0000-000000000000")
    assert client.get(f"{API}/auth/me", headers=auth(token)).status_code == 401


def test_me_rejects_wrong_signing_key(client):
    reg = register(client).json()
    forged = jwt.encode(
        {"sub": reg["id"], "iat": 0, "exp": 9999999999, "type": "access"},
        "attacker-key-attacker-key-attacker-key",
        algorithm="HS256",
    )
    r = client.get(f"{API}/auth/me", headers={"Authorization": f"Bearer {forged}"})
    assert r.status_code == 401
