import pytest
from pydantic import ValidationError

from backend.core.config import Settings


def test_production_requires_secret(monkeypatch):
    monkeypatch.delenv("TS_SECRET_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(environment="production", secret_key="", _env_file=None)


def test_short_secret_rejected():
    with pytest.raises(ValidationError):
        Settings(secret_key="too-short", _env_file=None)


def test_dev_generates_ephemeral_secret(monkeypatch):
    monkeypatch.delenv("TS_SECRET_KEY", raising=False)
    s = Settings(environment="development", secret_key="", _env_file=None)
    assert len(s.secret_key) >= 32
