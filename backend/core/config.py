"""Application settings loaded from environment variables (prefix ``TS_``)."""

from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TS_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    environment: Literal["development", "test", "production"] = "development"
    app_name: str = "TerraSentinel AI"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    log_json: bool = False

    # Database. Docker Compose sets a PostgreSQL+PostGIS URL; without it we fall back to a
    # local SQLite file so the API can run on a laptop with no services installed.
    database_url: str = f"sqlite:///{REPO_ROOT / 'data' / 'terrasentinel-dev.db'}"

    # Celery / Redis.
    redis_url: str = "redis://localhost:6379/0"
    celery_task_always_eager: bool = False

    # Object storage for imagery, masks and model weights (never stored in DB rows).
    storage_backend: Literal["local", "s3"] = "local"
    storage_root: Path = REPO_ROOT / "data" / "storage"  # local store, or S3 read/write cache
    s3_bucket: str = ""

    # Uploads & inference.
    max_upload_mb: int = 1024
    max_image_pixels: int = 400_000_000
    inference_device: str = "auto"  # auto | cpu | mps | cuda
    inference_tile: int = 512
    inference_overlap: int = 64
    inference_batch: int = 4
    job_max_retries: int = 2

    # Auth.
    secret_key: str = Field(default="", repr=False)
    access_token_expire_minutes: int = 60
    jwt_algorithm: str = "HS256"
    allow_registration: bool = True

    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    @model_validator(mode="after")
    def _check_secret(self) -> Settings:
        if not self.secret_key:
            if self.environment == "production":
                raise ValueError("TS_SECRET_KEY must be set in production")
            # Ephemeral key: tokens are invalidated on restart. Fine for local development only.
            self.secret_key = secrets.token_urlsafe(48)
        elif len(self.secret_key) < 32:
            raise ValueError("TS_SECRET_KEY must be at least 32 characters")
        return self

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    return Settings()
