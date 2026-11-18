"""Model registry, uploaded imagery, and analysis jobs."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base, TimestampMixin
from backend.db.types import Geometry


class ModelVersion(TimestampMixin, Base):
    """A trained model artifact. Weights live in object storage; the DB holds metadata only."""

    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    task: Mapped[str] = mapped_column(String(50), nullable=False, default="flood_segmentation")
    arch: Mapped[str] = mapped_column(String(50), nullable=False)
    artifact_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    training: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    is_default: Mapped[bool] = mapped_column(default=False, nullable=False)


class SatelliteImage(TimestampMixin, Base):
    __tablename__ = "satellite_images"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sensor: Mapped[str] = mapped_column(String(50), nullable=False, default="sentinel-1")
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    crs: Mapped[str] = mapped_column(String(100), nullable=False)
    footprint = mapped_column(Geometry("POLYGON", 4326), nullable=False)
    acquired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class JobStatus(enum.StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class AnalysisJob(TimestampMixin, Base):
    __tablename__ = "analysis_jobs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    image_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("satellite_images.id"), nullable=False)
    # Optional pre-event scene for change detection (warped onto the post-event grid).
    pre_image_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("satellite_images.id"), nullable=True
    )
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("model_versions.id"), nullable=False
    )
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status", native_enum=False, length=20),
        default=JobStatus.queued,
        nullable=False,
        index=True,
    )
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    output_prefix: Mapped[str | None] = mapped_column(String(500))
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    flood_extent = mapped_column(Geometry("MULTIPOLYGON", 4326), nullable=True)

    image: Mapped[SatelliteImage] = relationship(lazy="joined", foreign_keys=[image_id])
    pre_image: Mapped[SatelliteImage | None] = relationship(
        lazy="joined", foreign_keys=[pre_image_id]
    )
    model_version: Mapped[ModelVersion] = relationship(lazy="joined")
