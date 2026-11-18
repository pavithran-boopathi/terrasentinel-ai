from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, computed_field, field_serializer
from shapely.geometry import mapping

from backend.models.registry import JobStatus


class ModelVersionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: uuid.UUID
    name: str
    version: str
    task: str
    arch: str
    sha256: str
    size_bytes: int
    threshold: float
    metrics: dict[str, Any]
    training: dict[str, Any]
    is_default: bool
    created_at: datetime


class ImageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    sha256: str
    size_bytes: int
    sensor: str
    width: int
    height: int
    crs: str
    footprint: Any
    created_at: datetime

    @computed_field  # [west, south, east, north] in WGS84, for map image sources
    @property
    def bbox(self) -> list[float]:
        return list(self.footprint.bounds)

    @field_serializer("footprint")
    def _geojson(self, g):  # shapely → GeoJSON dict
        return mapping(g) if g is not None else None


class AnalysisCreate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    image_id: uuid.UUID
    model_version_id: uuid.UUID | None = None  # default model when omitted


class AnalysisRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: uuid.UUID
    image_id: uuid.UUID
    image_filename: str | None = None
    model_name: str | None = None
    model_version_id: uuid.UUID
    status: JobStatus
    progress: float
    attempts: int
    error: str | None
    summary: dict[str, Any] | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    files: list[str] = []


class MonitoringSummary(BaseModel):
    jobs_by_status: dict[str, int]
    succeeded_last_24h: int
    failed_last_24h: int
    duration_seconds: dict[str, float | None]
    per_model: list[dict[str, Any]]
