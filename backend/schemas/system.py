from __future__ import annotations

from pydantic import BaseModel


class HealthStatus(BaseModel):
    status: str
    version: str


class ReadinessStatus(BaseModel):
    status: str
    checks: dict[str, str]


class ComputeInfo(BaseModel):
    platform: str
    machine: str
    python_version: str
    torch_version: str | None
    mps_available: bool | None
    cuda_available: bool | None
    recommended_device: str
