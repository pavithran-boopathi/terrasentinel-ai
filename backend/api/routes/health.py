from __future__ import annotations

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from backend import __version__
from backend.api.deps import DbSession
from backend.schemas.system import ComputeInfo, HealthStatus, ReadinessStatus
from backend.services.compute import compute_info

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthStatus)
def health() -> HealthStatus:
    """Liveness probe: the process is up."""
    return HealthStatus(status="ok", version=__version__)


@router.get("/health/ready", response_model=ReadinessStatus)
def ready(db: DbSession, response: Response) -> ReadinessStatus:
    """Readiness probe: dependencies the API needs are reachable."""
    checks: dict[str, str] = {}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001 - report any DB failure
        checks["database"] = f"error: {type(exc).__name__}"
    ok = all(v == "ok" for v in checks.values())
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessStatus(status="ok" if ok else "degraded", checks=checks)


@router.get("/system/compute", response_model=ComputeInfo)
def compute() -> ComputeInfo:
    return ComputeInfo(**compute_info())
