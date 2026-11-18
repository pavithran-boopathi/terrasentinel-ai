"""Import all models here so Alembic autogenerate and ``create_all`` can see them."""

from backend.models.registry import AnalysisJob, JobStatus, ModelVersion, SatelliteImage
from backend.models.user import User

__all__ = ["AnalysisJob", "JobStatus", "ModelVersion", "SatelliteImage", "User"]
