from __future__ import annotations

from backend.services.compute import compute_info
from backend.workers.celery_app import celery_app


@celery_app.task(name="system.compute_info")
def compute_info_task() -> dict:
    """Report the worker's compute device. Used to verify the worker pipeline end to end."""
    return compute_info()
