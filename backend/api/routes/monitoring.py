from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
from fastapi import APIRouter
from sqlalchemy import func, select

from backend.api.deps import AdminUser, CurrentUser, DbSession
from backend.models.registry import AnalysisJob, JobStatus, ModelVersion
from backend.schemas.analysis import MonitoringSummary

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/summary", response_model=MonitoringSummary)
def summary(db: DbSession, _: CurrentUser) -> MonitoringSummary:
    by_status = dict(
        db.execute(select(AnalysisJob.status, func.count()).group_by(AnalysisJob.status)).all()
    )
    since = datetime.now(UTC) - timedelta(hours=24)
    recent = list(db.scalars(select(AnalysisJob).where(AnalysisJob.finished_at >= since)))
    done = [
        j
        for j in db.scalars(
            select(AnalysisJob).where(
                AnalysisJob.status == JobStatus.succeeded, AnalysisJob.started_at.is_not(None)
            )
        )
    ]
    durs = np.array([(j.finished_at - j.started_at).total_seconds() for j in done])
    per_model = []
    for m in db.scalars(select(ModelVersion)):
        js = [j for j in done if j.model_version_id == m.id]
        secs = [j.summary.get("seconds") for j in js if j.summary]
        per_model.append(
            {
                "model": f"{m.name}:{m.version}",
                "succeeded_jobs": len(js),
                "mean_inference_seconds": float(np.mean(secs)) if secs else None,
            }
        )
    return MonitoringSummary(
        jobs_by_status={
            str(k.value if hasattr(k, "value") else k): v for k, v in by_status.items()
        },
        succeeded_last_24h=sum(j.status == JobStatus.succeeded for j in recent),
        failed_last_24h=sum(j.status == JobStatus.failed for j in recent),
        duration_seconds={
            "mean": float(durs.mean()) if durs.size else None,
            "p95": float(np.percentile(durs, 95)) if durs.size else None,
        },
        per_model=per_model,
    )


@router.post("/recover-stale-jobs")
def recover_stale(db: DbSession, _: AdminUser, older_than_minutes: int = 60) -> dict[str, int]:
    """Requeue jobs stuck in ``running`` (e.g. the worker was killed mid-job)."""
    from backend.workers.tasks import run_flood_analysis

    cutoff = datetime.now(UTC) - timedelta(minutes=older_than_minutes)
    stale = list(
        db.scalars(
            select(AnalysisJob).where(
                AnalysisJob.status == JobStatus.running, AnalysisJob.started_at < cutoff
            )
        )
    )
    for j in stale:
        j.status = JobStatus.queued
        j.progress = 0.0
    db.commit()
    for j in stale:
        run_flood_analysis.delay(str(j.id))
    return {"requeued": len(stale)}
