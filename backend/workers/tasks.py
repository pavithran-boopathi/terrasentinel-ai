from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from shapely.geometry import MultiPolygon, shape
from shapely.ops import unary_union
from sqlalchemy.orm import Session

from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.db.session import get_sessionmaker
from backend.models.registry import AnalysisJob, JobStatus
from backend.services.compute import compute_info
from backend.services.storage import get_storage
from backend.workers.celery_app import celery_app
from geospatial.sentinel1 import InputValidationError

log = get_logger(__name__)


def session_factory() -> Session:
    """Indirection so tests can point the worker at their database."""
    return get_sessionmaker()()


@celery_app.task(name="system.compute_info")
def compute_info_task() -> dict:
    """Report the worker's compute device. Used to verify the worker pipeline end to end."""
    return compute_info()


@lru_cache(maxsize=2)
def _predictor(model_path: str, threshold: float, model_id: str, device: str):
    # Imported lazily: torch is only installed in the worker image.
    from ml.inference.predictor import FloodPredictor

    return FloodPredictor.from_checkpoint(
        model_path, device=device, threshold=threshold, model_id=model_id
    )


class PermanentJobError(Exception):
    """Retrying cannot help (bad input, missing artifact)."""


def execute_analysis(job_id: str) -> dict[str, Any]:
    """Run one analysis job to completion. Raises on failure; status bookkeeping is done here."""
    settings = get_settings()
    storage = get_storage()
    with session_factory() as db:
        job = db.get(AnalysisJob, uuid.UUID(job_id))
        if job is None:
            raise PermanentJobError(f"job {job_id} not found")
        if job.status == JobStatus.succeeded:
            return job.summary or {}
        job.status = JobStatus.running
        job.attempts += 1
        job.started_at = datetime.now(UTC)
        job.error = None
        db.commit()
        model = job.model_version
        image_path = storage.local_path(job.image.storage_uri)
        model_path = storage.local_path(model.artifact_uri)
        if not model_path.exists():
            raise PermanentJobError(f"model artifact missing: {model.artifact_uri}")
        prefix = f"outputs/{job.id}"
        out_dir = storage.workdir(prefix)
        last = {"p": 0.0}

        def progress(p: float) -> None:
            if p - last["p"] >= 0.1 or p >= 1.0:
                last["p"] = p
                job.progress = round(min(p, 1.0) * 0.95, 3)  # final 5%: vectorising + saving
                db.commit()

        predictor = _predictor(
            str(model_path),
            model.threshold,
            f"{model.name}:{model.version}",
            settings.inference_device,
        )
        summary = predictor.run(
            image_path,
            out_dir,
            tile=settings.inference_tile,
            overlap=settings.inference_overlap,
            batch=settings.inference_batch,
            progress=progress,
        )
        if job.pre_image is not None:
            summary["change"] = _change_detection(
                predictor,
                storage.local_path(job.pre_image.storage_uri),
                image_path,
                out_dir,
                settings,
            )
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        gj = json.loads((out_dir / "flood_polygons.geojson").read_text())
        polys = [shape(f["geometry"]) for f in gj["features"]]
        if polys:
            merged = unary_union(polys)
            job.flood_extent = (
                merged
                if isinstance(merged, MultiPolygon)
                else MultiPolygon(
                    [g for g in getattr(merged, "geoms", [merged]) if g.geom_type == "Polygon"]
                )
            )
        storage.commit_dir(prefix)
        job.summary = summary
        job.output_prefix = storage.uri(prefix)
        job.status = JobStatus.succeeded
        job.progress = 1.0
        job.finished_at = datetime.now(UTC)
        db.commit()
        log.info(
            "analysis_succeeded",
            job_id=job_id,
            seconds=summary["seconds"],
            flooded_km2=round(summary["flooded_area_km2"], 3),
        )
        return summary


def _change_detection(predictor, pre_path, post_path, out_dir, settings) -> dict[str, Any]:
    """Warp the pre-event scene onto the post grid, segment it with the same model, and compare."""
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling

    from geospatial.change import NODATA, darkening_mask, new_water, receded_water
    from geospatial.raster import reproject_to_match, write_cog
    from ml.inference.predictor import row_pixel_areas_m2

    pre_dir = out_dir / "pre_event"
    pre_dir.mkdir(exist_ok=True)
    aligned = reproject_to_match(
        pre_path, post_path, pre_dir / "pre_aligned.tif", resampling=Resampling.bilinear
    )
    predictor.run(
        aligned,
        pre_dir,
        tile=settings.inference_tile,
        overlap=settings.inference_overlap,
        batch=settings.inference_batch,
    )
    with (
        rasterio.open(out_dir / "flood_mask.tif") as post_m,
        rasterio.open(pre_dir / "flood_mask.tif") as pre_m,
        rasterio.open(aligned) as pre_img,
        rasterio.open(post_path) as post_img,
    ):
        post_mask, pre_mask = post_m.read(1), pre_m.read(1)
        nw, rc = new_water(pre_mask, post_mask), receded_water(pre_mask, post_mask)
        dark = darkening_mask(pre_img.read().astype("float32"), post_img.read().astype("float32"))
        write_cog(
            nw,
            out_dir / "new_water.tif",
            crs=post_m.crs,
            transform=post_m.transform,
            nodata=NODATA,
            descriptions=["new_water"],
        )
        area = row_pixel_areas_m2(post_m)[:, None]

    def km2(mask: np.ndarray) -> float:
        return float(((mask == 1) * area).sum() / 1e6)

    for f in pre_dir.glob("*.tif"):
        if f.name != "flood_mask.tif":
            f.unlink()
    return {
        "pre_water_km2": km2(pre_mask),
        "post_water_km2": km2(post_mask),
        "new_water_km2": km2(nw),
        "receded_water_km2": km2(rc),
        "darkening_rule_km2": km2(dark),
        "agreement_new_water_vs_darkening": float(
            ((nw == 1) & (dark == 1)).sum() / max(1, ((nw == 1) | (dark == 1)).sum())
        ),
        "note": "new_water = model water after AND not before; darkening = VH drop ≥ 3 dB to "
        "≤ −18 dB (unsupervised log-ratio rule)",
    }


def mark_failed(job_id: str, error: str) -> None:
    with session_factory() as db:
        job = db.get(AnalysisJob, uuid.UUID(job_id))
        if job is not None:
            job.status = JobStatus.failed
            job.error = error[:2000]
            job.finished_at = datetime.now(UTC)
            db.commit()


def is_permanent(exc: BaseException) -> bool:
    return isinstance(exc, PermanentJobError | InputValidationError | FileNotFoundError)


@celery_app.task(bind=True, name="analysis.flood_segmentation")
def run_flood_analysis(self, job_id: str) -> dict[str, Any]:
    max_retries = get_settings().job_max_retries
    try:
        return execute_analysis(job_id)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        if not is_permanent(exc) and self.request.retries < max_retries:
            log.warning("analysis_retry", job_id=job_id, error=msg, retry=self.request.retries + 1)
            with session_factory() as db:
                job = db.get(AnalysisJob, uuid.UUID(job_id))
                if job is not None:
                    job.status = JobStatus.queued
                    job.error = f"retrying after: {msg}"[:2000]
                    db.commit()
            raise self.retry(exc=exc, countdown=15 * (self.request.retries + 1)) from exc
        log.error("analysis_failed", job_id=job_id, error=msg)
        mark_failed(job_id, msg)
        return {"error": msg}
