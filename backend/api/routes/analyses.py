from __future__ import annotations

import io
import json
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select

from backend.api.deps import CurrentUser, DbSession
from backend.core.logging import get_logger
from backend.models.registry import AnalysisJob, JobStatus, ModelVersion, SatelliteImage
from backend.schemas.analysis import AnalysisCreate, AnalysisRead
from backend.services.storage import get_storage

router = APIRouter(prefix="/analyses", tags=["analyses"])
log = get_logger(__name__)
OUTPUT_FILES = {
    "flood_mask.tif": "image/tiff",
    "probability.tif": "image/tiff",
    "flood_polygons.geojson": "application/geo+json",
    "summary.json": "application/json",
    "exposure.json": "application/json",
    "exposed_buildings.geojson": "application/geo+json",
    "exposed_roads.geojson": "application/geo+json",
    "new_water.tif": "image/tiff",
}


def _files(job: AnalysisJob) -> list[str]:
    if not job.output_prefix:
        return []
    names = set(get_storage().list_names(job.output_prefix))
    return [f for f in OUTPUT_FILES if f in names]


def _read(job: AnalysisJob) -> dict[str, Any]:
    out = AnalysisRead.model_validate(job).model_dump()
    out["files"] = _files(job)
    out["image_filename"] = job.image.filename if job.image else None
    mv = job.model_version
    out["model_name"] = f"{mv.name}:{mv.version}" if mv else None
    return out


def _owned(db: DbSession, job_id: uuid.UUID, user: CurrentUser) -> AnalysisJob:
    job = db.get(AnalysisJob, job_id)
    if job is None or job.owner_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Analysis not found")
    return job


@router.post("", response_model=AnalysisRead, status_code=status.HTTP_202_ACCEPTED)
def create_analysis(payload: AnalysisCreate, db: DbSession, user: CurrentUser) -> dict[str, Any]:
    img = db.get(SatelliteImage, payload.image_id)
    if img is None or img.owner_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")
    if payload.model_version_id:
        model = db.get(ModelVersion, payload.model_version_id)
    else:
        model = db.scalar(select(ModelVersion).where(ModelVersion.is_default.is_(True)))
    if model is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found (no default registered)")
    pre_id = None
    if payload.pre_image_id:
        pre = db.get(SatelliteImage, payload.pre_image_id)
        if pre is None or pre.owner_id != user.id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Pre-event image not found")
        if pre.id == img.id:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "Pre-event image must differ from the post-event image",
            )
        if not pre.footprint.intersects(img.footprint):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, "Pre- and post-event images do not overlap"
            )
        pre_id = pre.id
    job = AnalysisJob(
        owner_id=user.id, image_id=img.id, model_version_id=model.id, pre_image_id=pre_id
    )
    db.add(job)
    db.commit()
    from backend.workers.tasks import run_flood_analysis

    run_flood_analysis.delay(str(job.id))
    db.refresh(job)
    log.info("analysis_queued", job_id=str(job.id), model=f"{model.name}:{model.version}")
    return _read(job)


@router.get("", response_model=list[AnalysisRead])
def list_analyses(db: DbSession, user: CurrentUser) -> list[dict[str, Any]]:
    q = select(AnalysisJob).where(AnalysisJob.owner_id == user.id)
    return [_read(j) for j in db.scalars(q.order_by(AnalysisJob.created_at.desc()))]


@router.get("/{job_id}", response_model=AnalysisRead)
def get_analysis(job_id: uuid.UUID, db: DbSession, user: CurrentUser) -> dict[str, Any]:
    return _read(_owned(db, job_id, user))


@router.get("/{job_id}/files/{name}")
def download_output(job_id: uuid.UUID, name: str, db: DbSession, user: CurrentUser):
    job = _owned(db, job_id, user)
    if name not in OUTPUT_FILES:  # allow-list: no path traversal possible
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown output")
    if name not in _files(job):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Output not available")
    path = get_storage().local_path(f"{job.output_prefix}/{name}")
    return FileResponse(path, media_type=OUTPUT_FILES[name], filename=f"{job_id}_{name}")


EXPOSURE_MAX_BYTES = 50 * 1024**2
EXPOSURE_MAX_FEATURES = 500_000


def _read_layer(upload, kind: str):
    import geopandas as gpd

    if upload is None:
        return None
    data = upload.file.read(EXPOSURE_MAX_BYTES + 1)
    if len(data) > EXPOSURE_MAX_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"{kind} layer too large")
    try:
        gdf = gpd.read_file(io.BytesIO(data))
    except Exception:  # noqa: BLE001 - any parse failure is a client error
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"{kind}: not a readable GeoJSON"
        ) from None
    if len(gdf) > EXPOSURE_MAX_FEATURES:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"{kind}: too many features")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")  # RFC 7946 GeoJSON is WGS84
    want = {"buildings": {"Polygon", "MultiPolygon"}, "roads": {"LineString", "MultiLineString"}}
    bad = set(gdf.geom_type.dropna()) - want[kind]
    if bad:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"{kind}: unexpected geometry types {sorted(bad)}",
        )
    return gdf


@router.post("/{job_id}/exposure")
def compute_exposure(
    job_id: uuid.UUID,
    db: DbSession,
    user: CurrentUser,
    buildings: UploadFile | None = None,
    roads: UploadFile | None = None,
    min_building_overlap: float = 0.25,
) -> dict[str, Any]:
    """Overlay building/road layers (GeoJSON) on this analysis's predicted flood extent.

    Result is *potential exposure* (geometric overlap with predicted water), not damage."""
    import geopandas as gpd

    from geospatial.exposure import (
        ExposureParams,
        exposure_report,
        features_to_geojson,
        summary_only,
    )

    job = _owned(db, job_id, user)
    if job.status != JobStatus.succeeded:
        raise HTTPException(status.HTTP_409_CONFLICT, "Analysis has not succeeded")
    if buildings is None and roads is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Upload buildings and/or roads")
    if not 0 < min_building_overlap <= 1:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "min_building_overlap must be in (0, 1]"
        )
    b, r = _read_layer(buildings, "buildings"), _read_layer(roads, "roads")
    flood = gpd.GeoDataFrame(
        geometry=[job.flood_extent] if job.flood_extent is not None else [], crs="EPSG:4326"
    )
    if not len(flood):  # no water predicted: exposure is zero but still well-defined
        ref = next(g for g in (b, r) if g is not None)
        flood = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        from geospatial.exposure import utm_crs_for

        crs = utm_crs_for(ref.to_crs("EPSG:4326").geometry.union_all())
    else:
        crs = None
    rep = exposure_report(flood, b, r, ExposureParams(min_building_overlap), crs=crs)
    out = summary_only(rep)
    storage = get_storage()
    prefix = f"outputs/{job.id}"
    base = storage.workdir(prefix)
    if "building_features" in rep:
        gj = features_to_geojson(
            rep["building_features"], ["flooded_fraction", "potentially_exposed"]
        )
        (base / "exposed_buildings.geojson").write_text(json.dumps(gj))
    if "road_features" in rep:
        gj = features_to_geojson(rep["road_features"], ["flooded_length_m", "potentially_exposed"])
        (base / "exposed_roads.geojson").write_text(json.dumps(gj))
    (base / "exposure.json").write_text(json.dumps(out, default=float, indent=1))
    storage.commit_dir(prefix)
    job.summary = {**(job.summary or {}), "exposure": out}
    db.commit()
    return out


def _succeeded(db: DbSession, job_id: uuid.UUID, user: CurrentUser) -> AnalysisJob:
    job = _owned(db, job_id, user)
    if job.status != JobStatus.succeeded:
        raise HTTPException(status.HTTP_409_CONFLICT, "Analysis has not succeeded")
    return job


@router.get("/{job_id}/preview.png")
def mask_preview_png(job_id: uuid.UUID, db: DbSession, user: CurrentUser) -> Response:
    """Predicted flood water as a transparent PNG warped to WGS84 (same bbox as the image)."""
    from geospatial.preview import mask_preview

    job = _succeeded(db, job_id, user)
    png, _ = mask_preview(get_storage().local_path(f"{job.output_prefix}/flood_mask.tif"))
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


@router.get("/{job_id}/report.csv")
def report_csv(job_id: uuid.UUID, db: DbSession, user: CurrentUser) -> Response:
    from backend.services.reports import summary_csv

    job = _succeeded(db, job_id, user)
    return Response(
        summary_csv(job),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{job_id}_summary.csv"'},
    )


@router.get("/{job_id}/report.pdf")
def report_pdf(job_id: uuid.UUID, db: DbSession, user: CurrentUser) -> Response:
    from backend.services.reports import pdf_report
    from geospatial.preview import mask_preview, sar_preview

    job = _succeeded(db, job_id, user)
    storage = get_storage()
    sar, _ = sar_preview(storage.local_path(job.image.storage_uri), max_size=900)
    mask, _ = mask_preview(storage.local_path(f"{job.output_prefix}/flood_mask.tif"), max_size=900)
    return Response(
        pdf_report(job, sar, mask),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{job_id}_report.pdf"'},
    )
