from __future__ import annotations

import uuid
from pathlib import PurePath

from fastapi import APIRouter, HTTPException, Response, UploadFile, status
from shapely.geometry import shape
from sqlalchemy import select

from backend.api.deps import CurrentUser, DbSession
from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.models.registry import SatelliteImage
from backend.schemas.analysis import ImageRead
from backend.services.storage import StorageError, get_storage
from geospatial.raster import footprint
from geospatial.sentinel1 import InputValidationError, validate_input

router = APIRouter(prefix="/images", tags=["images"])
log = get_logger(__name__)
ALLOWED_SUFFIXES = {".tif", ".tiff"}


@router.post("", response_model=ImageRead, status_code=status.HTTP_201_CREATED)
def upload_image(file: UploadFile, db: DbSession, user: CurrentUser) -> SatelliteImage:
    """Upload a Sentinel-1 GeoTIFF (2 bands: VV, VH in dB). Validated before it is accepted."""
    settings = get_settings()
    name = PurePath(file.filename or "").name  # strip any client-supplied directories
    if PurePath(name).suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only GeoTIFF (.tif) accepted")
    image_id = uuid.uuid4()
    key = f"images/{image_id}/scene.tif"  # never use the client filename on disk
    storage = get_storage()
    try:
        uri, sha, size = storage.save_stream(key, file.file, settings.max_upload_mb * 1024**2)
    except StorageError as exc:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(exc)) from None
    path = storage.local_path(uri)
    try:
        info = validate_input(path, max_pixels=settings.max_image_pixels)
        fp = shape(footprint(path))
    except InputValidationError as exc:
        storage.delete_prefix(f"images/{image_id}")
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from None
    img = SatelliteImage(
        id=image_id,
        owner_id=user.id,
        filename=name[:255],
        storage_uri=uri,
        sha256=sha,
        size_bytes=size,
        width=info["width"],
        height=info["height"],
        crs=info["crs"],
        footprint=fp,
    )
    db.add(img)
    db.commit()
    db.refresh(img)
    log.info("image_uploaded", image_id=str(image_id), size=size, crs=info["crs"])
    return img


@router.get("", response_model=list[ImageRead])
def list_images(db: DbSession, user: CurrentUser) -> list[SatelliteImage]:
    q = select(SatelliteImage).where(SatelliteImage.owner_id == user.id)
    return list(db.scalars(q.order_by(SatelliteImage.created_at.desc())))


@router.get("/{image_id}", response_model=ImageRead)
def get_image(image_id: uuid.UUID, db: DbSession, user: CurrentUser) -> SatelliteImage:
    img = db.get(SatelliteImage, image_id)
    if img is None or img.owner_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")
    return img


@router.get("/{image_id}/preview.png")
def image_preview(image_id: uuid.UUID, db: DbSession, user: CurrentUser) -> Response:
    """False-colour SAR preview warped to WGS84; place it on a map with ``ImageRead.bbox``."""
    from geospatial.preview import sar_preview

    img = get_image(image_id, db, user)
    png, _ = sar_preview(get_storage().local_path(img.storage_uri))
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})
