from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from backend.api.deps import CurrentUser, DbSession
from backend.models.registry import ModelVersion
from backend.schemas.analysis import ModelVersionRead

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelVersionRead])
def list_models(db: DbSession, _: CurrentUser) -> list[ModelVersion]:
    q = select(ModelVersion).order_by(ModelVersion.is_default.desc(), ModelVersion.created_at)
    return list(db.scalars(q))


@router.get("/{model_id}", response_model=ModelVersionRead)
def get_model(model_id: uuid.UUID, db: DbSession, _: CurrentUser) -> ModelVersion:
    m = db.get(ModelVersion, model_id)
    if m is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")
    return m
