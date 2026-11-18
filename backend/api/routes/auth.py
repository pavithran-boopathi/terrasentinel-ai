from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func, select

from backend.api.deps import CurrentUser, DbSession
from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.core.security import create_access_token, hash_password, verify_password
from backend.models.user import User
from backend.schemas.auth import Token, UserCreate, UserRead

router = APIRouter(prefix="/auth", tags=["auth"])
log = get_logger(__name__)

# Pre-computed hash used to equalise timing when the email does not exist.
_DUMMY_HASH = hash_password("timing-equaliser-not-a-real-password")


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, db: DbSession) -> User:
    if not get_settings().allow_registration:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Registration is disabled")
    email = payload.email.lower()
    if db.scalar(select(User).where(func.lower(User.email) == email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")
    # First registered account becomes admin so a fresh install can be administered.
    is_first = db.scalar(select(func.count()).select_from(User)) == 0
    user = User(
        email=email,
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
        is_admin=is_first,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    log.info("user_registered", user_id=str(user.id), is_admin=is_first)
    return user


@router.post("/login", response_model=Token)
def login(form: Annotated[OAuth2PasswordRequestForm, Depends()], db: DbSession) -> Token:
    user = db.scalar(select(User).where(func.lower(User.email) == form.username.lower()))
    valid = verify_password(form.password, user.hashed_password if user else _DUMMY_HASH)
    if not user or not valid or not user.is_active:
        log.info("login_failed")
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    settings = get_settings()
    token = create_access_token(str(user.id))
    return Token(access_token=token, expires_in=settings.access_token_expire_minutes * 60)


@router.get("/me", response_model=UserRead)
def me(user: CurrentUser) -> User:
    return user
