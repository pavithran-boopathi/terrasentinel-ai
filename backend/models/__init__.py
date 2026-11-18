"""Import all models here so Alembic autogenerate and ``create_all`` can see them."""

from backend.models.user import User

__all__ = ["User"]
