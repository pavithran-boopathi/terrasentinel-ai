"""model registry, satellite images, analysis jobs (PostGIS geometry on PostgreSQL)

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from backend.db.types import Geometry

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _ts() -> list[sa.Column]:
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("task", sa.String(50), nullable=False),
        sa.Column("arch", sa.String(50), nullable=False),
        sa.Column("artifact_uri", sa.String(500), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("training", sa.JSON(), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False),
        *_ts(),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_versions")),
    )
    op.create_table(
        "satellite_images",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("storage_uri", sa.String(500), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sensor", sa.String(50), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("crs", sa.String(100), nullable=False),
        sa.Column("footprint", Geometry("POLYGON", 4326), nullable=False),
        sa.Column("acquired_at", sa.DateTime(timezone=True), nullable=True),
        *_ts(),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.id"], name=op.f("fk_satellite_images_owner_id_users")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_satellite_images")),
    )
    op.create_index(op.f("ix_satellite_images_owner_id"), "satellite_images", ["owner_id"])
    op.create_table(
        "analysis_jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("image_id", sa.Uuid(), nullable=False),
        sa.Column("model_version_id", sa.Uuid(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "queued",
                "running",
                "succeeded",
                "failed",
                name="job_status",
                native_enum=False,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("output_prefix", sa.String(500), nullable=True),
        sa.Column("summary", sa.JSON(), nullable=True),
        sa.Column("flood_extent", Geometry("MULTIPOLYGON", 4326), nullable=True),
        *_ts(),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.id"], name=op.f("fk_analysis_jobs_owner_id_users")
        ),
        sa.ForeignKeyConstraint(
            ["image_id"],
            ["satellite_images.id"],
            name=op.f("fk_analysis_jobs_image_id_satellite_images"),
        ),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name=op.f("fk_analysis_jobs_model_version_id_model_versions"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_analysis_jobs")),
    )
    op.create_index(op.f("ix_analysis_jobs_owner_id"), "analysis_jobs", ["owner_id"])
    op.create_index(op.f("ix_analysis_jobs_status"), "analysis_jobs", ["status"])
    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            "CREATE INDEX ix_satellite_images_footprint_gist "
            "ON satellite_images USING GIST (footprint)"
        )
        op.execute(
            "CREATE INDEX ix_analysis_jobs_flood_extent_gist "
            "ON analysis_jobs USING GIST (flood_extent)"
        )


def downgrade() -> None:
    op.drop_table("analysis_jobs")
    op.drop_table("satellite_images")
    op.drop_table("model_versions")
