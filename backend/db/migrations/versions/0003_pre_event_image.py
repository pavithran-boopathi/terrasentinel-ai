"""optional pre-event image for change detection

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("analysis_jobs") as b:
        b.add_column(sa.Column("pre_image_id", sa.Uuid(), nullable=True))
        b.create_foreign_key(
            op.f("fk_analysis_jobs_pre_image_id_satellite_images"),
            "satellite_images",
            ["pre_image_id"],
            ["id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("analysis_jobs") as b:
        b.drop_constraint(
            op.f("fk_analysis_jobs_pre_image_id_satellite_images"), type_="foreignkey"
        )
        b.drop_column("pre_image_id")
