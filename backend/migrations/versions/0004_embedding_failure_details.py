"""Persist safe failure details for embedding attempts."""

from alembic import op
import sqlalchemy as sa

revision = "0004_embedding_failure_details"
down_revision = "0003_document_embeddings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "document_chunks",
        sa.Column("embedding_error", sa.String(length=500), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("document_chunks", "embedding_error")