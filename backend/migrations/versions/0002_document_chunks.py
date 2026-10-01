"""Create retrieval-ready document chunks."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002_document_chunks"
down_revision = "0001_document_persistence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "document_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("character_count", sa.Integer(), nullable=False),
        sa.Column("estimated_token_count", sa.Integer(), nullable=False),
        sa.Column("source_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("chunk_index >= 0", name="ck_document_chunks_nonnegative_index"),
        sa.CheckConstraint("character_count > 0", name="ck_document_chunks_positive_characters"),
        sa.CheckConstraint("estimated_token_count > 0", name="ck_document_chunks_positive_tokens"),
        sa.ForeignKeyConstraint(
            ["document_version_id"], ["document_versions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_version_id", "chunk_index", name="uq_document_chunks_version_index"
        ),
    )

def downgrade() -> None:
    op.drop_table("document_chunks")