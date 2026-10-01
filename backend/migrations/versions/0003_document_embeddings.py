"""Add configurable pgvector embeddings to document chunks."""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from app.core.config import settings

revision = "0003_document_embeddings"
down_revision = "0002_document_chunks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column(
        "document_chunks",
        sa.Column("embedding", Vector(settings.embedding_dimension), nullable=True),
    )
    op.add_column(
        "document_chunks",
        sa.Column("embedding_model", sa.String(length=120), nullable=True),
    )
    op.add_column(
        "document_chunks",
        sa.Column("embedding_dimension", sa.Integer(), nullable=True),
    )
    op.add_column(
        "document_chunks",
        sa.Column(
            "embedding_status",
            sa.String(length=16),
            server_default="pending",
            nullable=False,
        ),
    )
    op.add_column(
        "document_chunks",
        sa.Column("embedding_generated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        "ck_document_chunks_embedding_status",
        "document_chunks",
        "embedding_status IN ('pending', 'processing', 'ready', 'failed')",
    )
    op.create_check_constraint(
        "ck_document_chunks_embedding_dimension",
        "document_chunks",
        f"embedding_dimension IS NULL OR embedding_dimension = {settings.embedding_dimension}",
    )
    op.create_check_constraint(
        "ck_document_chunks_ready_embedding_complete",
        "document_chunks",
        "embedding_status <> 'ready' OR "
        "(embedding IS NOT NULL AND embedding_dimension IS NOT NULL "
        "AND embedding_model IS NOT NULL AND embedding_generated_at IS NOT NULL)",
    )
    op.create_index(
        "ix_document_chunks_embedding_hnsw_cosine",
        "document_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_where=sa.text("embedding_status = 'ready' AND embedding IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_document_chunks_embedding_hnsw_cosine", table_name="document_chunks"
    )
    op.drop_constraint(
        "ck_document_chunks_ready_embedding_complete", "document_chunks", type_="check"
    )
    op.drop_constraint("ck_document_chunks_embedding_dimension", "document_chunks", type_="check")
    op.drop_constraint("ck_document_chunks_embedding_status", "document_chunks", type_="check")
    op.drop_column("document_chunks", "embedding_generated_at")
    op.drop_column("document_chunks", "embedding_status")
    op.drop_column("document_chunks", "embedding_dimension")
    op.drop_column("document_chunks", "embedding_model")
    op.drop_column("document_chunks", "embedding")