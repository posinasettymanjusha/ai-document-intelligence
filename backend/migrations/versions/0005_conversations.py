"""Add private conversations and ordered conversation messages."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0005_conversations"
down_revision = "0004_embedding_failure_details"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_documents_workspace_id_id",
        "documents",
        ["workspace_id", "id"],
    )
    op.create_unique_constraint(
        "uq_document_versions_document_id_id",
        "document_versions",
        ["document_id", "id"],
    )
    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("owner_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "version_id IS NULL OR document_id IS NOT NULL",
            name="ck_conversations_version_requires_document",
        ),
        sa.ForeignKeyConstraint(
            ["owner_user_id"],
            ["profiles.id"],
            name="fk_conversations_owner_user_id_profiles",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name="fk_conversations_workspace_id_workspaces",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "document_id"],
            ["documents.workspace_id", "documents.id"],
            name="fk_conversations_workspace_document",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["document_id", "version_id"],
            ["document_versions.document_id", "document_versions.id"],
            name="fk_conversations_document_version",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_conversations"),
    )
    op.create_index(
        "ix_conversations_workspace_owner_updated",
        "conversations",
        ["workspace_id", "owner_user_id", "updated_at"],
    )
    op.create_table(
        "conversation_messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("sequence", sa.BigInteger(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="pending", nullable=False),
        sa.Column(
            "citations",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("insufficient_context", sa.Boolean(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint("sequence > 0", name="ck_conversation_messages_positive_sequence"),
        sa.CheckConstraint("role IN ('user', 'assistant')", name="ck_conversation_messages_role"),
        sa.CheckConstraint(
            "status IN ('pending', 'complete', 'failed')",
            name="ck_conversation_messages_status",
        ),
        sa.CheckConstraint("char_length(btrim(content)) > 0", name="ck_conversation_messages_content"),
        sa.CheckConstraint(
            "(role = 'user' AND insufficient_context IS NULL AND citations = '[]'::jsonb) OR "
            "(role = 'assistant' AND status = 'complete' AND insufficient_context IS NOT NULL)",
            name="ck_conversation_messages_role_fields",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_conversation_messages_conversation_id_conversations",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_conversation_messages"),
        sa.UniqueConstraint(
            "conversation_id",
            "sequence",
            name="uq_conversation_messages_conversation_sequence",
        ),
    )
    op.create_index(
        "ix_conversation_messages_conversation_status",
        "conversation_messages",
        ["conversation_id", "status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_conversation_messages_conversation_status",
        table_name="conversation_messages",
    )
    op.drop_table("conversation_messages")
    op.drop_index(
        "ix_conversations_workspace_owner_updated",
        table_name="conversations",
    )
    op.drop_table("conversations")
    op.drop_constraint(
        "uq_document_versions_document_id_id",
        "document_versions",
        type_="unique",
    )
    op.drop_constraint(
        "uq_documents_workspace_id_id",
        "documents",
        type_="unique",
    )