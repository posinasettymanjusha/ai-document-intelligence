"""Create initial workspace and document persistence schema."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_document_persistence"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "profiles",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("display_name", sa.String(length=120), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["id"], ["auth.users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "workspaces",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["created_by"], ["profiles.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "workspace_members",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("role IN ('owner', 'admin', 'member')", name="ck_workspace_members_role"),
        sa.ForeignKeyConstraint(["user_id"], ["profiles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("workspace_id", "user_id", name="uq_workspace_members_workspace_user"),
    )
    op.create_index(
        "ix_workspace_members_user_workspace", "workspace_members", ["user_id", "workspace_id"]
    )
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("file_type", sa.String(length=8), nullable=False),
        sa.Column("mime_type", sa.String(length=127), nullable=False),
        sa.Column("file_size", sa.BigInteger(), nullable=False),
        sa.Column("storage_path", sa.String(length=700), nullable=False),
        sa.Column("processing_status", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("file_type IN ('pdf', 'docx', 'txt')", name="ck_documents_file_type"),
        sa.CheckConstraint(
            "processing_status IN ('uploaded', 'processing', 'ready', 'failed')",
            name="ck_documents_processing_status",
        ),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("workspace_id", "filename", name="uq_documents_workspace_filename"),
    )
    op.create_index("ix_documents_workspace_created", "documents", ["workspace_id", "created_at"])
    op.create_index("ix_documents_workspace_status", "documents", ["workspace_id", "processing_status"])
    op.create_table(
        "document_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("checksum", sa.String(length=64), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("file_type", sa.String(length=8), nullable=False),
        sa.Column("mime_type", sa.String(length=127), nullable=False),
        sa.Column("file_size", sa.BigInteger(), nullable=False),
        sa.Column("storage_path", sa.String(length=700), nullable=False),
        sa.Column("processing_status", sa.String(length=16), nullable=False),
        sa.Column("extracted_content", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("extraction_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("version_number > 0", name="ck_document_versions_positive_number"),
        sa.CheckConstraint("file_type IN ('pdf', 'docx', 'txt')", name="ck_document_versions_file_type"),
        sa.CheckConstraint(
            "processing_status IN ('uploaded', 'processing', 'ready', 'failed')",
            name="ck_document_versions_processing_status",
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "checksum", name="uq_document_versions_checksum"),
        sa.UniqueConstraint("document_id", "version_number", name="uq_document_versions_number"),
        sa.UniqueConstraint("storage_path", name="uq_document_versions_storage_path"),
    )
    op.create_index(
        "ix_document_versions_document_created", "document_versions", ["document_id", "created_at"]
    )
    op.execute(
        "INSERT INTO storage.buckets (id, name, public) VALUES ('documents', 'documents', FALSE) "
        "ON CONFLICT (id) DO UPDATE SET public = FALSE"
    )
    op.execute(
        "INSERT INTO public.profiles (id) SELECT id FROM auth.users "
        "ON CONFLICT (id) DO NOTHING"
    )
    op.execute(
        "CREATE FUNCTION public.handle_new_auth_user() RETURNS trigger "
        "LANGUAGE plpgsql SECURITY DEFINER SET search_path = '' AS $$ "
        "BEGIN INSERT INTO public.profiles (id, display_name) "
        "VALUES (NEW.id, NEW.raw_user_meta_data ->> 'name') "
        "ON CONFLICT (id) DO NOTHING; RETURN NEW; END; $$"
    )
    op.execute(
        "CREATE TRIGGER on_auth_user_created AFTER INSERT ON auth.users "
        "FOR EACH ROW EXECUTE FUNCTION public.handle_new_auth_user()"
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users")
    op.execute("DROP FUNCTION IF EXISTS public.handle_new_auth_user()")
    op.drop_index("ix_document_versions_document_created", table_name="document_versions")
    op.drop_table("document_versions")
    op.drop_index("ix_documents_workspace_status", table_name="documents")
    op.drop_index("ix_documents_workspace_created", table_name="documents")
    op.drop_table("documents")
    op.drop_index("ix_workspace_members_user_workspace", table_name="workspace_members")
    op.drop_table("workspace_members")
    op.drop_table("workspaces")
    op.drop_table("profiles")
    op.execute(
        "DELETE FROM storage.buckets WHERE id = 'documents' "
        "AND NOT EXISTS (SELECT 1 FROM storage.objects WHERE bucket_id = 'documents')"
    )