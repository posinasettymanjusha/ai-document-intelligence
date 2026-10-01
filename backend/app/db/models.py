from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
    func,
    text as sql_text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from app.core.config import settings
from app.db.base import Base
from app.embeddings.models import EmbeddingStatus


class Profile(Base):
    __tablename__ = "profiles"

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("auth.users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    display_name: Mapped[str | None] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    created_by: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("profiles.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    members: Mapped[list["WorkspaceMember"]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan"
    )


class WorkspaceMember(Base):
    __tablename__ = "workspace_members"
    __table_args__ = (
        UniqueConstraint("workspace_id", "user_id", name="uq_workspace_members_workspace_user"),
        CheckConstraint("role IN ('owner', 'admin', 'member')", name="ck_workspace_members_role"),
        Index("ix_workspace_members_user_workspace", "user_id", "workspace_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="member")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    workspace: Mapped[Workspace] = relationship(back_populates="members")


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("workspace_id", "filename", name="uq_documents_workspace_filename"),
        CheckConstraint("file_type IN ('pdf', 'docx', 'txt')", name="ck_documents_file_type"),
        CheckConstraint(
            "processing_status IN ('uploaded', 'processing', 'ready', 'failed')",
            name="ck_documents_processing_status",
        ),
        Index("ix_documents_workspace_created", "workspace_id", "created_at"),
        Index("ix_documents_workspace_status", "workspace_id", "processing_status"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(8), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(127), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    storage_path: Mapped[str] = mapped_column(String(700), nullable=False)
    processing_status: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    versions: Mapped[list["DocumentVersion"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentVersion.version_number",
    )


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("document_id", "version_number", name="uq_document_versions_number"),
        UniqueConstraint("document_id", "checksum", name="uq_document_versions_checksum"),
        UniqueConstraint("storage_path", name="uq_document_versions_storage_path"),
        CheckConstraint("version_number > 0", name="ck_document_versions_positive_number"),
        CheckConstraint("file_type IN ('pdf', 'docx', 'txt')", name="ck_document_versions_file_type"),
        CheckConstraint(
            "processing_status IN ('uploaded', 'processing', 'ready', 'failed')",
            name="ck_document_versions_processing_status",
        ),
        Index("ix_document_versions_document_created", "document_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(8), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(127), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    storage_path: Mapped[str] = mapped_column(String(700), nullable=False)
    processing_status: Mapped[str] = mapped_column(String(16), nullable=False)
    extracted_content: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    extraction_metadata: Mapped[dict[str, object]] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    document: Mapped[Document] = relationship(back_populates="versions")
    chunks: Mapped[list["DocumentChunkRecord"]] = relationship(
        back_populates="document_version",
        cascade="all, delete-orphan",
        order_by="DocumentChunkRecord.chunk_index",
    )


class DocumentChunkRecord(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint(
            "document_version_id",
            "chunk_index",
            name="uq_document_chunks_version_index",
        ),
        CheckConstraint("chunk_index >= 0", name="ck_document_chunks_nonnegative_index"),
        CheckConstraint("character_count > 0", name="ck_document_chunks_positive_characters"),
        CheckConstraint("estimated_token_count > 0", name="ck_document_chunks_positive_tokens"),
        CheckConstraint(
            "embedding_status IN ('pending', 'processing', 'ready', 'failed')",
            name="ck_document_chunks_embedding_status",
        ),
        CheckConstraint(
            f"embedding_dimension IS NULL OR embedding_dimension = {settings.embedding_dimension}",
            name="ck_document_chunks_embedding_dimension",
        ),
        CheckConstraint(
            "embedding_status <> 'ready' OR "
            "(embedding IS NOT NULL AND embedding_dimension IS NOT NULL "
            "AND embedding_model IS NOT NULL AND embedding_generated_at IS NOT NULL)",
            name="ck_document_chunks_ready_embedding_complete",
        ),
        Index(
            "ix_document_chunks_embedding_hnsw_cosine",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_where=sql_text("embedding_status = 'ready' AND embedding IS NOT NULL"),
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    document_version_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(String, nullable=False)
    character_count: Mapped[int] = mapped_column(Integer, nullable=False)
    estimated_token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    source_metadata: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.embedding_dimension), nullable=True
    )
    embedding_model: Mapped[str | None] = mapped_column(String(120))
    embedding_dimension: Mapped[int | None] = mapped_column(Integer)
    embedding_error: Mapped[str | None] = mapped_column(String(500))
    embedding_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=EmbeddingStatus.PENDING.value,
        server_default=sql_text("'pending'"),
    )
    embedding_generated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document_version: Mapped[DocumentVersion] = relationship(back_populates="chunks")