from datetime import UTC, datetime
import math
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload

from app.chunking.models import DocumentChunk
from app.core.config import settings
from app.db.models import Document, DocumentChunkRecord, DocumentVersion, WorkspaceMember
from app.embeddings.models import EmbeddingResult, EmbeddingStatus
from app.documents.models import DocumentProcessingStatus, ProcessedDocument


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def is_workspace_member(self, workspace_id: UUID, user_id: UUID) -> bool:
        query = select(WorkspaceMember.id).where(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == user_id,
        )
        return self._session.scalar(query) is not None

    def user_can_access_version(
        self,
        document_id: UUID,
        version_id: UUID,
        user_id: UUID,
    ) -> bool:
        query = (
            select(DocumentVersion.id)
            .join(Document, Document.id == DocumentVersion.document_id)
            .join(WorkspaceMember, WorkspaceMember.workspace_id == Document.workspace_id)
            .where(
                Document.id == document_id,
                DocumentVersion.id == version_id,
                Document.deleted_at.is_(None),
                WorkspaceMember.user_id == user_id,
            )
        )
        return self._session.scalar(query) is not None

    def search_chunks(
        self,
        *,
        workspace_id: UUID,
        user_id: UUID,
        query_vector: list[float],
        embedding_model: str,
        top_k: int,
        document_id: UUID | None = None,
        version_id: UUID | None = None,
    ) -> list[tuple[DocumentChunkRecord, Document, DocumentVersion, float]]:
        distance = DocumentChunkRecord.embedding.cosine_distance(query_vector)
        query = (
            select(DocumentChunkRecord, Document, DocumentVersion, distance)
            .join(
                DocumentVersion,
                DocumentVersion.id == DocumentChunkRecord.document_version_id,
            )
            .join(Document, Document.id == DocumentVersion.document_id)
            .join(WorkspaceMember, WorkspaceMember.workspace_id == Document.workspace_id)
            .where(
                Document.workspace_id == workspace_id,
                WorkspaceMember.user_id == user_id,
                Document.deleted_at.is_(None),
                DocumentChunkRecord.embedding_status == EmbeddingStatus.READY.value,
                DocumentChunkRecord.embedding.is_not(None),
                DocumentChunkRecord.embedding_model == embedding_model,
            )
            .order_by(distance.asc())
            .limit(top_k)
        )
        if document_id is not None:
            query = query.where(Document.id == document_id)
        if version_id is not None:
            query = query.where(DocumentVersion.id == version_id)

        return list(self._session.execute(query).all())

    def claim_embedding_batch(
        self,
        version_id: UUID,
        model_id: str,
        batch_size: int,
        *,
        retry_failed: bool,
    ) -> list[DocumentChunkRecord]:
        self._validate_model_id(model_id)
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")

        eligible_statuses = [EmbeddingStatus.PENDING.value]
        if retry_failed:
            eligible_statuses.append(EmbeddingStatus.FAILED.value)
        query = (
            select(DocumentChunkRecord)
            .where(
                DocumentChunkRecord.document_version_id == version_id,
                DocumentChunkRecord.embedding_status.in_(eligible_statuses),
            )
            .order_by(DocumentChunkRecord.chunk_index)
            .limit(batch_size)
            .with_for_update(skip_locked=True)
        )
        chunks = list(self._session.scalars(query))
        for chunk in chunks:
            chunk.embedding = None
            chunk.embedding_dimension = None
            chunk.embedding_generated_at = None
            chunk.embedding_error = None
            chunk.embedding_model = model_id
            chunk.embedding_status = EmbeddingStatus.PROCESSING.value
        self._session.commit()
        return chunks

    def get_by_workspace_filename(self, workspace_id: UUID, filename: str) -> Document | None:
        query = (
            select(Document)
            .options(selectinload(Document.versions))
            .where(
                Document.workspace_id == workspace_id,
                Document.filename == filename,
                Document.deleted_at.is_(None),
            )
        )
        return self._session.scalar(query)

    def get_for_user(
        self,
        document_id: UUID,
        user_id: UUID,
        *,
        include_deleted: bool = False,
    ) -> Document | None:
        query = (
            select(Document)
            .join(WorkspaceMember, WorkspaceMember.workspace_id == Document.workspace_id)
            .options(selectinload(Document.versions))
            .where(Document.id == document_id, WorkspaceMember.user_id == user_id)
        )
        if not include_deleted:
            query = query.where(Document.deleted_at.is_(None))
        return self._session.scalar(query)

    def list_for_workspace(self, workspace_id: UUID) -> list[Document]:
        query = (
            select(Document)
            .options(selectinload(Document.versions))
            .where(Document.workspace_id == workspace_id, Document.deleted_at.is_(None))
            .order_by(Document.created_at.desc(), Document.id)
        )
        return list(self._session.scalars(query).unique())

    def create_version(
        self,
        workspace_id: UUID,
        document_id: UUID,
        filename: str,
        file_type: str,
        mime_type: str,
        file_size: int,
        storage_path: str,
        checksum: str,
    ) -> tuple[Document, DocumentVersion]:
        document = self.get_by_workspace_filename(workspace_id, filename)
        if document is None:
            document = Document(
                id=document_id,
                workspace_id=workspace_id,
                filename=filename,
                file_type=file_type,
                mime_type=mime_type,
                file_size=file_size,
                storage_path=storage_path,
                processing_status=DocumentProcessingStatus.UPLOADED.value,
            )
            self._session.add(document)
            self._session.flush()
        else:
            self._session.refresh(document, with_for_update=True)

        if any(version.checksum == checksum for version in document.versions):
            raise ValueError("A document version with this checksum already exists")

        version = DocumentVersion(
            document_id=document.id,
            version_number=max((item.version_number for item in document.versions), default=0) + 1,
            checksum=checksum,
            original_filename=filename,
            file_type=file_type,
            mime_type=mime_type,
            file_size=file_size,
            storage_path=storage_path,
            processing_status=DocumentProcessingStatus.UPLOADED.value,
        )
        document.file_type = file_type
        document.mime_type = mime_type
        document.file_size = file_size
        document.storage_path = storage_path
        document.processing_status = DocumentProcessingStatus.UPLOADED.value
        self._session.add(version)
        document.versions.append(version)
        self._session.commit()
        self._session.refresh(document)
        self._session.refresh(version)
        return document, version

    def set_status(
        self,
        document_id: UUID,
        version_id: UUID,
        status: DocumentProcessingStatus,
        processed: ProcessedDocument | None = None,
        chunks: list[DocumentChunk] | None = None,
    ) -> tuple[Document, DocumentVersion]:
        document = self._session.get(Document, document_id, with_for_update=True)
        version = self._session.get(DocumentVersion, version_id, with_for_update=True)
        if document is None or version is None or version.document_id != document_id:
            raise LookupError("Document version not found")

        document.processing_status = status.value
        version.processing_status = status.value
        if processed is not None:
            version.extracted_content = [
                {
                    "sequence": segment.sequence,
                    "text": segment.text,
                    "page_number": segment.page_number,
                    "paragraph_index": segment.paragraph_index,
                    "paragraph_style": segment.paragraph_style,
                    "line_number": segment.line_number,
                    "table_index": segment.table_index,
                    "table_row_index": segment.table_row_index,
                    "table_cell_index": segment.table_cell_index,
                }
                for segment in processed.segments
            ]
            version.extraction_metadata = processed.extraction_metadata
        if chunks is not None:
            self._session.execute(
                delete(DocumentChunkRecord).where(
                    DocumentChunkRecord.document_version_id == version_id
                )
            )
            self._session.add_all(
                [
                    DocumentChunkRecord(
                        document_version_id=version_id,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                        character_count=chunk.character_count,
                        estimated_token_count=chunk.estimated_token_count,
                        source_metadata=chunk.source_metadata,
                    )
                    for chunk in chunks
                ]
            )
        self._session.commit()
        self._session.refresh(document)
        self._session.refresh(version)
        return document, version

    def soft_delete(self, document: Document) -> None:
        document.deleted_at = datetime.now(UTC)
        self._session.commit()

    def delete(self, document: Document) -> None:
        self._session.delete(document)
        self._session.commit()

    def rollback(self) -> None:
        self._session.rollback()

    def mark_embedding_processing(self, chunk_id: UUID, model_id: str) -> DocumentChunkRecord:
        chunk = self._get_chunk_for_update(chunk_id)
        self._validate_model_id(model_id)
        chunk.embedding = None
        chunk.embedding_dimension = None
        chunk.embedding_generated_at = None
        chunk.embedding_model = model_id
        chunk.embedding_status = EmbeddingStatus.PROCESSING.value
        self._session.commit()
        self._session.refresh(chunk)
        return chunk

    def store_embedding(
        self,
        chunk_id: UUID,
        embedding: EmbeddingResult,
    ) -> DocumentChunkRecord:
        self._validate_embedding(embedding)
        chunk = self._get_chunk_for_update(chunk_id)
        if chunk.embedding_status == EmbeddingStatus.READY.value:
            if chunk.embedding_model == embedding.model_id and chunk.embedding == embedding.vector:
                return chunk
            raise ValueError("Ready embeddings cannot be overwritten")
        self._assign_embedding(chunk, embedding)
        self._session.commit()
        self._session.refresh(chunk)
        return chunk

    def store_embeddings(
        self,
        embeddings: Sequence[tuple[UUID, EmbeddingResult]],
    ) -> list[DocumentChunkRecord]:
        if not embeddings:
            return []
        chunk_ids = [chunk_id for chunk_id, _ in embeddings]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("A batch cannot contain duplicate chunk ids")
        for _, embedding in embeddings:
            self._validate_embedding(embedding)

        chunks = {
            chunk.id: chunk
            for chunk in self._session.scalars(
                select(DocumentChunkRecord)
                .where(DocumentChunkRecord.id.in_(chunk_ids))
                .with_for_update()
            )
        }
        if len(chunks) != len(chunk_ids):
            raise LookupError("One or more document chunks were not found")

        stored = [chunks[chunk_id] for chunk_id in chunk_ids]
        for chunk, (_, embedding) in zip(stored, embeddings, strict=True):
            if chunk.embedding_status == EmbeddingStatus.READY.value:
                raise ValueError("Ready embeddings cannot be overwritten")
            if chunk.embedding_status != EmbeddingStatus.PROCESSING.value:
                raise ValueError("Only processing chunks can be marked ready")
            if chunk.embedding_model != embedding.model_id:
                raise ValueError("Embedding model does not match the claimed model")
        for chunk, (_, embedding) in zip(stored, embeddings, strict=True):
            self._assign_embedding(chunk, embedding)
        self._session.commit()
        for chunk in stored:
            self._session.refresh(chunk)
        return stored

    def get_chunk_embedding(self, chunk_id: UUID) -> EmbeddingResult | None:
        chunk = self._session.get(DocumentChunkRecord, chunk_id)
        if chunk is None:
            raise LookupError("Document chunk not found")
        if chunk.embedding is None or chunk.embedding_status != EmbeddingStatus.READY.value:
            return None
        return EmbeddingResult(model_id=chunk.embedding_model, vector=list(chunk.embedding))

    def mark_embedding_failure(
        self,
        chunk_id: UUID,
        error_message: str = "Embedding generation failed.",
    ) -> DocumentChunkRecord:
        return self.mark_embedding_failures([chunk_id], error_message)[0]

    def mark_embedding_failures(
        self,
        chunk_ids: Sequence[UUID],
        error_message: str,
    ) -> list[DocumentChunkRecord]:
        if not chunk_ids:
            return []
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("A failure batch cannot contain duplicate chunk ids")
        chunks = {
            chunk.id: chunk
            for chunk in self._session.scalars(
                select(DocumentChunkRecord)
                .where(DocumentChunkRecord.id.in_(chunk_ids))
                .with_for_update()
            )
        }
        if len(chunks) != len(chunk_ids):
            raise LookupError("One or more document chunks were not found")
        safe_message = error_message.strip()[:500] or "Embedding generation failed."
        for chunk in chunks.values():
            if chunk.embedding_status != EmbeddingStatus.PROCESSING.value:
                continue
            chunk.embedding = None
            chunk.embedding_dimension = None
            chunk.embedding_generated_at = None
            chunk.embedding_error = safe_message
            chunk.embedding_status = EmbeddingStatus.FAILED.value
        self._session.commit()
        result = [chunks[chunk_id] for chunk_id in chunk_ids]
        for chunk in result:
            self._session.refresh(chunk)
        return result

    def _get_chunk_for_update(self, chunk_id: UUID) -> DocumentChunkRecord:
        chunk = self._session.get(DocumentChunkRecord, chunk_id, with_for_update=True)
        if chunk is None:
            raise LookupError("Document chunk not found")
        return chunk

    @staticmethod
    def _validate_model_id(model_id: str) -> None:
        if not model_id or len(model_id) > 120:
            raise ValueError("Embedding model identifier must contain 1 to 120 characters")

    @classmethod
    def _validate_embedding(cls, embedding: EmbeddingResult) -> None:
        cls._validate_model_id(embedding.model_id)
        if embedding.dimension != settings.embedding_dimension:
            raise ValueError(
                f"Embedding dimension {embedding.dimension} does not match configured "
                f"dimension {settings.embedding_dimension}"
            )
        if not all(math.isfinite(value) for value in embedding.vector):
            raise ValueError("Embedding values must all be finite numbers")

    @staticmethod
    def _assign_embedding(chunk: DocumentChunkRecord, embedding: EmbeddingResult) -> None:
        chunk.embedding = embedding.vector
        chunk.embedding_model = embedding.model_id
        chunk.embedding_dimension = embedding.dimension
        chunk.embedding_status = EmbeddingStatus.READY.value
        chunk.embedding_generated_at = datetime.now(UTC)
        chunk.embedding_error = None