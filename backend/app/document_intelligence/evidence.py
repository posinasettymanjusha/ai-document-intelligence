from dataclasses import dataclass
from time import monotonic
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.core.errors import AppError
from app.documents.models import DocumentProcessingStatus
from app.documents.repository import DocumentRepository


@dataclass(frozen=True)
class AnalysisEvidence:
    source_id: str
    document_id: UUID
    filename: str
    version_id: UUID
    version_number: int
    chunk_id: UUID
    chunk_index: int
    text: str
    estimated_token_count: int
    source_metadata: dict[str, object]


@dataclass(frozen=True)
class AnalysisDocument:
    document_id: UUID
    version_id: UUID
    filename: str
    version_number: int


class AnalysisBudget:
    def __init__(self) -> None:
        self.started_at = monotonic()
        self.chunk_count = 0
        self.source_estimated_input_tokens = 0
        self.provider_estimated_input_tokens = 0
        self.provider_calls = 0

    def add_page(self, evidence: list[AnalysisEvidence]) -> None:
        next_chunks = self.chunk_count + len(evidence)
        next_tokens = self.source_estimated_input_tokens + sum(
            item.estimated_token_count for item in evidence
        )
        if (
            next_chunks > settings.document_analysis_max_chunks
            or next_tokens > settings.document_analysis_max_estimated_input_tokens
        ):
            raise AppError(
                413,
                "document_analysis_budget_exceeded",
                "The document exceeds the configured synchronous analysis budget.",
            )
        self.chunk_count = next_chunks
        self.source_estimated_input_tokens = next_tokens

    def add_provider_input(self, estimated_tokens: int) -> None:
        if estimated_tokens > settings.document_analysis_batch_estimated_tokens:
            raise AppError(
                413,
                "document_analysis_budget_exceeded",
                "An analysis request exceeds the configured per-call token budget.",
            )
        next_total = self.provider_estimated_input_tokens + estimated_tokens
        if next_total > settings.document_analysis_max_estimated_input_tokens:
            raise AppError(
                413,
                "document_analysis_budget_exceeded",
                "The analysis exceeds the configured provider input-token budget.",
            )
        self.provider_estimated_input_tokens = next_total

    def next_provider_timeout(self) -> float:
        elapsed = monotonic() - self.started_at
        remaining = settings.document_analysis_max_duration_seconds - elapsed
        if remaining <= 0:
            raise AppError(
                504,
                "document_analysis_timeout",
                "Document analysis exceeded its configured time limit.",
            )
        if self.provider_calls >= settings.document_analysis_max_provider_calls:
            raise AppError(
                413,
                "document_analysis_budget_exceeded",
                "The analysis requires more provider calls than the configured limit.",
            )
        self.provider_calls += 1
        return min(float(settings.gemini_generation_timeout_seconds), remaining)

    def check_deadline(self) -> None:
        if monotonic() - self.started_at > settings.document_analysis_max_duration_seconds:
            raise AppError(
                504,
                "document_analysis_timeout",
                "Document analysis exceeded its configured time limit.",
            )


class DocumentEvidenceService:
    def __init__(self, repository: DocumentRepository) -> None:
        self._repository = repository

    def authorize_version(
        self,
        workspace_id: UUID,
        document_id: UUID,
        version_id: UUID,
        user_id: str,
    ) -> tuple[UUID, AnalysisDocument]:
        try:
            user_uuid = UUID(user_id)
        except ValueError as error:
            raise AppError(401, "unauthorized", "A valid user identity is required.") from error

        try:
            document = self._repository.get_for_user(document_id, user_uuid)
            if document is None or document.workspace_id != workspace_id:
                self._not_found()
            version = next(
                (item for item in document.versions if item.id == version_id),
                None,
            )
            if version is None or version.document_id != document.id:
                self._not_found()
            if (
                version.processing_status != DocumentProcessingStatus.READY.value
            ):
                raise AppError(
                    409,
                    "document_not_ready",
                    "The requested document version is not ready for analysis.",
                )
            if not self._repository.user_can_access_version(
                document_id,
                version_id,
                user_uuid,
            ):
                self._not_found()
            self._repository.commit()
        except AppError:
            raise
        except SQLAlchemyError as error:
            self._repository.rollback()
            raise AppError(
                503,
                "persistence_unavailable",
                "Document access could not be verified.",
            ) from error

        metadata = AnalysisDocument(
            document_id=document.id,
            version_id=version.id,
            filename=document.filename,
            version_number=version.version_number,
        )
        return user_uuid, metadata

    def load_version_chunks(
        self,
        workspace_id: UUID,
        user_uuid: UUID,
        metadata: AnalysisDocument,
        budget: AnalysisBudget,
        *,
        source_prefix: str = "S",
    ) -> list[AnalysisEvidence]:
        evidence: list[AnalysisEvidence] = []
        after_chunk_index: int | None = None
        page_size = min(settings.document_analysis_max_chunks + 1, 100)
        try:
            while True:
                rows = self._repository.list_version_analysis_chunk_page(
                    workspace_id=workspace_id,
                    document_id=metadata.document_id,
                    version_id=metadata.version_id,
                    user_id=user_uuid,
                    after_chunk_index=after_chunk_index,
                    limit=page_size,
                )
                if not rows:
                    break
                page = [
                    AnalysisEvidence(
                        source_id=f"{source_prefix}{len(evidence) + index}",
                        document_id=row[1].id,
                        filename=row[1].filename,
                        version_id=row[2].id,
                        version_number=row[2].version_number,
                        chunk_id=row[0].id,
                        chunk_index=row[0].chunk_index,
                        text=row[0].text,
                        estimated_token_count=row[0].estimated_token_count,
                        source_metadata=dict(row[0].source_metadata),
                    )
                    for index, row in enumerate(rows, start=1)
                ]
                budget.add_page(page)
                evidence.extend(page)
                budget.check_deadline()
                after_chunk_index = rows[-1][0].chunk_index
                if len(rows) < page_size:
                    break
        except AppError:
            raise
        except SQLAlchemyError as error:
            self._repository.rollback()
            raise AppError(
                503,
                "persistence_unavailable",
                "Document chunks could not be loaded for analysis.",
            ) from error
        return evidence

    def load_version(
        self,
        workspace_id: UUID,
        document_id: UUID,
        version_id: UUID,
        user_id: str,
        budget: AnalysisBudget,
        *,
        source_prefix: str = "S",
    ) -> tuple[AnalysisDocument, list[AnalysisEvidence]]:
        user_uuid, metadata = self.authorize_version(
            workspace_id,
            document_id,
            version_id,
            user_id,
        )
        evidence = self.load_version_chunks(
            workspace_id,
            user_uuid,
            metadata,
            budget,
            source_prefix=source_prefix,
        )
        return metadata, evidence

    @staticmethod
    def _not_found() -> None:
        raise AppError(404, "not_found", "The requested resource was not found.")