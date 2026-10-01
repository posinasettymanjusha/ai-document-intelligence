import logging
from pathlib import PurePosixPath

from app.core.config import settings
from app.documents.errors import DocumentProcessingError
from app.documents.models import (
    DocumentProcessingStatus,
    ProcessedDocument,
    ValidatedDocument,
)
from app.integrations.document_extractors import (
    CANONICAL_CONTENT_TYPES,
    EXPECTED_CONTENT_TYPES,
    EXTRACTORS,
)

logger = logging.getLogger(__name__)


class DocumentProcessingService:
    def __init__(self, max_file_size_bytes: int | None = None) -> None:
        self._max_file_size_bytes = (
            max_file_size_bytes
            if max_file_size_bytes is not None
            else settings.max_document_size_bytes
        )
        if self._max_file_size_bytes < 1:
            raise ValueError("max_file_size_bytes must be greater than zero")

    def process(
        self,
        filename: str,
        content_type: str | None,
        content: bytes,
    ) -> ProcessedDocument:
        return self.extract(self.validate(filename, content_type, content))

    def validate(
        self,
        filename: str,
        content_type: str | None,
        content: bytes,
    ) -> ValidatedDocument:
        if not content:
            raise DocumentProcessingError(422, "empty_file", "The uploaded file is empty.")
        if len(content) > self._max_file_size_bytes:
            raise DocumentProcessingError(
                413,
                "file_too_large",
                f"The file exceeds the {self._max_file_size_bytes}-byte size limit.",
            )

        basename = PurePosixPath(filename.replace("\\", "/")).name if filename else ""
        extension = PurePosixPath(basename).suffix.casefold()
        if extension not in EXTRACTORS:
            raise DocumentProcessingError(
                415,
                "unsupported_file_type",
                "Supported file extensions are .pdf, .docx, and .txt.",
            )
        if len(basename) > 255:
            raise DocumentProcessingError(422, "filename_too_long", "The filename is too long.")

        supplied_type = (content_type or "").split(";", maxsplit=1)[0].strip().casefold()
        if supplied_type and supplied_type not in EXPECTED_CONTENT_TYPES[extension]:
            raise DocumentProcessingError(
                415,
                "content_type_mismatch",
                "The content type does not match the file extension.",
            )

        return ValidatedDocument(
            filename=basename,
            extension=extension,
            content_type=CANONICAL_CONTENT_TYPES[extension],
            size_bytes=len(content),
            content=content,
        )

    def extract(self, document: ValidatedDocument) -> ProcessedDocument:
        try:
            segments = EXTRACTORS[document.extension].extract(document.content)
            if not any(segment.text.strip() for segment in segments):
                raise DocumentProcessingError(
                    422,
                    "no_extractable_text",
                    "No readable text was found in the uploaded document.",
                )
        except DocumentProcessingError:
            raise
        except Exception as error:
            logger.warning(
                "Document extraction failed",
                extra={
                    "document_filename": document.filename,
                    "extension": document.extension,
                },
                exc_info=error,
            )
            raise DocumentProcessingError(
                422,
                "extraction_failed",
                "The document could not be parsed. Verify that it is a valid, unencrypted file.",
            ) from error

        return ProcessedDocument(
            filename=document.filename,
            content_type=document.content_type,
            size_bytes=document.size_bytes,
            status=DocumentProcessingStatus.READY,
            segments=segments,
            extraction_metadata={
                "segment_count": len(segments),
                "format": document.extension.lstrip("."),
            },
        )