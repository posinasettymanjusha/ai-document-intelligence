from dataclasses import dataclass
from enum import StrEnum


class DocumentProcessingStatus(StrEnum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class ExtractedTextSegment:
    sequence: int
    text: str
    page_number: int | None = None
    paragraph_index: int | None = None
    paragraph_style: str | None = None
    line_number: int | None = None
    table_index: int | None = None
    table_row_index: int | None = None
    table_cell_index: int | None = None


@dataclass(frozen=True)
class ValidatedDocument:
    filename: str
    extension: str
    content_type: str
    size_bytes: int
    content: bytes


@dataclass(frozen=True)
class ProcessedDocument:
    filename: str
    content_type: str
    size_bytes: int
    status: DocumentProcessingStatus
    segments: list[ExtractedTextSegment]
    extraction_metadata: dict[str, object]