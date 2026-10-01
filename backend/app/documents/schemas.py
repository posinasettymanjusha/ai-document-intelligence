from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.documents.models import DocumentProcessingStatus


class ExtractedTextSegmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    sequence: int = Field(ge=1)
    text: str
    page_number: int | None = Field(default=None, ge=1)
    paragraph_index: int | None = Field(default=None, ge=1)
    paragraph_style: str | None = None
    line_number: int | None = Field(default=None, ge=1)
    table_index: int | None = Field(default=None, ge=1)
    table_row_index: int | None = Field(default=None, ge=1)
    table_cell_index: int | None = Field(default=None, ge=1)


class DocumentProcessingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    filename: str
    content_type: str
    size_bytes: int = Field(ge=1)
    status: DocumentProcessingStatus
    segments: list[ExtractedTextSegmentResponse]


class DocumentVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    version_number: int = Field(ge=1)
    checksum: str
    original_filename: str
    file_type: str
    mime_type: str
    file_size: int = Field(ge=1)
    processing_status: DocumentProcessingStatus
    extracted_content: list[ExtractedTextSegmentResponse]
    extraction_metadata: dict[str, object]
    created_at: datetime
    updated_at: datetime


class DocumentListItemResponse(BaseModel):
    id: UUID
    workspace_id: UUID
    filename: str
    file_type: str
    mime_type: str
    file_size: int = Field(ge=1)
    processing_status: DocumentProcessingStatus
    current_version: int | None
    created_at: datetime
    updated_at: datetime


class DocumentResponse(DocumentListItemResponse):
    versions: list[DocumentVersionResponse]