from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.config import settings
from app.rag.schemas import GroundedCitation

SourceLabel = Annotated[str, Field(pattern=r"^(S|A|B)\d+$", min_length=2, max_length=12)]
ExtractionStatus = Literal["found", "missing", "ambiguous"]


class DocumentSummaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    style: Literal["executive", "detailed"] = "executive"
    max_sections: int = Field(
        default=min(8, settings.document_analysis_max_output_items),
        ge=1,
        le=settings.document_analysis_max_output_items,
    )


class GeneratedSummarySection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heading: str = Field(min_length=1, max_length=120)
    summary: str = Field(min_length=1, max_length=2_000)
    source_ids: list[SourceLabel] = Field(min_length=1, max_length=40)


class GeneratedSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sections: list[GeneratedSummarySection] = Field(
        default_factory=list,
        max_length=settings.document_analysis_max_output_items,
    )
    insufficient_evidence: bool


class DocumentSummarySection(BaseModel):
    heading: str
    summary: str
    citations: list[GroundedCitation]


class DocumentSummaryResponse(BaseModel):
    document_id: UUID
    version_id: UUID
    filename: str
    version_number: int
    sections: list[DocumentSummarySection]
    insufficient_evidence: bool


class DocumentOverviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: Literal["document_overview"] = "document_overview"


class GeneratedFactValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(min_length=1, max_length=500)
    source_ids: list[SourceLabel] = Field(min_length=1, max_length=20)

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized


class GeneratedOverviewField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ExtractionStatus
    values: list[GeneratedFactValue] = Field(default_factory=list, max_length=40)
    alternatives: list[GeneratedFactValue] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_status_values(self) -> "GeneratedOverviewField":
        if self.status == "found" and (not self.values or self.alternatives):
            raise ValueError("found fields require values and no alternatives")
        if self.status == "missing" and (self.values or self.alternatives):
            raise ValueError("missing fields cannot contain values")
        if self.status == "ambiguous" and (self.values or len(self.alternatives) < 2):
            raise ValueError("ambiguous fields require at least two alternatives")
        return self


class GeneratedDocumentOverview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: GeneratedOverviewField
    document_type: GeneratedOverviewField
    parties: GeneratedOverviewField
    dates: GeneratedOverviewField
    amounts: GeneratedOverviewField
    obligations_or_requirements: GeneratedOverviewField
    insufficient_evidence: bool


class DocumentOverviewField(BaseModel):
    status: ExtractionStatus
    values: list["CitedFactValue"]
    alternatives: list["CitedFactValue"]


class CitedFactValue(BaseModel):
    value: str
    citations: list[GroundedCitation]


class DocumentOverviewFields(BaseModel):
    title: DocumentOverviewField
    document_type: DocumentOverviewField
    parties: DocumentOverviewField
    dates: DocumentOverviewField
    amounts: DocumentOverviewField
    obligations_or_requirements: DocumentOverviewField


class DocumentOverviewResponse(BaseModel):
    document_id: UUID
    version_id: UUID
    filename: str
    version_number: int
    fields: DocumentOverviewFields
    insufficient_evidence: bool


class GeneratedComparisonFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=1, max_length=1_000)
    source_ids_a: list[SourceLabel] = Field(default_factory=list, max_length=20)
    source_ids_b: list[SourceLabel] = Field(default_factory=list, max_length=20)


class GeneratedComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    similarities: list[GeneratedComparisonFinding] = Field(
        default_factory=list,
        max_length=settings.document_analysis_max_output_items,
    )
    differences: list[GeneratedComparisonFinding] = Field(
        default_factory=list,
        max_length=settings.document_analysis_max_output_items,
    )
    document_a_only: list[GeneratedComparisonFinding] = Field(
        default_factory=list,
        max_length=settings.document_analysis_max_output_items,
    )
    document_b_only: list[GeneratedComparisonFinding] = Field(
        default_factory=list,
        max_length=settings.document_analysis_max_output_items,
    )
    insufficient_evidence: bool


class DocumentComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_a_id: UUID
    version_a_id: UUID
    document_b_id: UUID
    version_b_id: UUID
    detail_level: Literal["concise", "detailed"] = "concise"

    @model_validator(mode="after")
    def reject_identical_version_pair(self) -> "DocumentComparisonRequest":
        if self.document_a_id == self.document_b_id and self.version_a_id == self.version_b_id:
            raise ValueError("the same document version cannot be compared with itself")
        return self


class ComparisonFinding(BaseModel):
    statement: str
    citations_a: list[GroundedCitation]
    citations_b: list[GroundedCitation]


class DocumentComparisonResponse(BaseModel):
    document_a_id: UUID
    version_a_id: UUID
    filename_a: str
    version_number_a: int
    document_b_id: UUID
    version_b_id: UUID
    filename_b: str
    version_number_b: int
    similarities: list[ComparisonFinding]
    differences: list[ComparisonFinding]
    document_a_only: list[ComparisonFinding]
    document_b_only: list[ComparisonFinding]
    insufficient_evidence: bool