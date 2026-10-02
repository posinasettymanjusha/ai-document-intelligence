import re
from collections.abc import Mapping, Sequence
from typing import Protocol

from app.rag.schemas import GroundedCitation

_SOURCE_LABEL = re.compile(r"\[((?:S|A|B)\d+)\]")


class CitationSource(Protocol):
    document_id: object
    filename: str
    version_id: object
    version_number: int
    chunk_id: object
    chunk_index: int
    source_metadata: dict[str, object]


class InvalidCitationLabels(ValueError):
    pass


def sanitize_analysis_text(text: str, sources: Mapping[str, CitationSource]) -> str:
    labels = _SOURCE_LABEL.findall(text)
    if any(label not in sources for label in labels):
        raise InvalidCitationLabels("Generated text included an unknown source label.")
    return _SOURCE_LABEL.sub("", text).strip()


def clean_answer_citations(
    answer: str,
    sources: Mapping[str, CitationSource],
) -> tuple[str, list[str]]:
    mentioned = list(dict.fromkeys(_SOURCE_LABEL.findall(answer)))
    allowed = [label for label in mentioned if label in sources]
    cleaned = _SOURCE_LABEL.sub(
        lambda match: match.group(0) if match.group(1) in sources else "",
        answer,
    ).strip()
    return cleaned, allowed


def map_source_ids_to_citations(
    source_ids: Sequence[str],
    sources: Mapping[str, CitationSource],
    *,
    required: bool = True,
) -> list[GroundedCitation]:
    labels = list(dict.fromkeys(source_ids))
    if any(label not in sources for label in labels):
        raise InvalidCitationLabels("A generated source label was not part of this evidence set.")
    if required and not labels:
        raise InvalidCitationLabels("A grounded value must cite at least one source.")
    return [make_grounded_citation(label, sources[label]) for label in labels]


def make_grounded_citation(source_id: str, source: CitationSource) -> GroundedCitation:
    metadata = dict(source.source_metadata)
    raw_pages = metadata.get("page_numbers", [])
    page_numbers = (
        [page for page in raw_pages if isinstance(page, int) and not isinstance(page, bool)]
        if isinstance(raw_pages, list)
        else []
    )
    return GroundedCitation(
        source_id=source_id,
        document_id=source.document_id,
        filename=source.filename,
        version_id=source.version_id,
        version_number=source.version_number,
        chunk_id=source.chunk_id,
        chunk_index=source.chunk_index,
        page_numbers=page_numbers,
        source_metadata=metadata,
    )