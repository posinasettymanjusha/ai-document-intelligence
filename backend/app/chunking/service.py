import re
from dataclasses import dataclass

from app.chunking.models import DocumentChunk
from app.documents.models import ExtractedTextSegment

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n+")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class _TextUnit:
    text: str
    segment: ExtractedTextSegment
    part_index: int
    start: int
    end: int


class DocumentChunkingService:
    def __init__(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
        minimum_chunk_size: int,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
        if minimum_chunk_size < 1 or minimum_chunk_size > chunk_size:
            raise ValueError("minimum_chunk_size must be between one and chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.minimum_chunk_size = minimum_chunk_size

    def chunk(self, segments: list[ExtractedTextSegment]) -> list[DocumentChunk]:
        units = self._normalize_segments(segments)
        if not units:
            return []

        text, positioned_units = self._join_units(units)
        chunks: list[DocumentChunk] = []
        start = 0

        while start < len(text):
            end = self._choose_end(text, start)
            chunk_text = text[start:end].strip()
            if chunk_text:
                visible_start = start + len(text[start:end]) - len(text[start:end].lstrip())
                visible_end = visible_start + len(chunk_text)
                chunks.append(
                    DocumentChunk(
                        chunk_index=len(chunks),
                        text=chunk_text,
                        character_count=len(chunk_text),
                        estimated_token_count=(len(chunk_text) + 3) // 4,
                        source_metadata=self._source_metadata(
                            visible_start, visible_end, positioned_units
                        ),
                    )
                )

            if end >= len(text):
                break
            next_start = self._overlap_start(text, start, end)
            if next_start <= start:
                next_start = end
            start = next_start

        return chunks

    @staticmethod
    def _normalize_segments(
        segments: list[ExtractedTextSegment],
    ) -> list[tuple[str, ExtractedTextSegment, int]]:
        units: list[tuple[str, ExtractedTextSegment, int]] = []
        for segment in segments:
            for part_index, part in enumerate(_PARAGRAPH_BREAK.split(segment.text)):
                normalized = _WHITESPACE.sub(" ", part).strip()
                if normalized:
                    units.append((normalized, segment, part_index))
        return units

    @staticmethod
    def _join_units(
        units: list[tuple[str, ExtractedTextSegment, int]],
    ) -> tuple[str, list[_TextUnit]]:
        text_parts: list[str] = []
        positioned: list[_TextUnit] = []
        offset = 0
        for unit_index, (unit_text, segment, part_index) in enumerate(units):
            if unit_index:
                text_parts.append("\n\n")
                offset += 2
            start = offset
            end = start + len(unit_text)
            positioned.append(_TextUnit(unit_text, segment, part_index, start, end))
            text_parts.append(unit_text)
            offset = end
        return "".join(text_parts), positioned

    def _choose_end(self, text: str, start: int) -> int:
        length = len(text)
        limit = min(start + self.chunk_size, length)
        if limit == length:
            return length

        minimum_end = min(start + self.minimum_chunk_size, limit)
        paragraph_end = text.rfind("\n\n", minimum_end, limit + 1)
        if paragraph_end >= minimum_end:
            return paragraph_end

        word_end = text.rfind(" ", minimum_end, limit + 1)
        if word_end >= minimum_end:
            return word_end

        next_space = text.find(" ", limit)
        return length if next_space < 0 else next_space

    def _overlap_start(self, text: str, start: int, end: int) -> int:
        next_start = max(start + 1, end - self.chunk_overlap)
        if next_start < end and not text[next_start - 1].isspace() and not text[next_start].isspace():
            next_space = text.find(" ", next_start, end)
            if next_space >= 0:
                next_start = next_space + 1
            else:
                next_start = end
        while next_start < end and text[next_start].isspace():
            next_start += 1
        return next_start

    @staticmethod
    def _source_metadata(
        start: int,
        end: int,
        units: list[_TextUnit],
    ) -> dict[str, object]:
        sources: list[dict[str, object]] = []
        for unit in units:
            overlap_start = max(start, unit.start)
            overlap_end = min(end, unit.end)
            if overlap_start >= overlap_end:
                continue

            segment = unit.segment
            source: dict[str, object] = {
                "segment_sequence": segment.sequence,
                "segment_part_index": unit.part_index,
                "segment_start_char": overlap_start - unit.start,
                "segment_end_char": overlap_end - unit.start,
            }
            for name in (
                "page_number",
                "paragraph_index",
                "paragraph_style",
                "line_number",
                "table_index",
                "table_row_index",
                "table_cell_index",
            ):
                value = getattr(segment, name)
                if value is not None:
                    source[name] = value
            sources.append(source)

        metadata: dict[str, object] = {
            "document_start_char": start,
            "document_end_char": end,
            "sources": sources,
        }
        pages = sorted({source["page_number"] for source in sources if "page_number" in source})
        if pages:
            metadata["page_numbers"] = pages
        lines = [source["line_number"] for source in sources if "line_number" in source]
        if lines:
            metadata["line_start"] = min(lines)
            metadata["line_end"] = max(lines)
        return metadata