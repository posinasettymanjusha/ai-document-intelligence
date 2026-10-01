from io import BytesIO
from typing import Protocol

import pymupdf
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.documents.models import ExtractedTextSegment


class DocumentExtractor(Protocol):
    def extract(self, content: bytes) -> list[ExtractedTextSegment]: ...


class PdfExtractor:
    def extract(self, content: bytes) -> list[ExtractedTextSegment]:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            return [
                ExtractedTextSegment(
                    sequence=page_number,
                    page_number=page_number,
                    text=page.get_text("text"),
                )
                for page_number, page in enumerate(document, start=1)
            ]


class DocxExtractor:
    def extract(self, content: bytes) -> list[ExtractedTextSegment]:
        document = Document(BytesIO(content))
        segments: list[ExtractedTextSegment] = []
        paragraph_index = 0
        table_index = 0

        for block in document.iter_inner_content():
            if isinstance(block, Paragraph):
                paragraph_index += 1
                segments.append(self._paragraph_segment(block, paragraph_index))
            elif isinstance(block, Table):
                table_index += 1
                for row_index, row in enumerate(block.rows, start=1):
                    for cell_index, cell in enumerate(row.cells, start=1):
                        for paragraph in cell.paragraphs:
                            paragraph_index += 1
                            segments.append(
                                self._paragraph_segment(
                                    paragraph,
                                    paragraph_index,
                                    table_index=table_index,
                                    table_row_index=row_index,
                                    table_cell_index=cell_index,
                                )
                            )

        return segments

    @staticmethod
    def _paragraph_segment(
        paragraph: Paragraph,
        paragraph_index: int,
        *,
        table_index: int | None = None,
        table_row_index: int | None = None,
        table_cell_index: int | None = None,
    ) -> ExtractedTextSegment:
        return ExtractedTextSegment(
            sequence=paragraph_index,
            paragraph_index=paragraph_index,
            paragraph_style=paragraph.style.name if paragraph.style else None,
            text=paragraph.text,
            table_index=table_index,
            table_row_index=table_row_index,
            table_cell_index=table_cell_index,
        )


class PlainTextExtractor:
    def extract(self, content: bytes) -> list[ExtractedTextSegment]:
        text = content.decode("utf-8-sig")
        return [
            ExtractedTextSegment(sequence=line_number, line_number=line_number, text=line)
            for line_number, line in enumerate(text.splitlines(), start=1)
        ]


EXTRACTORS: dict[str, DocumentExtractor] = {
    ".pdf": PdfExtractor(),
    ".docx": DocxExtractor(),
    ".txt": PlainTextExtractor(),
}

EXPECTED_CONTENT_TYPES: dict[str, frozenset[str]] = {
    ".pdf": frozenset({"application/pdf", "application/octet-stream"}),
    ".docx": frozenset(
        {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/zip",
            "application/octet-stream",
        }
    ),
    ".txt": frozenset({"text/plain", "application/octet-stream"}),
}

CANONICAL_CONTENT_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
}