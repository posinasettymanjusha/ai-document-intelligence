from app.chunking.service import DocumentChunkingService
from app.documents.models import ExtractedTextSegment


def chunker(
    chunk_size: int = 120,
    overlap: int = 20,
    minimum: int = 30,
) -> DocumentChunkingService:
    return DocumentChunkingService(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        minimum_chunk_size=minimum,
    )


def test_short_document_produces_one_chunk() -> None:
    chunks = chunker().chunk([ExtractedTextSegment(sequence=1, text="A short paragraph.")])

    assert len(chunks) == 1
    assert chunks[0].text == "A short paragraph."
    assert chunks[0].chunk_index == 0
    assert chunks[0].character_count == 18


def test_long_document_is_split_at_word_boundaries() -> None:
    text = " ".join(f"word{index}" for index in range(40))
    chunks = chunker(chunk_size=60, overlap=10, minimum=20).chunk(
        [ExtractedTextSegment(sequence=1, text=text)]
    )

    assert len(chunks) > 1
    assert all(not chunk.text.startswith(" ") and not chunk.text.endswith(" ") for chunk in chunks)
    assert all(chunk.text.split()[0] != "" for chunk in chunks)


def test_chunking_prefers_paragraph_boundaries() -> None:
    segments = [
        ExtractedTextSegment(sequence=index, paragraph_index=index, text=f"Paragraph {index}." * 3)
        for index in range(1, 4)
    ]

    chunks = chunker(chunk_size=80, overlap=0, minimum=20).chunk(segments)

    assert len(chunks) == 2
    assert chunks[0].text.startswith("Paragraph 1.")
    first_sources = chunks[0].source_metadata["sources"]
    assert [source["paragraph_index"] for source in first_sources] == [1, 2]
    assert all(source["segment_start_char"] == 0 for source in first_sources)
    assert all(source["segment_end_char"] == 36 for source in first_sources)
    assert chunks[1].source_metadata["sources"][0]["paragraph_index"] == 3


def test_adjacent_chunks_have_word_safe_overlap() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    chunks = chunker(chunk_size=35, overlap=12, minimum=15).chunk(
        [ExtractedTextSegment(sequence=1, text=text)]
    )

    assert len(chunks) > 1
    assert any(
        left_word == right_word
        for left in chunks
        for right in chunks
        for left_word in left.text.split()
        for right_word in right.text.split()
        if left_word in right.text.split()
    )


def test_chunking_is_deterministic() -> None:
    segments = [
        ExtractedTextSegment(sequence=1, page_number=1, text="Repeat this text. " * 20),
        ExtractedTextSegment(sequence=2, page_number=2, text="Next page text. " * 10),
    ]

    first = chunker().chunk(segments)
    second = chunker().chunk(segments)

    assert first == second


def test_chunk_size_is_configurable() -> None:
    text = "one two three four five six seven eight nine ten " * 4
    smaller = chunker(chunk_size=40, overlap=0, minimum=20).chunk(
        [ExtractedTextSegment(sequence=1, text=text)]
    )
    larger = chunker(chunk_size=80, overlap=0, minimum=20).chunk(
        [ExtractedTextSegment(sequence=1, text=text)]
    )

    assert len(smaller) > len(larger)


def test_overlap_is_configurable() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu " * 3
    no_overlap = chunker(chunk_size=45, overlap=0, minimum=20).chunk(
        [ExtractedTextSegment(sequence=1, text=text)]
    )
    with_overlap = chunker(chunk_size=45, overlap=15, minimum=20).chunk(
        [ExtractedTextSegment(sequence=1, text=text)]
    )

    assert sum(chunk.character_count for chunk in with_overlap) > sum(
        chunk.character_count for chunk in no_overlap
    )


def test_empty_input_and_whitespace_only_segments_produce_no_chunks() -> None:
    assert chunker().chunk([]) == []
    assert chunker().chunk([ExtractedTextSegment(sequence=1, text=" \n \t ")]) == []


def test_pdf_page_metadata_is_preserved_across_chunks() -> None:
    chunks = chunker(chunk_size=40, overlap=0, minimum=20).chunk(
        [
            ExtractedTextSegment(sequence=1, page_number=1, text="Page one content." * 2),
            ExtractedTextSegment(sequence=2, page_number=2, text="Page two content." * 2),
        ]
    )

    assert [chunk.source_metadata["page_numbers"] for chunk in chunks] == [[1], [2]]
    assert chunks[0].source_metadata["sources"][0]["segment_sequence"] == 1


def test_docx_paragraph_style_and_table_location_are_preserved() -> None:
    chunks = chunker().chunk(
        [
            ExtractedTextSegment(
                sequence=1,
                paragraph_index=1,
                paragraph_style="Heading 1",
                text="Project overview",
            ),
            ExtractedTextSegment(
                sequence=2,
                paragraph_index=2,
                paragraph_style="Normal",
                table_index=1,
                table_row_index=2,
                table_cell_index=1,
                text="Budget details",
            ),
        ]
    )

    sources = [source for chunk in chunks for source in chunk.source_metadata["sources"]]
    assert sources[0]["paragraph_style"] == "Heading 1"
    assert sources[1]["table_index"] == 1
    assert sources[1]["table_row_index"] == 2
    assert sources[1]["table_cell_index"] == 1


def test_txt_metadata_includes_line_range() -> None:
    chunks = chunker(chunk_size=36, overlap=0, minimum=20).chunk(
        [
            ExtractedTextSegment(sequence=line, line_number=line, text=f"Line {line} content")
            for line in range(1, 5)
        ]
    )

    assert chunks[0].source_metadata["line_start"] == 1
    assert chunks[0].source_metadata["line_end"] == 2


def test_oversized_single_segment_splits_without_splitting_words() -> None:
    text = "extraordinary" + " " + "ordinary " * 20
    chunks = chunker(chunk_size=30, overlap=5, minimum=15).chunk(
        [ExtractedTextSegment(sequence=1, text=text)]
    )

    assert len(chunks) > 1
    assert all(word in text.split() for chunk in chunks for word in chunk.text.split())


def test_character_and_token_counts_are_deterministic() -> None:
    chunks = chunker().chunk([ExtractedTextSegment(sequence=1, text="123456789")])

    assert chunks[0].character_count == 9
    assert chunks[0].estimated_token_count == 3