import re
from typing import Protocol
from uuid import UUID

from app.core.errors import AppError
from app.rag.schemas import (
    GeneratedAnswer,
    GroundedAnswerRequest,
    GroundedAnswerResponse,
    GroundedCitation,
)
from app.search.schemas import SemanticSearchRequest, SemanticSearchResult
from app.search.service import SemanticSearchService

_SOURCE_LABEL = re.compile(r"\[(S\d+)\]")
_INSUFFICIENT_ANSWER = "The provided documents do not contain enough information to answer this question."


class AnswerGenerationTimeout(Exception):
    pass


class AnswerGenerationError(Exception):
    pass


class InvalidAnswerGeneration(Exception):
    pass


class AnswerGenerator(Protocol):
    def generate(
        self,
        question: str,
        sources: list[tuple[str, str]],
    ) -> GeneratedAnswer: ...


class GroundedAnswerService:
    def __init__(
        self,
        search_service: SemanticSearchService,
        answer_generator: AnswerGenerator,
    ) -> None:
        self._search_service = search_service
        self._answer_generator = answer_generator

    def answer(
        self,
        workspace_id: UUID,
        user_id: str,
        request: GroundedAnswerRequest,
    ) -> GroundedAnswerResponse:
        search_results = self._search_service.search(
            workspace_id,
            user_id,
            SemanticSearchRequest(
                query=request.question,
                top_k=request.top_k,
                document_id=request.document_id,
                version_id=request.version_id,
            ),
        )
        if not search_results:
            return self._insufficient_response()

        source_results = {
            f"S{index}": result
            for index, result in enumerate(search_results, start=1)
        }
        labeled_sources = [
            (source_id, result.text)
            for source_id, result in source_results.items()
        ]

        try:
            generated = self._answer_generator.generate(request.question, labeled_sources)
        except AnswerGenerationTimeout as error:
            raise AppError(
                504,
                "answer_generation_timeout",
                "Answer generation timed out. Please try again.",
            ) from error
        except InvalidAnswerGeneration as error:
            raise AppError(
                502,
                "invalid_answer_generation_response",
                "The answer provider returned an invalid response.",
            ) from error
        except Exception as error:
            raise AppError(
                503,
                "answer_generation_unavailable",
                "An answer could not be generated right now.",
            ) from error

        if generated.insufficient_context:
            return self._insufficient_response()

        mentioned_labels = list(dict.fromkeys(_SOURCE_LABEL.findall(generated.answer)))
        allowed_labels = [label for label in mentioned_labels if label in source_results]
        answer = _SOURCE_LABEL.sub(
            lambda match: match.group(0) if match.group(1) in source_results else "",
            generated.answer,
        ).strip()
        if not allowed_labels or not answer:
            return self._insufficient_response()

        citations = [
            self._citation(label, source_results[label])
            for label in allowed_labels
        ]
        return GroundedAnswerResponse(
            answer=answer,
            insufficient_context=False,
            citations=citations,
        )

    @staticmethod
    def _citation(source_id: str, result: SemanticSearchResult) -> GroundedCitation:
        metadata = dict(result.source_metadata)
        raw_pages = metadata.get("page_numbers", [])
        page_numbers = (
            [page for page in raw_pages if isinstance(page, int) and not isinstance(page, bool)]
            if isinstance(raw_pages, list)
            else []
        )
        return GroundedCitation(
            source_id=source_id,
            document_id=result.document_id,
            filename=result.filename,
            version_id=result.version_id,
            version_number=result.version_number,
            chunk_id=result.chunk_id,
            chunk_index=result.chunk_index,
            page_numbers=page_numbers,
            source_metadata=metadata,
        )

    @staticmethod
    def _insufficient_response() -> GroundedAnswerResponse:
        return GroundedAnswerResponse(
            answer=_INSUFFICIENT_ANSWER,
            insufficient_context=True,
            citations=[],
        )