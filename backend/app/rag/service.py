from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from app.core.errors import AppError
from app.rag.citations import clean_answer_citations, make_grounded_citation
from app.rag.schemas import (
    GeneratedAnswer,
    GroundedAnswerRequest,
    GroundedAnswerResponse,
    GroundedCitation,
    ConversationContextMessage,
)
from app.search.schemas import SemanticSearchRequest, SemanticSearchResult
from app.search.service import SemanticSearchService

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
        conversation_history: Sequence[ConversationContextMessage] = (),
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
        return self.answer_with_context(
            workspace_id,
            user_id,
            request,
            retrieval_query=request.question,
        )

    def answer_with_context(
        self,
        workspace_id: UUID,
        user_id: str,
        request: GroundedAnswerRequest,
        *,
        conversation_history: Sequence[ConversationContextMessage] = (),
        retrieval_query: str | None = None,
    ) -> GroundedAnswerResponse:
        query = retrieval_query or request.question
        search_results = self._search_service.search(
            workspace_id,
            user_id,
            SemanticSearchRequest(
                query=query,
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
            if conversation_history:
                generated = self._answer_generator.generate(
                    request.question,
                    labeled_sources,
                    conversation_history,
                )
            else:
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

        answer, allowed_labels = clean_answer_citations(generated.answer, source_results)
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
        return make_grounded_citation(source_id, result)

    @staticmethod
    def _insufficient_response() -> GroundedAnswerResponse:
        return GroundedAnswerResponse(
            answer=_INSUFFICIENT_ANSWER,
            insufficient_context=True,
            citations=[],
        )