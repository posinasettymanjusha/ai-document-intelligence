from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends
from google import genai
from google.genai import types

from app.api.v1.routes.search import get_semantic_search_service
from app.auth.dependencies import get_current_user
from app.auth.schemas import AuthenticatedUser
from app.core.config import settings
from app.integrations.gemini_generation import GeminiGenerationProvider
from app.rag.schemas import GroundedAnswerRequest, GroundedAnswerResponse
from app.rag.service import GroundedAnswerService
from app.search.service import SemanticSearchService

router = APIRouter()


def get_gemini_generation_provider() -> GeminiGenerationProvider:
    def create_client(timeout_seconds: float | None = None) -> genai.Client:
        api_key = settings.gemini_api_key
        if api_key is None or not api_key.get_secret_value().strip():
            raise RuntimeError("Gemini generation is not configured")
        timeout = min(
            float(settings.gemini_generation_timeout_seconds),
            timeout_seconds if timeout_seconds is not None else float("inf"),
        )
        return genai.Client(
            api_key=api_key.get_secret_value(),
            http_options=types.HttpOptions(
                timeout=max(1, int(timeout * 1000)),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    return GeminiGenerationProvider(
        None,
        settings.gemini_generation_model,
        client_factory=create_client,
    )


def get_grounded_answer_service(
    search_service: Annotated[
        SemanticSearchService,
        Depends(get_semantic_search_service),
    ],
    answer_generator: Annotated[
        GeminiGenerationProvider,
        Depends(get_gemini_generation_provider),
    ],
) -> GroundedAnswerService:
    return GroundedAnswerService(search_service, answer_generator)


@router.post(
    "/workspaces/{workspace_id}/answers",
    response_model=GroundedAnswerResponse,
    dependencies=[Depends(get_current_user)],
    summary="Answer a question using grounded workspace documents",
)
def answer_workspace_question(
    workspace_id: UUID,
    request: Annotated[GroundedAnswerRequest, Body()],
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    service: Annotated[GroundedAnswerService, Depends(get_grounded_answer_service)],
) -> GroundedAnswerResponse:
    return service.answer(workspace_id, user.id, request)