from fastapi import APIRouter

from app.api.v1.schemas import HealthResponse
from app.core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        environment=settings.environment,
    )