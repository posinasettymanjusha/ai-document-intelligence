from fastapi import APIRouter

from app.api.v1.routes.answers import router as answers_router
from app.api.v1.routes.documents import router as documents_router
from app.api.v1.routes.health import router as health_router
from app.api.v1.routes.search import router as search_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(documents_router, tags=["documents"])
api_router.include_router(search_router, tags=["search"])
api_router.include_router(answers_router, tags=["answers"])