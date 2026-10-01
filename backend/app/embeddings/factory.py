from google import genai

from app.core.config import settings
from app.embeddings.providers import DeterministicHashEmbeddingProvider, EmbeddingProvider
from app.embeddings.service import EmbeddingService


def create_embedding_provider() -> EmbeddingProvider:
    if settings.embedding_provider == "local":
        return DeterministicHashEmbeddingProvider(settings.embedding_dimension)

    api_key = settings.gemini_api_key
    if api_key is None or not api_key.get_secret_value().strip():
        raise RuntimeError("GEMINI_API_KEY must be configured for the Gemini embedding provider")

    from app.integrations.gemini_embeddings import GeminiEmbeddingProvider

    client = genai.Client(api_key=api_key.get_secret_value())
    return GeminiEmbeddingProvider(
        client=client,
        model_id=settings.embedding_model,
        dimension=settings.embedding_dimension,
    )


def create_embedding_service() -> EmbeddingService:
    return EmbeddingService(create_embedding_provider(), settings.embedding_dimension)