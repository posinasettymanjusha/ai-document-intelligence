from types import SimpleNamespace
from unittest.mock import MagicMock

from google.genai import types

from app.core.config import settings
from app.integrations.gemini_embeddings import GeminiEmbeddingProvider


def make_response(count: int) -> SimpleNamespace:
    return SimpleNamespace(
        embeddings=[
            SimpleNamespace(values=[float(index)] * settings.embedding_dimension)
            for index in range(count)
        ]
    )


def test_gemini_adapter_sends_distinct_contents_and_configured_dimension() -> None:
    client = MagicMock()
    client.models.embed_content.return_value = make_response(2)
    provider = GeminiEmbeddingProvider(client)

    result = provider.embed_texts(["first chunk", "second chunk"])

    request = client.models.embed_content.call_args.kwargs
    assert request["model"] == "gemini-embedding-2"
    assert request["config"].output_dimensionality == settings.embedding_dimension
    assert len(request["contents"]) == 2
    assert all(isinstance(content, types.Content) for content in request["contents"])
    assert [vector[0] for vector in result] == [0.0, 1.0]


def test_gemini_adapter_single_text_returns_one_vector() -> None:
    client = MagicMock()
    client.models.embed_content.return_value = make_response(1)
    provider = GeminiEmbeddingProvider(client)

    result = provider.embed_text("single chunk")

    assert len(result) == settings.embedding_dimension
    assert client.models.embed_content.call_args.kwargs["contents"][0].parts[0].text == "single chunk"


def test_gemini_adapter_empty_batch_skips_sdk_request() -> None:
    client = MagicMock()
    provider = GeminiEmbeddingProvider(client)

    assert provider.embed_texts([]) == []
    client.models.embed_content.assert_not_called()
