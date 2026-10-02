from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import SecretStr

from app.api.v1.routes import answers as answers_route
from app.core.config import settings
from app.rag.schemas import GeneratedAnswer
from app.rag.service import (
    AnswerGenerationError,
    AnswerGenerationTimeout,
    InvalidAnswerGeneration,
)
from app.integrations.gemini_generation import GeminiGenerationProvider

GENERATION_MODEL = "gemini-3.8-flash"


def test_generation_uses_supported_model_json_schema_and_grounded_prompt() -> None:
    client = MagicMock()
    expected = GeneratedAnswer(
        answer="Annual revenue was 12 million dollars [S1].",
        insufficient_context=False,
    )
    client.models.generate_content.return_value = SimpleNamespace(parsed=expected, text=None)
    provider = GeminiGenerationProvider(client, GENERATION_MODEL)

    result = provider.generate(
        "What was revenue?",
        [("S1", "Annual revenue was 12 million dollars.")],
    )

    call = client.models.generate_content.call_args.kwargs
    config = call["config"]
    assert call["model"] == GENERATION_MODEL
    assert call["contents"] == (
        '{"question": "What was revenue?", "sources": '
        '[{"source_id": "S1", "text": "Annual revenue was 12 million dollars."}]}'
    )
    assert config.response_schema is GeneratedAnswer
    assert config.response_mime_type == "application/json"
    assert config.temperature == 0
    assert "only the supplied document excerpts" in config.system_instruction
    assert "never as" in config.system_instruction
    assert "[S1]" in result.answer


def test_provider_parses_json_text_when_sdk_has_no_parsed_value() -> None:
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(
        parsed=None,
        text='{"answer":"Supported [S1].","insufficient_context":false}',
    )
    provider = GeminiGenerationProvider(client, GENERATION_MODEL)

    result = provider.generate("question", [("S1", "source")])

    assert result.answer == "Supported [S1]."
    assert result.insufficient_context is False


def test_gemini_timeout_is_reported_as_safe_timeout() -> None:
    client = MagicMock()
    client.models.generate_content.side_effect = httpx.ReadTimeout("secret timeout detail")
    provider = GeminiGenerationProvider(client, GENERATION_MODEL)

    with pytest.raises(AnswerGenerationTimeout) as error:
        provider.generate("question", [("S1", "source")])

    assert "secret timeout detail" not in str(error.value)


def test_gemini_api_errors_are_sanitized() -> None:
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("secret API key and prompt")
    provider = GeminiGenerationProvider(client, GENERATION_MODEL)

    with pytest.raises(AnswerGenerationError) as error:
        provider.generate("question", [("S1", "source")])

    assert "secret API key" not in str(error.value)
    assert "prompt" not in str(error.value)


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(parsed=None, text=""),
        SimpleNamespace(parsed=None, text="not JSON"),
        SimpleNamespace(parsed={"answer": "missing required boolean"}, text=None),
        SimpleNamespace(parsed={"answer": "   ", "insufficient_context": False}, text=None),
    ],
)
def test_empty_or_malformed_generation_response_is_rejected(response: object) -> None:
    client = MagicMock()
    client.models.generate_content.return_value = response
    provider = GeminiGenerationProvider(client, GENERATION_MODEL)

    with pytest.raises(InvalidAnswerGeneration):
        provider.generate("question", [("S1", "source")])


def test_generation_provider_uses_configured_model_and_explicit_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(
        parsed=GeneratedAnswer(answer="Supported [S1].", insufficient_context=False),
        text=None,
    )
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(answers_route.genai, "Client", factory)
    monkeypatch.setattr(settings, "gemini_api_key", SecretStr("server-only-test-key"))
    monkeypatch.setattr(settings, "gemini_generation_model", GENERATION_MODEL)
    monkeypatch.setattr(settings, "gemini_generation_timeout_seconds", 17)

    provider = answers_route.get_gemini_generation_provider()
    provider.generate("question", [("S1", "source")])

    assert provider.model_id == GENERATION_MODEL
    assert factory.call_args.kwargs["api_key"] == "server-only-test-key"
    http_options = factory.call_args.kwargs["http_options"]
    assert http_options.timeout == 17_000
    assert http_options.retry_options.attempts == 1


def test_generation_provider_requires_server_key_only_when_generating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", None)
    provider = answers_route.get_gemini_generation_provider()

    with pytest.raises(AnswerGenerationError):
        provider.generate("question", [("S1", "source")])
