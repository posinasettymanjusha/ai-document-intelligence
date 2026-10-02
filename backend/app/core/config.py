from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AI Document Intelligence API"
    environment: str = "development"
    log_level: str = "INFO"
    frontend_origin: str = "http://localhost:5173"
    api_v1_prefix: str = "/api/v1"
    max_document_size_bytes: int = Field(default=10_485_760, gt=0)
    document_chunk_size: int = Field(default=1200, gt=0)
    document_chunk_overlap: int = Field(default=150, ge=0)
    document_minimum_chunk_size: int = Field(default=100, gt=0)
    embedding_dimension: int = Field(default=768, ge=1, le=2000)
    embedding_provider: str = "local"
    embedding_model: str = "gemini-embedding-2"
    embedding_batch_size: int = Field(default=16, ge=1, le=100)
    gemini_generation_model: str = "gemini-3.8-flash"
    gemini_generation_timeout_seconds: int = Field(default=30, gt=0, le=300)
    conversation_max_history_turns: int = Field(default=8, ge=1, le=50)
    conversation_max_history_characters: int = Field(default=12_000, ge=1_000, le=100_000)
    conversation_max_question_length: int = Field(default=8_000, ge=1, le=32_000)
    conversation_stale_pending_seconds: int = Field(default=300, ge=60, le=86_400)
    document_analysis_max_chunks: int = Field(default=200, ge=1, le=2_000)
    document_analysis_max_estimated_input_tokens: int = Field(
        default=60_000,
        ge=1_000,
        le=1_000_000,
    )
    document_analysis_batch_estimated_tokens: int = Field(
        default=6_000,
        ge=256,
        le=100_000,
    )
    document_analysis_max_provider_calls: int = Field(default=12, ge=1, le=100)
    document_analysis_max_duration_seconds: int = Field(default=240, ge=10, le=3_600)
    document_analysis_max_output_items: int = Field(default=40, ge=1, le=200)
    document_analysis_max_output_tokens: int = Field(default=4_096, ge=256, le=32_768)
    gemini_api_key: SecretStr | None = None
    supabase_url: str | None = None
    supabase_service_role_key: str | None = None
    database_url: str | None = None

    @model_validator(mode="after")
    def validate_chunking_configuration(self) -> "Settings":
        if self.document_chunk_overlap >= self.document_chunk_size:
            raise ValueError("DOCUMENT_CHUNK_OVERLAP must be smaller than DOCUMENT_CHUNK_SIZE")
        if self.document_minimum_chunk_size > self.document_chunk_size:
            raise ValueError("DOCUMENT_MINIMUM_CHUNK_SIZE cannot exceed DOCUMENT_CHUNK_SIZE")
        if self.embedding_provider not in {"local", "gemini"}:
            raise ValueError("EMBEDDING_PROVIDER must be 'local' or 'gemini'")
        if self.conversation_max_history_characters < self.conversation_max_question_length:
            raise ValueError(
                "CONVERSATION_MAX_HISTORY_CHARACTERS must be at least "
                "CONVERSATION_MAX_QUESTION_LENGTH"
            )
        if (
            self.document_analysis_batch_estimated_tokens
            > self.document_analysis_max_estimated_input_tokens
        ):
            raise ValueError(
                "DOCUMENT_ANALYSIS_BATCH_ESTIMATED_TOKENS cannot exceed "
                "DOCUMENT_ANALYSIS_MAX_ESTIMATED_INPUT_TOKENS"
            )
        api_key = self.gemini_api_key.get_secret_value().strip() if self.gemini_api_key else ""
        if self.embedding_provider == "gemini" and not api_key:
            raise ValueError("GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini")
        return self


settings = Settings()