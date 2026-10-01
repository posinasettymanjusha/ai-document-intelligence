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
        api_key = self.gemini_api_key.get_secret_value().strip() if self.gemini_api_key else ""
        if self.embedding_provider == "gemini" and not api_key:
            raise ValueError("GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini")
        return self


settings = Settings()