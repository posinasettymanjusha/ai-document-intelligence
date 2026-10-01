from typing import Protocol
from urllib.parse import quote

import httpx

from app.core.config import settings


class StorageError(Exception):
    """Raised when a private object storage operation fails."""


class ObjectStorage(Protocol):
    def upload(self, path: str, content: bytes, content_type: str) -> None: ...

    def delete(self, path: str) -> None: ...


class SupabaseStorage:
    def __init__(self) -> None:
        self._base_url = (settings.supabase_url or "").rstrip("/")
        self._bucket = "documents"
        self._service_key = settings.supabase_service_role_key or ""

    def upload(self, path: str, content: bytes, content_type: str) -> None:
        self._ensure_configured()
        try:
            response = httpx.post(
                self._object_url(path),
                content=content,
                headers=self._headers(content_type) | {"x-upsert": "false"},
                timeout=30.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise StorageError("Could not store the uploaded file privately.") from error

    def delete(self, path: str) -> None:
        self._ensure_configured()
        url = f"{self._base_url}/storage/v1/object/{quote(self._bucket, safe='')}"
        try:
            response = httpx.delete(
                url,
                json={"prefixes": [path]},
                headers=self._headers("application/json"),
                timeout=30.0,
            )
            if response.status_code != 404:
                response.raise_for_status()
        except httpx.HTTPError as error:
            raise StorageError("Could not remove the stored file.") from error

    def _object_url(self, path: str) -> str:
        return (
            f"{self._base_url}/storage/v1/object/"
            f"{quote(self._bucket, safe='')}/{quote(path, safe='/')}"
        )

    def _headers(self, content_type: str) -> dict[str, str]:
        return {
            "apikey": self._service_key,
            "Authorization": f"Bearer {self._service_key}",
            "Content-Type": content_type,
        }

    def _ensure_configured(self) -> None:
        if not self._base_url or not self._service_key:
            raise StorageError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be configured for file storage."
            )