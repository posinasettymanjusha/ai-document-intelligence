from collections.abc import Mapping

from app.core.errors import AppError
from app.documents.models import DocumentProcessingStatus


class DocumentProcessingError(AppError):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        context: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(
            status_code,
            code,
            message,
            context={"status": DocumentProcessingStatus.FAILED.value, **(context or {})},
        )