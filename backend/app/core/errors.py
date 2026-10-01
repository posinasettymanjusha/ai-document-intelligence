import logging
from collections.abc import Mapping

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        context: Mapping[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.context = dict(context or {})
        super().__init__(message)


class AuthenticationError(AppError):
    def __init__(self) -> None:
        super().__init__(401, "unauthorized", "A valid access token is required.")


def register_exception_handlers(application: FastAPI) -> None:
    @application.exception_handler(AppError)
    async def handle_app_error(_: Request, error: AppError) -> JSONResponse:
        content = {"error": {"code": error.code, "message": error.message}, **error.context}
        return JSONResponse(
            status_code=error.status_code,
            content=content,
            headers={"WWW-Authenticate": "Bearer"} if error.status_code == 401 else None,
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, error: Exception) -> JSONResponse:
        logger.exception("Unhandled request error", extra={"path": request.url.path}, exc_info=error)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_server_error",
                    "message": "An unexpected error occurred.",
                }
            },
        )