from collections.abc import Sequence
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError

from app.core.errors import AppError
from app.workspaces.repository import WorkspaceRepository
from app.workspaces.schemas import WorkspaceSummary


class WorkspaceService:
    def __init__(self, repository: WorkspaceRepository) -> None:
        self._repository = repository

    def list_for_user(self, user_id: str) -> Sequence[WorkspaceSummary]:
        try:
            normalized_user_id = str(UUID(user_id))
        except ValueError as error:
            raise AppError(401, "unauthorized", "A valid user identity is required.") from error

        try:
            return self._repository.list_for_user(normalized_user_id)
        except SQLAlchemyError as error:
            rollback = getattr(self._repository, "rollback", None)
            if rollback is not None:
                rollback()
            raise AppError(
                503,
                "persistence_unavailable",
                "Workspaces could not be loaded.",
            ) from error