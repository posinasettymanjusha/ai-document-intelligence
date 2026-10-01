from collections.abc import Sequence

from app.workspaces.repository import WorkspaceRepository
from app.workspaces.schemas import WorkspaceSummary


class WorkspaceService:
    def __init__(self, repository: WorkspaceRepository) -> None:
        self._repository = repository

    def list_for_user(self, user_id: str) -> Sequence[WorkspaceSummary]:
        return self._repository.list_for_user(user_id)