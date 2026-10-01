from typing import Protocol, Sequence

from app.workspaces.schemas import WorkspaceSummary


class WorkspaceRepository(Protocol):
    def list_for_user(self, user_id: str) -> Sequence[WorkspaceSummary]: ...