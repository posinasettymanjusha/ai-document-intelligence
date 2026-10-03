from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import Workspace, WorkspaceMember
from app.workspaces.schemas import WorkspaceSummary


class SqlAlchemyWorkspaceRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def list_for_user(self, user_id: str) -> list[WorkspaceSummary]:
        user_uuid = UUID(user_id)
        query = (
            select(Workspace, WorkspaceMember.role)
            .join(WorkspaceMember, WorkspaceMember.workspace_id == Workspace.id)
            .where(WorkspaceMember.user_id == user_uuid)
            .order_by(Workspace.name, Workspace.id)
        )
        try:
            rows = self._session.execute(query).all()
        except SQLAlchemyError:
            self._session.rollback()
            raise
        return [
            WorkspaceSummary(id=workspace.id, name=workspace.name, role=role)
            for workspace, role in rows
        ]

    def rollback(self) -> None:
        self._session.rollback()