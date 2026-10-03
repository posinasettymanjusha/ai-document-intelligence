from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.auth.schemas import AuthenticatedUser
from app.db.session import get_db
from app.workspaces.schemas import WorkspaceSummary
from app.workspaces.service import WorkspaceService
from app.workspaces.sql_repository import SqlAlchemyWorkspaceRepository

router = APIRouter()


def get_workspace_service(
    session: Annotated[Session, Depends(get_db)],
) -> WorkspaceService:
    return WorkspaceService(SqlAlchemyWorkspaceRepository(session))


@router.get(
    "/workspaces",
    response_model=list[WorkspaceSummary],
    dependencies=[Depends(get_current_user)],
    summary="List workspaces where the authenticated user is a member",
)
def list_workspaces(
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    service: Annotated[WorkspaceService, Depends(get_workspace_service)],
) -> list[WorkspaceSummary]:
    return list(service.list_for_user(user.id))