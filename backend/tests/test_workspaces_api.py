from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from app.api.v1.routes.workspaces import get_workspace_service
from app.auth.dependencies import get_current_user
from app.auth.schemas import AuthenticatedUser
from app.main import app
from app.workspaces.schemas import WorkspaceSummary
from app.workspaces.service import WorkspaceService
from app.workspaces.sql_repository import SqlAlchemyWorkspaceRepository

USER_ID = UUID("6aa5a694-34c4-44f9-a631-4d9e11f27d1e")
OTHER_USER_ID = UUID("aca07028-48ec-4177-9a5b-6710e11df2ce")
FIRST_WORKSPACE_ID = UUID("17d67f65-81f3-49db-a199-6357cd8f46f9")
SECOND_WORKSPACE_ID = UUID("cd5b0042-98da-448d-b9ae-55d4558d0b8a")


class FakeWorkspaceRepository:
    def __init__(self, memberships: dict[str, list[WorkspaceSummary]]) -> None:
        self.memberships = memberships
        self.requested_user_ids: list[str] = []

    def list_for_user(self, user_id: str) -> list[WorkspaceSummary]:
        self.requested_user_ids.append(user_id)
        return self.memberships.get(user_id, [])


def override_endpoint(service: WorkspaceService, user_id: str | None) -> None:
    if user_id is None:
        app.dependency_overrides.pop(get_current_user, None)
    else:
        app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(id=user_id)
    app.dependency_overrides[get_workspace_service] = lambda: service


def clear_overrides() -> None:
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(get_workspace_service, None)


def test_authenticated_user_receives_multiple_workspace_summaries() -> None:
    first = WorkspaceSummary(id=FIRST_WORKSPACE_ID, name="Research", role="owner")
    second = WorkspaceSummary(id=SECOND_WORKSPACE_ID, name="Seminar", role="member")
    repository = FakeWorkspaceRepository({str(USER_ID): [first, second]})
    service = WorkspaceService(repository)  # type: ignore[arg-type]
    override_endpoint(service, str(USER_ID))
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/workspaces")
    finally:
        clear_overrides()

    assert response.status_code == 200
    assert response.json() == [
        {"id": str(FIRST_WORKSPACE_ID), "name": "Research", "role": "owner"},
        {"id": str(SECOND_WORKSPACE_ID), "name": "Seminar", "role": "member"},
    ]
    assert repository.requested_user_ids == [str(USER_ID)]


def test_user_only_receives_memberships_for_the_authenticated_identity() -> None:
    user_workspace = WorkspaceSummary(
        id=FIRST_WORKSPACE_ID,
        name="User workspace",
        role="admin",
    )
    other_workspace = WorkspaceSummary(
        id=SECOND_WORKSPACE_ID,
        name="Other user's workspace",
        role="owner",
    )
    repository = FakeWorkspaceRepository(
        {
            str(USER_ID): [user_workspace],
            str(OTHER_USER_ID): [other_workspace],
        }
    )
    service = WorkspaceService(repository)  # type: ignore[arg-type]
    override_endpoint(service, str(USER_ID))
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/workspaces")
    finally:
        clear_overrides()

    assert response.status_code == 200
    assert [item["id"] for item in response.json()] == [str(FIRST_WORKSPACE_ID)]
    assert str(OTHER_USER_ID) not in repository.requested_user_ids


def test_user_with_no_workspaces_receives_empty_list() -> None:
    service = WorkspaceService(FakeWorkspaceRepository({}))  # type: ignore[arg-type]
    override_endpoint(service, str(USER_ID))
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/workspaces")
    finally:
        clear_overrides()

    assert response.status_code == 200
    assert response.json() == []


def test_workspace_list_requires_authentication() -> None:
    service = WorkspaceService(FakeWorkspaceRepository({}))  # type: ignore[arg-type]
    override_endpoint(service, None)
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/workspaces")
    finally:
        clear_overrides()

    assert response.status_code == 401


def test_sql_repository_filters_by_workspace_membership_user_id() -> None:
    session = MagicMock(spec=Session)
    first_workspace = SimpleNamespace(id=FIRST_WORKSPACE_ID, name="Research")
    second_workspace = SimpleNamespace(id=SECOND_WORKSPACE_ID, name="Seminar")
    session.execute.return_value.all.return_value = [
        (first_workspace, "owner"),
        (second_workspace, "member"),
    ]
    repository = SqlAlchemyWorkspaceRepository(session)

    result = repository.list_for_user(str(USER_ID))

    statement = session.execute.call_args.args[0]
    compiled = statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)
    assert "workspace_members.user_id =" in sql
    assert "workspace_members.workspace_id = workspaces.id" in sql
    assert USER_ID in compiled.params.values()
    assert [item.id for item in result] == [FIRST_WORKSPACE_ID, SECOND_WORKSPACE_ID]
    assert [item.role for item in result] == ["owner", "member"]
    session.commit.assert_not_called()


def test_invalid_authenticated_user_id_is_rejected() -> None:
    service = WorkspaceService(FakeWorkspaceRepository({}))  # type: ignore[arg-type]

    try:
        service.list_for_user("not-a-uuid")
    except Exception as error:
        assert getattr(error, "status_code", None) == 401
        assert getattr(error, "code", None) == "unauthorized"
    else:
        raise AssertionError("invalid user identity was not rejected")