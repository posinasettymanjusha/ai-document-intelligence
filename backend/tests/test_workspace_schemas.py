from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.workspaces.schemas import WorkspaceSummary


def test_workspace_summary_accepts_valid_role() -> None:
    workspace = WorkspaceSummary(id=uuid4(), name="Research", role="owner")

    assert workspace.name == "Research"
    assert workspace.role == "owner"


def test_workspace_summary_rejects_unknown_role() -> None:
    with pytest.raises(ValidationError):
        WorkspaceSummary(id=uuid4(), name="Research", role="superuser")