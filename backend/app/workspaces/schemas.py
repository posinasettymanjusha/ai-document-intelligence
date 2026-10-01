from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

WorkspaceRole = Literal["owner", "admin", "member"]


class WorkspaceSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str = Field(min_length=1, max_length=120)
    role: WorkspaceRole