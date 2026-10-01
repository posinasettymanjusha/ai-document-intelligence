from pydantic import BaseModel


class AuthenticatedUser(BaseModel):
    id: str
    email: str | None = None
    role: str = "authenticated"