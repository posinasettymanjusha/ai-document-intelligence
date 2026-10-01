from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.schemas import AuthenticatedUser
from app.auth.service import AuthService
from app.core.errors import AuthenticationError

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> AuthenticatedUser:
    if credentials is None:
        raise AuthenticationError()
    return AuthService().get_user_from_access_token(credentials.credentials)