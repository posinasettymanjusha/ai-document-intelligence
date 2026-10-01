from functools import lru_cache
from typing import Any

import jwt
from jwt import PyJWKClient

from app.auth.schemas import AuthenticatedUser
from app.core.config import settings
from app.core.errors import AppError, AuthenticationError


@lru_cache(maxsize=1)
def _get_jwks_client(url: str) -> PyJWKClient:
    return PyJWKClient(f"{url.rstrip('/')}/auth/v1/.well-known/jwks.json")


class AuthService:
    def get_user_from_access_token(self, access_token: str) -> AuthenticatedUser:
        if not settings.supabase_url:
            raise AppError(
                503,
                "auth_not_configured",
                "Set SUPABASE_URL to enable access-token authentication.",
            )

        issuer = f"{settings.supabase_url.rstrip('/')}/auth/v1"
        try:
            signing_key = _get_jwks_client(settings.supabase_url).get_signing_key_from_jwt(access_token)
            claims: dict[str, Any] = jwt.decode(
                access_token,
                signing_key.key,
                algorithms=["ES256", "RS256"],
                audience="authenticated",
                issuer=issuer,
            )
        except jwt.PyJWTError as error:
            raise AuthenticationError() from error

        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject:
            raise AuthenticationError()

        email = claims.get("email")
        role = claims.get("role", "authenticated")
        return AuthenticatedUser(
            id=subject,
            email=email if isinstance(email, str) else None,
            role=role if isinstance(role, str) else "authenticated",
        )