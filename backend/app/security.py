"""Authentication (Clerk session tokens) and encryption of user API keys at rest."""
import base64
import os

import jwt
from cryptography.fernet import Fernet, InvalidToken
from fastapi import Header, HTTPException

from .config import settings

# ------------------------------ encryption ------------------------------

_fernet: Fernet | None = None


def _cipher() -> Fernet:
    global _fernet
    if _fernet is None:
        key = settings.secret_key
        if not key:
            if not settings.dev_auth:
                raise RuntimeError(
                    "ATRIUM_SECRET_KEY is not set. Generate one with:\n"
                    '  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"')
            path = os.path.join(settings.data_dir, "dev-secret.key")  # dev convenience only
            if not os.path.exists(path):
                os.makedirs(settings.data_dir, exist_ok=True)
                with open(path, "wb") as f:
                    f.write(Fernet.generate_key())
            with open(path, "rb") as f:
                key = f.read().decode()
        _fernet = Fernet(key.encode())
    return _fernet


def encrypt(plain: str) -> str:
    return _cipher().encrypt(plain.encode()).decode()


def decrypt(token: str) -> str:
    try:
        return _cipher().decrypt(token.encode()).decode()
    except InvalidToken:
        return ""  # encrypted under a different secret key: treat as "no key saved"


# ------------------------------ authentication ------------------------------

_jwks: jwt.PyJWKClient | None = None


def clerk_issuer() -> str:
    """A Clerk publishable key is pk_(test|live)_<base64 of the instance's domain + '$'>."""
    encoded = settings.clerk_publishable_key.split("_", 2)[2]
    domain = base64.b64decode(encoded + "=" * (-len(encoded) % 4)).decode().rstrip("$")
    return f"https://{domain}"


def verify_token(token: str) -> str:
    """Verify a Clerk session JWT against the instance's public keys. Returns the Clerk user id."""
    global _jwks
    issuer = clerk_issuer()
    if _jwks is None:
        _jwks = jwt.PyJWKClient(f"{issuer}/.well-known/jwks.json", cache_keys=True, lifespan=3600)
    key = _jwks.get_signing_key_from_jwt(token).key
    claims = jwt.decode(token, key, algorithms=["RS256"], issuer=issuer, leeway=10,
                        options={"require": ["exp", "sub"], "verify_aud": False})
    return claims["sub"]


def authenticate(token: str | None, dev_user: str | None = None) -> str:
    """Resolve the caller to a user id, or raise 401/503."""
    if settings.dev_auth:
        return dev_user or "dev"
    if not settings.clerk_publishable_key:
        raise HTTPException(503, "Auth is not configured: set ATRIUM_CLERK_PUBLISHABLE_KEY (or ATRIUM_DEV_AUTH=1 locally).")
    if not token:
        raise HTTPException(401, "Sign in required")
    try:
        return verify_token(token)
    except jwt.PyJWTError as e:
        raise HTTPException(401, f"Invalid session: {e}")


def current_user(authorization: str | None = Header(default=None), x_user: str | None = Header(default=None)) -> str:
    token = authorization[7:] if authorization and authorization.lower().startswith("bearer ") else None
    return authenticate(token, dev_user=x_user)  # X-User is honoured only in dev_auth mode
