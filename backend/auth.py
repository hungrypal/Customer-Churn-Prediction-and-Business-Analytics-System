"""Authentication primitives for database-backed HTTP-only sessions."""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from pwdlib import PasswordHash

from backend.config import AUTH_SESSION_HOURS

password_hash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(password: str, stored_hash: str) -> bool:
    return password_hash.verify(password, stored_hash)


def create_session_token() -> str:
    return secrets.token_urlsafe(48)


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def session_expiry() -> datetime:
    return datetime.utcnow() + timedelta(hours=AUTH_SESSION_HOURS)


def public_user(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {"id": int(row["id"]), "name": row["name"], "email": row["email"]}
