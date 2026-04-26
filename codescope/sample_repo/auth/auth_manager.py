"""
auth/auth_manager.py
Handles user authentication: login, logout, token generation, and session management.
"""

import hashlib
import hmac
import os
import time
from dataclasses import dataclass
from typing import Optional
from db.database import Database


@dataclass
class User:
    user_id: int
    username: str
    email: str
    role: str  # "admin" | "viewer" | "editor"


class AuthManager:
    """
    Central authentication manager.
    Handles JWT-like token generation, password hashing, and session storage.
    """

    TOKEN_EXPIRY = 3600  # 1 hour

    def __init__(self, db: Database):
        self.db = db
        self.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")
        self._sessions: dict[str, dict] = {}

    def login(self, username: str, password: str) -> Optional[str]:
        """
        Authenticate a user and return a session token.
        Returns None if credentials are invalid.
        """
        user = self.db.get_user_by_username(username)
        if user is None:
            return None

        pw_hash = self._hash_password(password, user["salt"])
        if not hmac.compare_digest(pw_hash, user["password_hash"]):
            return None

        token = self._generate_token(user["user_id"])
        self._sessions[token] = {
            "user_id": user["user_id"],
            "username": username,
            "role": user["role"],
            "expires_at": time.time() + self.TOKEN_EXPIRY,
        }
        return token

    def logout(self, token: str) -> bool:
        """Invalidate a session token."""
        if token in self._sessions:
            del self._sessions[token]
            return True
        return False

    def validate_token(self, token: str) -> Optional[User]:
        """
        Validate a session token and return the associated User.
        Returns None if token is expired or invalid.
        """
        session = self._sessions.get(token)
        if session is None:
            return None
        if time.time() > session["expires_at"]:
            del self._sessions[token]
            return None
        return User(
            user_id=session["user_id"],
            username=session["username"],
            email="",
            role=session["role"],
        )

    def require_role(self, token: str, required_role: str) -> bool:
        """Check if the token holder has at least the required role."""
        user = self.validate_token(token)
        if user is None:
            return False
        role_hierarchy = {"viewer": 0, "editor": 1, "admin": 2}
        return role_hierarchy.get(user.role, -1) >= role_hierarchy.get(required_role, 99)

    def register(self, username: str, email: str, password: str) -> bool:
        """Register a new user. Returns True on success."""
        if self.db.get_user_by_username(username):
            return False
        salt = os.urandom(16).hex()
        pw_hash = self._hash_password(password, salt)
        self.db.create_user(username=username, email=email,
                            password_hash=pw_hash, salt=salt, role="viewer")
        return True

    def _hash_password(self, password: str, salt: str) -> str:
        return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()

    def _generate_token(self, user_id: int) -> str:
        raw = f"{user_id}:{time.time()}:{os.urandom(8).hex()}"
        return hashlib.sha256(raw.encode()).hexdigest()