"""
db/database.py
SQLite-backed database abstraction layer.
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional


class Database:
    """
    Thin wrapper around SQLite.
    Handles connection management, schema creation, and CRUD operations.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_schema(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                username  TEXT UNIQUE NOT NULL,
                email     TEXT,
                password_hash TEXT NOT NULL,
                salt      TEXT NOT NULL,
                role      TEXT DEFAULT 'viewer',
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS records (
                id        TEXT PRIMARY KEY,
                data      TEXT NOT NULL,
                created_at TEXT
            );
        """)
        conn.commit()

    def get_user_by_username(self, username: str) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None

    def get_all_users(self) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute("SELECT user_id, username, email, role FROM users").fetchall()
        return [dict(r) for r in rows]

    def create_user(self, username, email, password_hash, salt, role):
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO users (username, email, password_hash, salt, role) VALUES (?,?,?,?,?)",
            (username, email, password_hash, salt, role),
        )
        conn.commit()

    def insert_record(self, record: dict):
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO records (id, data, created_at) VALUES (?,?,?)",
            (record["id"], json.dumps(record), record.get("created_at", "")),
        )
        conn.commit()

    def fetch_records(self, limit: int = 100) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute("SELECT data FROM records LIMIT ?", (limit,)).fetchall()
        return [json.loads(r["data"]) for r in rows]