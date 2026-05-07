import os
import sqlite3
import threading
from datetime import datetime

class ChatStore:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or os.environ.get("CHAT_DB_PATH", "chat_history.db")
        self.lock = threading.Lock()
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._enable_foreign_keys()
        self._ensure_tables()

    def _enable_foreign_keys(self):
        self._execute("PRAGMA foreign_keys = ON")

    def _execute(self, query: str, params: tuple = (), commit: bool = False):
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            if commit:
                self.connection.commit()
            return cursor

    def _ensure_tables(self):
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                chat_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            )
            """,
            commit=True,
        )
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                seq INTEGER NOT NULL,
                FOREIGN KEY(chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE
            )
            """,
            commit=True,
        )

    def create_session(self, chat_id: str):
        self._execute(
            "INSERT OR IGNORE INTO chats (chat_id, created_at) VALUES (?, ?)",
            (chat_id, datetime.utcnow().isoformat()),
            commit=True,
        )

    def chat_exists(self, chat_id: str) -> bool:
        cursor = self._execute("SELECT 1 FROM chats WHERE chat_id = ?", (chat_id,))
        return cursor.fetchone() is not None

    def get_messages(self, chat_id: str) -> list[dict]:
        cursor = self._execute(
            "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY seq ASC", (chat_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def add_messages(self, chat_id: str, messages: list[dict]):
        existing_count = self._execute(
            "SELECT COUNT(*) AS count FROM messages WHERE chat_id = ?", (chat_id,)
        ).fetchone()[0]

        with self.lock:
            cursor = self.connection.cursor()
            for index, message in enumerate(messages, start=1):
                role = message.get("role", "user")
                content = message.get("content", "")
                cursor.execute(
                    "INSERT INTO messages (chat_id, role, content, created_at, seq) VALUES (?, ?, ?, ?, ?)",
                    (chat_id, role, content, datetime.utcnow().isoformat(), existing_count + index),
                )
            self.connection.commit()

    def delete_chat(self, chat_id: str):
        self._execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,), commit=True)
        self._execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,), commit=True)

    def list_chats(self) -> list[dict]:
        cursor = self._execute("SELECT chat_id, created_at FROM chats ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]
