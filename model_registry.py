import os
import sqlite3
import threading
from datetime import datetime

class ModelRegistry:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or os.environ.get("MODEL_DB_PATH", "model_registry.db")
        self.lock = threading.Lock()
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._enable_foreign_keys()
        self._ensure_tables()

    def _execute(self, query: str, params: tuple = (), commit: bool = False):
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            if commit:
                self.connection.commit()
            return cursor

    def _enable_foreign_keys(self):
        self._execute("PRAGMA foreign_keys = ON")

    def _ensure_tables(self):
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                path TEXT NOT NULL,
                model_type TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL
            )
            """,
            commit=True,
        )
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """,
            commit=True,
        )

    def register_model(
        self,
        model_id: str,
        source: str = "huggingface",
        path: str | None = None,
        model_type: str = "huggingface",
        description: str = "",
    ):
        path = path or model_id
        self._execute(
            "INSERT OR REPLACE INTO models (model_id, source, path, model_type, description, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (model_id, source, path, model_type, description, datetime.utcnow().isoformat()),
            commit=True,
        )

    def get_model(self, model_id: str) -> dict | None:
        cursor = self._execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_models(self) -> list[dict]:
        cursor = self._execute("SELECT * FROM models ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]

    def delete_model(self, model_id: str):
        self._execute("DELETE FROM models WHERE model_id = ?", (model_id,), commit=True)

    def set_default_model(self, model_id: str):
        self._execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES ('default_model', ?)",
            (model_id,),
            commit=True,
        )

    def get_default_model(self) -> str | None:
        cursor = self._execute("SELECT value FROM settings WHERE key = 'default_model'")
        row = cursor.fetchone()
        return row[0] if row else None

    def model_exists(self, model_id: str) -> bool:
        return self.get_model(model_id) is not None
