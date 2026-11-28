import json
from datetime import datetime
from typing import Any

from db import DEFAULT_DB_PATH, db_session, init_db


class ChatHistoryManager:
    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        init_db(self.db_path)

    def load_all(self) -> dict[str, dict[str, Any]]:
        chats: dict[str, dict[str, Any]] = {}
        with db_session(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT chat_id, title, created_at, updated_at, messages_json
                FROM chats
                ORDER BY chat_id DESC
                """
            ).fetchall()

        for row in rows:
            try:
                messages = json.loads(row["messages_json"])
                if not isinstance(messages, list):
                    messages = []
            except Exception:
                messages = []

            chats[row["chat_id"]] = {
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "messages": messages,
            }
        return chats

    @staticmethod
    def generate_chat_id() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    @staticmethod
    def build_title(messages: list[dict[str, str]]) -> str:
        for msg in messages:
            if msg.get("role") == "user" and msg.get("content", "").strip():
                cleaned = " ".join(msg["content"].strip().split())
                return (cleaned[:36] + "...") if len(clean := cleaned) > 36 else clean
        return "New Chat"

    def upsert_chat(
        self,
        chats: dict[str, dict[str, Any]],
        chat_id: str,
        messages: list[dict[str, str]],
    ) -> dict[str, dict[str, Any]]:
        if not messages:
            return chats

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        created_at = chats.get(chat_id, {}).get("created_at", now_str)
        title = self.build_title(messages)
        messages_json = json.dumps(messages, ensure_ascii=False)

        with db_session(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO chats (chat_id, title, created_at, updated_at, messages_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    title = excluded.title,
                    updated_at = excluded.updated_at,
                    messages_json = excluded.messages_json
                """,
                (chat_id, title, created_at, now_str, messages_json),
            )

        chats[chat_id] = {
            "title": title,
            "created_at": created_at,
            "updated_at": now_str,
            "messages": messages,
        }
        return chats

    def delete_chat(
        self,
        chats: dict[str, dict[str, Any]],
        chat_id: str,
    ) -> dict[str, dict[str, Any]]:
        with db_session(self.db_path) as conn:
            conn.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))

        chats.pop(chat_id, None)
        return chats