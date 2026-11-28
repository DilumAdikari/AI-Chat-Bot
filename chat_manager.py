import json
import os
from datetime import datetime
from typing import Any


class ChatHistoryManager:
    def __init__(self, file_path: str = "chat_history.json") -> None:
        self.file_path = file_path

    def load_all(self) -> dict[str, dict[str, Any]]:
        if not os.path.exists(self.file_path):
            return {}
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def save_all(self, chats: dict[str, dict[str, Any]]) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(chats, f, indent=2, ensure_ascii=False)

    @staticmethod
    def generate_chat_id() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    @staticmethod
    def build_title(messages: list[dict[str, str]]) -> str:
        for msg in messages:
            if msg.get("role") == "user" and msg.get("content", "").strip():
                cleaned = " ".join(msg["content"].strip().split())
                return (cleaned[:36] + "...") if len(cleaned) > 36 else cleaned
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

        chats[chat_id] = {
            "title": self.build_title(messages),
            "created_at": created_at,
            "updated_at": now_str,
            "messages": messages,
        }
        self.save_all(chats)
        return chats

    def delete_chat(
        self,
        chats: dict[str, dict[str, Any]],
        chat_id: str,
    ) -> dict[str, dict[str, Any]]:
        chats.pop(chat_id, None)
        self.save_all(chats)
        return chats