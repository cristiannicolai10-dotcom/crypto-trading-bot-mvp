from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from .settings import MAX_TELEGRAM_FILE_MB, TELEGRAM_CHAT_ID, TELEGRAM_TOKEN


class TelegramAPI:
    def __init__(self) -> None:
        self.base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
        self.default_chat_id = TELEGRAM_CHAT_ID

    def _request(self, method: str, *, data: dict | None = None,
                 files: dict | None = None, timeout: int = 40) -> Any:
        url = f"{self.base_url}/{method}"
        response = requests.post(
            url,
            data=data,
            files=files,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(f"Telegram API error: {payload}")
        return payload.get("result")

    @staticmethod
    def _chunks(text: str, limit: int = 3900):
        text = str(text)
        while len(text) > limit:
            split_at = text.rfind("\n", 0, limit)
            if split_at < limit // 2:
                split_at = limit
            yield text[:split_at]
            text = text[split_at:].lstrip("\n")
        if text:
            yield text

    def send_message(
        self,
        text: str,
        chat_id: str | None = None,
        reply_markup: str | None = None,
    ) -> None:
        target = str(chat_id or self.default_chat_id)
        chunks = list(self._chunks(text))
        for idx, chunk in enumerate(chunks):
            data = {
                "chat_id": target,
                "text": chunk,
                "disable_web_page_preview": "true",
            }
            if reply_markup and idx == len(chunks) - 1:
                data["reply_markup"] = reply_markup
            self._request("sendMessage", data=data)

    def answer_callback_query(
        self,
        callback_query_id: str,
        text: str = "",
    ) -> None:
        data = {"callback_query_id": callback_query_id}
        if text:
            data["text"] = text[:180]
        self._request("answerCallbackQuery", data=data)

    def send_document(
        self,
        path: str | Path,
        caption: str = "",
        chat_id: str | None = None,
    ) -> bool:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return False

        max_bytes = int(MAX_TELEGRAM_FILE_MB * 1024 * 1024)
        if p.stat().st_size > max_bytes:
            return False

        target = str(chat_id or self.default_chat_id)

        with p.open("rb") as handle:
            self._request(
                "sendDocument",
                data={
                    "chat_id": target,
                    "caption": caption[:1000],
                },
                files={
                    "document": (p.name, handle),
                },
                timeout=180,
            )

        return True

    def get_updates(self, offset: int | None, timeout: int) -> list[dict]:
        data = {
            "timeout": timeout,
            "allowed_updates": '["message","callback_query"]',
        }
        if offset is not None:
            data["offset"] = offset

        result = self._request(
            "getUpdates",
            data=data,
            timeout=timeout + 10,
        )
        return result or []
