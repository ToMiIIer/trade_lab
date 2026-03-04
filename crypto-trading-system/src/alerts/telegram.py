"""Telegram alert client for run summaries and failures."""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


@dataclass(slots=True)
class TelegramConfig:
    enabled: bool
    bot_token: str
    chat_id: str
    timeout_seconds: int = 8


class TelegramAlerter:
    def __init__(self, config: TelegramConfig) -> None:
        self.config = config

    @classmethod
    def from_env(cls) -> "TelegramAlerter":
        enabled = os.getenv("TELEGRAM_ENABLED", "false").strip().lower() in {"1", "true", "yes"}
        return cls(
            TelegramConfig(
                enabled=enabled,
                bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
                chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            )
        )

    def send(self, message: str) -> tuple[bool, str | None]:
        if not self.config.enabled:
            return False, "telegram_disabled"

        if not self.config.bot_token or not self.config.chat_id:
            return False, "telegram_missing_credentials"

        url = f"https://api.telegram.org/bot{self.config.bot_token}/sendMessage"
        payload = {"chat_id": self.config.chat_id, "text": message}

        try:
            with httpx.Client(timeout=self.config.timeout_seconds) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
        except Exception as exc:  # pragma: no cover - network integration
            return False, f"telegram_send_failed:{exc}"

        return True, None
