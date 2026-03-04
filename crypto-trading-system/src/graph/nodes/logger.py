"""Logging helper node for pipeline milestones."""

from __future__ import annotations

import logging
from typing import Any


class PipelineLoggerNode:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def run(self, run_id: str, stage: str, message: str, extra: dict[str, Any] | None = None) -> None:
        payload = {"run_id": run_id, "stage": stage, **(extra or {})}
        self.logger.info(message, extra=payload)
