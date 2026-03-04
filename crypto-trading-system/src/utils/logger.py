"""Application logging configuration."""

from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def setup_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format=_FORMAT)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
