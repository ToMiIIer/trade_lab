from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from strategies.base import BaseStrategy, StrategyConfig


def discover_strategy_names() -> list[str]:
    strategy_dir = Path(__file__).resolve().parent
    names = []
    for file in strategy_dir.glob("*.py"):
        if file.stem in {"__init__", "base"}:
            continue
        names.append(file.stem)
    return sorted(names)


def get_strategy_class(name: str) -> type[BaseStrategy]:
    if not name:
        raise ValueError("Strategy name cannot be empty")

    try:
        module = importlib.import_module(f"strategies.{name}")
    except ModuleNotFoundError as exc:
        raise ValueError(f"Strategy module not found: {name}") from exc

    strategy_cls = getattr(module, "Strategy", None)
    if strategy_cls is None:
        raise ValueError(f"strategies.{name} must expose a `Strategy` class")
    if not issubclass(strategy_cls, BaseStrategy):
        raise TypeError(f"strategies.{name}.Strategy must extend BaseStrategy")

    return strategy_cls


def get_default_parameters(name: str) -> dict[str, Any]:
    strategy_cls = get_strategy_class(name)
    return strategy_cls.default_parameters()


def load_strategy(name: str, params: dict[str, Any] | None = None) -> BaseStrategy:
    strategy_cls = get_strategy_class(name)
    config = StrategyConfig(name=name, params=params or {})
    return strategy_cls(config)


__all__ = [
    "BaseStrategy",
    "StrategyConfig",
    "discover_strategy_names",
    "get_default_parameters",
    "get_strategy_class",
    "load_strategy",
]
