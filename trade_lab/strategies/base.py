from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from core.types import Bar, OrderIntent, TargetPosition


@dataclass
class StrategyConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    @classmethod
    def default_parameters(cls) -> dict[str, Any]:
        return {}

    def param(self, key: str, default: Any = None) -> Any:
        return self.config.params.get(key, default)

    @abstractmethod
    def initialize(self, state: dict[str, Any]) -> None:
        """Prepare strategy state from full historical data before the event loop starts."""

    @abstractmethod
    def on_bar(
        self,
        i: int,
        bar: Bar,
        state: dict[str, Any],
    ) -> TargetPosition | list[OrderIntent] | None:
        """Return desired position or order intents for bar index i."""
