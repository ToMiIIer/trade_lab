from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from core.types import Bar, Position


KillSwitchHook = Callable[[dict[str, Any]], bool]


@dataclass
class RiskManager:
    max_leverage: float = 3.0
    max_notional: float | None = None
    long_only: bool = True
    kill_switch_hooks: list[KillSwitchHook] = field(default_factory=list)

    def max_qty_for_equity(self, equity: float, price: float) -> float:
        if equity <= 0 or price <= 0:
            return 0.0

        cap_notional = equity * self.max_leverage
        if self.max_notional is not None:
            cap_notional = min(cap_notional, self.max_notional)
        return max(0.0, cap_notional / price)

    def cap_target_qty(self, target_qty: float, price: float, equity: float) -> float:
        if self.long_only and target_qty < 0:
            return 0.0

        cap_qty = self.max_qty_for_equity(equity=equity, price=price)
        if target_qty >= 0:
            return min(target_qty, cap_qty)
        return max(target_qty, -cap_qty)

    def cap_order_qty(
        self,
        requested_qty: float,
        side: str,
        current_qty: float,
        price: float,
        equity: float,
    ) -> float:
        if requested_qty <= 0:
            return 0.0

        if side == "buy":
            available = self.max_qty_for_equity(equity=equity, price=price) - max(current_qty, 0.0)
            return max(0.0, min(requested_qty, available))

        if side == "sell" and self.long_only:
            return max(0.0, min(requested_qty, max(current_qty, 0.0)))

        return requested_qty

    def should_kill(self, state: dict[str, Any]) -> bool:
        return any(hook(state) for hook in self.kill_switch_hooks)

    def compute_atr_stop_price(self, entry_price: float, stop_distance: float, side: str = "long") -> float | None:
        if stop_distance <= 0:
            return None
        if side == "long":
            return max(0.0, entry_price - stop_distance)
        return entry_price + stop_distance

    def is_stop_hit(self, bar: Bar, position: Position) -> bool:
        if position.quantity > 0 and position.stop_price is not None:
            return bar.low <= position.stop_price
        if position.quantity < 0 and position.stop_price is not None:
            return bar.high >= position.stop_price
        return False
