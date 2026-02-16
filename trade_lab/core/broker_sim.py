from __future__ import annotations

from core.types import Fill


class BrokerSim:
    def __init__(self, fee_rate: float, slippage_bps: float) -> None:
        if fee_rate < 0:
            raise ValueError("fee_rate must be >= 0")
        if slippage_bps < 0:
            raise ValueError("slippage_bps must be >= 0")
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps

    def execute_market_order(
        self,
        timestamp,
        side: str,
        quantity: float,
        reference_price: float,
        reason: str = "",
    ) -> Fill:
        if side not in {"buy", "sell"}:
            raise ValueError(f"Unsupported side: {side}")
        if quantity <= 0:
            raise ValueError("quantity must be > 0")
        if reference_price <= 0:
            raise ValueError("reference_price must be > 0")

        slip_factor = self.slippage_bps / 10_000
        if side == "buy":
            fill_price = reference_price * (1 + slip_factor)
            slippage = fill_price - reference_price
        else:
            fill_price = reference_price * (1 - slip_factor)
            slippage = reference_price - fill_price

        notional = quantity * fill_price
        fee = notional * self.fee_rate
        return Fill(
            timestamp=timestamp,
            side=side,
            quantity=quantity,
            price=fill_price,
            notional=notional,
            fee=fee,
            slippage=slippage,
            reason=reason,
        )
