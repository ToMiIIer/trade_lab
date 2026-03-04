"""Paper execution node. Never sends live orders in Phase 1."""

from __future__ import annotations

from typing import Any

from src.risk.manager import RiskDecision
from src.storage.repository import StorageRepository


class PaperExecutorNode:
    def __init__(self, repository: StorageRepository) -> None:
        self.repository = repository

    def run(
        self,
        *,
        run_id: str,
        pair: str,
        timeframe: str,
        risk_decision: RiskDecision,
        market_data: dict[str, Any],
        portfolio_state: dict[str, Any],
    ) -> dict[str, Any]:
        if not risk_decision.approved:
            return {
                "status": "NO_TRADE",
                "reason": risk_decision.reason,
                "action": "HOLD",
            }

        price = float(
            market_data.get("ticker_24h", {}).get("last_price")
            or market_data.get("last_price")
            or 0.0
        )
        if price <= 0:
            return {
                "status": "NO_TRADE",
                "reason": "invalid_execution_price",
                "action": "HOLD",
            }

        equity = float(portfolio_state.get("equity", 10000.0))
        notional = equity * risk_decision.position_pct
        quantity = notional / price

        trade = self.repository.create_simulated_trade(
            run_id=run_id,
            pair=pair,
            timeframe=timeframe,
            action=risk_decision.action,
            quantity=quantity,
            entry_price=price,
            reason=risk_decision.reason,
        )

        return {
            "status": "SIMULATED_TRADE",
            "trade_id": trade.id,
            "action": trade.action,
            "quantity": trade.quantity,
            "entry_price": trade.entry_price,
            "reason": trade.reason,
        }
