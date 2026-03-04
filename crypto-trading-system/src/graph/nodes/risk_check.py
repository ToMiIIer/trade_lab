"""Graph node wrapping risk manager checks."""

from __future__ import annotations

from typing import Any

from src.graph.nodes.consensus import ConsensusDecision
from src.risk.manager import RiskDecision, RiskManager


class RiskCheckNode:
    def __init__(self, manager: RiskManager) -> None:
        self.manager = manager

    def run(
        self,
        consensus: ConsensusDecision,
        portfolio_state: dict[str, Any],
        market_state: dict[str, Any],
    ) -> RiskDecision:
        return self.manager.evaluate(
            action=consensus.action,
            weighted_confidence=consensus.weighted_confidence,
            portfolio_state=portfolio_state,
            market_state=market_state,
        )
