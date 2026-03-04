"""Weighted consensus logic for combining agent hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.agents.base_agent import AgentResult

ConsensusAction = Literal["BUY", "SELL", "HOLD"]


class ConsensusError(RuntimeError):
    """Raised when consensus cannot be computed."""


@dataclass(slots=True)
class ConsensusDecision:
    run_id: str
    pair: str
    timeframe: str
    action: ConsensusAction
    weighted_confidence: float
    threshold_passed: bool
    scores: dict[str, float]
    weights_used: dict[str, float]
    reasoning: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "pair": self.pair,
            "timeframe": self.timeframe,
            "action": self.action,
            "weighted_confidence": self.weighted_confidence,
            "threshold_passed": self.threshold_passed,
            "scores": self.scores,
            "weights_used": self.weights_used,
            "reasoning": self.reasoning,
        }


class WeightedConsensusEngine:
    def __init__(self, config: dict[str, Any]) -> None:
        self.mode = str(config.get("mode", "weighted_voting"))
        self.default_weights = {
            str(agent): float(weight)
            for agent, weight in dict(config.get("default_weights", {})).items()
        }
        self.min_weighted_confidence = float(config.get("min_weighted_confidence", 0.55))
        self.hold_threshold = float(config.get("hold_threshold", 0.50))

    def run(
        self,
        run_id: str,
        pair: str,
        timeframe: str,
        hypotheses: list[AgentResult],
    ) -> ConsensusDecision:
        if self.mode != "weighted_voting":
            raise ConsensusError(f"unsupported consensus mode: {self.mode}")
        if not hypotheses:
            raise ConsensusError("cannot compute consensus without hypotheses")

        scores = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        weights_used: dict[str, float] = {}
        total_weight = 0.0

        for hypothesis in hypotheses:
            weight = self.default_weights.get(hypothesis.agent_id, 1.0)
            score = weight * max(0.0, min(1.0, hypothesis.confidence))
            scores[hypothesis.action] += score
            weights_used[hypothesis.agent_id] = weight
            total_weight += weight

        if total_weight <= 0:
            raise ConsensusError("invalid total voting weight")

        winning_action = max(scores.keys(), key=lambda action: scores[action])
        winning_score = scores[winning_action]
        weighted_confidence = winning_score / total_weight

        passed = (
            weighted_confidence >= self.min_weighted_confidence
            and weighted_confidence >= self.hold_threshold
            and winning_action != "HOLD"
        )

        final_action: ConsensusAction = winning_action if passed else "HOLD"
        reasoning = (
            f"winner={winning_action} weighted_confidence={weighted_confidence:.4f} "
            f"min={self.min_weighted_confidence:.4f} hold={self.hold_threshold:.4f}"
        )

        return ConsensusDecision(
            run_id=run_id,
            pair=pair,
            timeframe=timeframe,
            action=final_action,
            weighted_confidence=weighted_confidence,
            threshold_passed=passed,
            scores=scores,
            weights_used=weights_used,
            reasoning=reasoning,
        )
