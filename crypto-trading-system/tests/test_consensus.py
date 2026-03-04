from __future__ import annotations

import unittest

from src.agents.base_agent import AgentResult
from src.graph.nodes.consensus import WeightedConsensusEngine


class ConsensusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = WeightedConsensusEngine(
            {
                "mode": "weighted_voting",
                "default_weights": {
                    "technical_analyst": 0.65,
                    "sentiment_analyst": 0.35,
                },
                "min_weighted_confidence": 0.55,
                "hold_threshold": 0.50,
            }
        )

    def test_weighted_buy_passes_threshold(self) -> None:
        hypotheses = [
            AgentResult("r1", "BTC/USDC", "4h", "technical_analyst", "BUY", 0.95, "", ""),
            AgentResult("r1", "BTC/USDC", "4h", "sentiment_analyst", "BUY", 0.60, "", ""),
        ]

        decision = self.engine.run("r1", "BTC/USDC", "4h", hypotheses)

        self.assertEqual(decision.action, "BUY")
        self.assertTrue(decision.threshold_passed)
        self.assertGreaterEqual(decision.weighted_confidence, 0.55)

    def test_low_confidence_forces_hold(self) -> None:
        hypotheses = [
            AgentResult("r2", "BTC/USDC", "4h", "technical_analyst", "BUY", 0.40, "", ""),
            AgentResult("r2", "BTC/USDC", "4h", "sentiment_analyst", "BUY", 0.45, "", ""),
        ]

        decision = self.engine.run("r2", "BTC/USDC", "4h", hypotheses)

        self.assertEqual(decision.action, "HOLD")
        self.assertFalse(decision.threshold_passed)

    def test_missing_weight_uses_default_of_one(self) -> None:
        hypotheses = [
            AgentResult("r3", "ETH/USDC", "1h", "unknown_agent", "SELL", 0.80, "", ""),
            AgentResult("r3", "ETH/USDC", "1h", "sentiment_analyst", "BUY", 0.10, "", ""),
        ]

        decision = self.engine.run("r3", "ETH/USDC", "1h", hypotheses)

        self.assertEqual(decision.weights_used["unknown_agent"], 1.0)
        self.assertEqual(decision.action, "SELL")


if __name__ == "__main__":
    unittest.main()
