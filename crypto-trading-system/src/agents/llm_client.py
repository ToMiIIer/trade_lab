"""LLM client abstraction with a deterministic mock provider for MVP."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from src.agents.base_agent import ModelConfig


@dataclass(slots=True)
class LLMCompletion:
    action: str
    confidence: float
    reasoning: str
    risk_notes: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "risk_notes": self.risk_notes,
        }


class MockLLMProvider:
    """Deterministic provider used for tests and offline development."""

    def complete(self, agent_id: str, _prompt: str, context: dict[str, Any]) -> LLMCompletion:
        if "technical" in agent_id:
            return self._technical(context)
        if "sentiment" in agent_id:
            return self._sentiment(context)
        return LLMCompletion(
            action="HOLD",
            confidence=0.50,
            reasoning="Unknown agent type, defaulting to HOLD.",
            risk_notes="Fallback behavior.",
        )

    def _technical(self, context: dict[str, Any]) -> LLMCompletion:
        indicators = context.get("indicators", {})
        ema_21 = indicators.get("ema_21")
        ema_50 = indicators.get("ema_50")
        rsi_14 = indicators.get("rsi_14")

        if ema_21 is None or ema_50 is None or rsi_14 is None:
            return LLMCompletion(
                action="HOLD",
                confidence=0.45,
                reasoning="Missing technical inputs.",
                risk_notes="Indicator completeness required.",
            )

        if ema_21 > ema_50 and rsi_14 < 70:
            confidence = min(0.90, 0.55 + ((70 - rsi_14) / 100.0))
            return LLMCompletion(
                action="BUY",
                confidence=round(confidence, 4),
                reasoning="Trend is constructive and RSI is below overbought.",
                risk_notes="Watch volatility around resistance.",
            )

        if ema_21 < ema_50 and rsi_14 > 30:
            confidence = min(0.90, 0.55 + ((rsi_14 - 30) / 100.0))
            return LLMCompletion(
                action="SELL",
                confidence=round(confidence, 4),
                reasoning="Downtrend bias with weakening momentum.",
                risk_notes="Short signals are sensitive to sharp reversals.",
            )

        return LLMCompletion(
            action="HOLD",
            confidence=0.50,
            reasoning="Mixed technical conditions.",
            risk_notes="Await clearer setup.",
        )

    def _sentiment(self, context: dict[str, Any]) -> LLMCompletion:
        sentiment = context.get("sentiment", {})
        score = float(sentiment.get("score", 0.0))

        if score > 0.2:
            return LLMCompletion(
                action="BUY",
                confidence=round(min(0.90, 0.50 + (score / 2.0)), 4),
                reasoning="Positive sentiment tilt.",
                risk_notes="Sentiment can invert quickly on headlines.",
            )

        if score < -0.2:
            return LLMCompletion(
                action="SELL",
                confidence=round(min(0.90, 0.50 + (abs(score) / 2.0)), 4),
                reasoning="Negative sentiment pressure.",
                risk_notes="Bearish signals may squeeze on short covering.",
            )

        return LLMCompletion(
            action="HOLD",
            confidence=0.50,
            reasoning="Neutral sentiment profile.",
            risk_notes="No directional edge from sentiment.",
        )


class MultiProviderLLMClient:
    """Selects provider, defaulting to mock for MVP fail-safe operation."""

    def __init__(self) -> None:
        self.mock = MockLLMProvider()

    def _provider_available(self, provider: str) -> bool:
        key_lookup = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
        }
        env_key = key_lookup.get(provider)
        if not env_key:
            return False
        return bool(os.getenv(env_key))

    def complete(
        self,
        agent_id: str,
        model_cfg: ModelConfig,
        prompt: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        provider = model_cfg.provider.lower().strip()

        # Real provider integrations are intentionally deferred for Phase 1.
        # If credentials are missing or provider is unsupported, deterministic mock is used.
        if provider == "mock" or not self._provider_available(provider):
            return self.mock.complete(agent_id=agent_id, _prompt=prompt, context=context).as_dict()

        return self.mock.complete(agent_id=agent_id, _prompt=prompt, context=context).as_dict()
