"""Sentiment fetcher stub for MVP Phase 1."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SentimentSnapshot:
    pair: str
    score: float
    label: str
    source: str


class SentimentFetcher:
    """Returns deterministic placeholder sentiment values."""

    def fetch(self, pair: str) -> SentimentSnapshot:
        return SentimentSnapshot(
            pair=pair,
            score=0.0,
            label="neutral",
            source="mvp_stub",
        )
