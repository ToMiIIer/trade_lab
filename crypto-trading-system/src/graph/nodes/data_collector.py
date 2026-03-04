"""Graph node that gathers market data from configured sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.data.price_fetcher import BinancePriceFetcher, OHLCV, PriceFetcherError


class DataCollectorError(RuntimeError):
    """Raised when data collection is incomplete."""


@dataclass(slots=True)
class MarketDataBundle:
    pair: str
    timeframe: str
    ohlcv: list[OHLCV]
    ticker_24h: dict[str, Any]


class DataCollectorNode:
    """Collects market inputs required by the pipeline."""

    def __init__(self, fetcher: BinancePriceFetcher, source_cfg: dict[str, Any]) -> None:
        self.fetcher = fetcher
        self.source_cfg = source_cfg

    def run(self, pair: str, timeframe: str) -> MarketDataBundle:
        try:
            ohlcv = self.fetcher.fetch_ohlcv(
                pair=pair,
                timeframe=timeframe,
                endpoint=self.source_cfg["klines_endpoint"],
                limit=int(self.source_cfg.get("default_limit", 300)),
            )
            ticker = self.fetcher.fetch_24h_ticker(
                pair=pair,
                endpoint=self.source_cfg["ticker_endpoint"],
            )
        except (KeyError, PriceFetcherError, ValueError, TypeError) as exc:
            raise DataCollectorError(f"failed collecting market data: {exc}") from exc

        if not ohlcv:
            raise DataCollectorError("no candles returned")

        return MarketDataBundle(pair=pair, timeframe=timeframe, ohlcv=ohlcv, ticker_24h=ticker)
