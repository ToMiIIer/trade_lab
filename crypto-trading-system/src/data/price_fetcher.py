"""Binance market data fetcher for OHLCV and 24h ticker snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

VALID_TIMEFRAMES = {"1h", "4h", "1d"}


class PriceFetcherError(RuntimeError):
    """Raised when market data cannot be collected."""


@dataclass(slots=True)
class OHLCV:
    """Normalized OHLCV candle."""

    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int


class BinancePriceFetcher:
    """Simple Binance REST API client."""

    def __init__(self, base_url: str, timeout_seconds: int = 10) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def normalize_pair(pair: str) -> str:
        return pair.replace("/", "")

    def _request(self, path: str, params: dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                return response.json()
        except Exception as exc:  # pragma: no cover - network failures are integration-level
            raise PriceFetcherError(f"request failed for {url}: {exc}") from exc

    def fetch_ohlcv(
        self,
        pair: str,
        timeframe: str,
        endpoint: str,
        limit: int = 300,
    ) -> list[OHLCV]:
        if timeframe not in VALID_TIMEFRAMES:
            raise PriceFetcherError(f"unsupported timeframe: {timeframe}")

        payload = self._request(
            endpoint,
            {
                "symbol": self.normalize_pair(pair),
                "interval": timeframe,
                "limit": limit,
            },
        )
        if not isinstance(payload, list) or not payload:
            raise PriceFetcherError("empty ohlcv payload")

        candles: list[OHLCV] = []
        for item in payload:
            # Binance kline schema indices: 0=open_time,1=open,2=high,3=low,4=close,5=volume,6=close_time
            candles.append(
                OHLCV(
                    open_time=int(item[0]),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                    close_time=int(item[6]),
                )
            )
        return candles

    def fetch_24h_ticker(self, pair: str, endpoint: str) -> dict[str, Any]:
        payload = self._request(endpoint, {"symbol": self.normalize_pair(pair)})
        if not isinstance(payload, dict) or "lastPrice" not in payload:
            raise PriceFetcherError("invalid 24h ticker payload")

        return {
            "symbol": payload.get("symbol"),
            "price_change_percent": float(payload.get("priceChangePercent", 0.0)),
            "last_price": float(payload.get("lastPrice", 0.0)),
            "volume": float(payload.get("volume", 0.0)),
            "quote_volume": float(payload.get("quoteVolume", 0.0)),
            "high_price": float(payload.get("highPrice", 0.0)),
            "low_price": float(payload.get("lowPrice", 0.0)),
        }
