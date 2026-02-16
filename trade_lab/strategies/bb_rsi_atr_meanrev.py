from __future__ import annotations

from typing import Any

import pandas as pd

from core.indicators import atr, bollinger_bands, rsi
from core.types import Bar, TargetPosition
from strategies.base import BaseStrategy


class Strategy(BaseStrategy):
    """
    Mean-reversion strategy for BTC perpetual on 4h bars.

    Regime filter uses ATR% = ATR / close and only allows entries when ATR%
    is below a configurable threshold (low-vol / ranging proxy).
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.indicators: pd.DataFrame | None = None

    @classmethod
    def default_parameters(cls) -> dict[str, Any]:
        return {
            "bb_length": 20,
            "bb_std": 2.0,
            "rsi_length": 14,
            "atr_length": 14,
            "regime_atr_pct_threshold": 0.02,
            "rsi_entry": 30.0,
            "rsi_exit": 70.0,
            "atr_k": 2.0,
            "target_notional_fraction": 0.5,
        }

    def initialize(self, state: dict[str, Any]) -> None:
        data = state.get("data")
        if data is None or not isinstance(data, pd.DataFrame):
            raise ValueError("Strategy requires state['data'] DataFrame")

        params = self.default_parameters() | self.config.params

        bb = bollinger_bands(
            close=data["close"],
            length=int(params["bb_length"]),
            stdev=float(params["bb_std"]),
        )
        rsi_series = rsi(close=data["close"], length=int(params["rsi_length"]))
        atr_series = atr(
            high=data["high"],
            low=data["low"],
            close=data["close"],
            length=int(params["atr_length"]),
        )

        indicators = pd.DataFrame(
            {
                "bb_upper": bb["bb_upper"],
                "bb_lower": bb["bb_lower"],
                "rsi": rsi_series,
                "atr": atr_series,
                "atr_pct": atr_series / data["close"].replace(0, pd.NA),
            }
        )

        self.indicators = indicators
        state["indicators"] = indicators

    def on_bar(self, i: int, bar: Bar, state: dict[str, Any]) -> TargetPosition | None:
        if self.indicators is None:
            raise RuntimeError("Strategy not initialized")

        signal = self.indicators.iloc[i]
        if signal.isna().any():
            return None

        params = self.default_parameters() | self.config.params
        position = state["position"]
        equity = float(state["equity"])
        stop_hit = bool(state.get("atr_stop_hit", False))

        regime_ok = float(signal["atr_pct"]) <= float(params["regime_atr_pct_threshold"])

        if position.quantity <= 0:
            long_entry = (
                bar.close <= float(signal["bb_lower"])
                and float(signal["rsi"]) < float(params["rsi_entry"])
                and regime_ok
            )
            if long_entry:
                target_notional = max(0.0, equity * float(params["target_notional_fraction"]))
                target_qty = target_notional / bar.close if bar.close > 0 else 0.0
                stop_distance = float(signal["atr"]) * float(params["atr_k"])
                return TargetPosition(
                    target_qty=target_qty,
                    reason="meanrev_entry",
                    stop_distance=stop_distance,
                )
            return None

        long_exit = (
            stop_hit
            or bar.close >= float(signal["bb_upper"])
            or float(signal["rsi"]) > float(params["rsi_exit"])
        )
        if long_exit:
            return TargetPosition(target_qty=0.0, reason="meanrev_exit")

        return TargetPosition(target_qty=float(position.quantity), reason="hold")
