from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd


@dataclass(slots=True)
class Bar:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_row(cls, row: pd.Series) -> "Bar":
        return cls(
            timestamp=pd.Timestamp(row["timestamp"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )


@dataclass(slots=True)
class OrderIntent:
    side: Literal["buy", "sell"]
    quantity: float
    reason: str = ""


@dataclass(slots=True)
class TargetPosition:
    target_qty: float
    reason: str = ""
    stop_distance: float | None = None


@dataclass(slots=True)
class Fill:
    timestamp: pd.Timestamp
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    notional: float
    fee: float
    slippage: float
    reason: str = ""


@dataclass(slots=True)
class Position:
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    entry_timestamp: pd.Timestamp | None = None
    stop_price: float | None = None


@dataclass(slots=True)
class RunConfig:
    symbol: str = "BTC-PERP"
    timeframe: str = "4h"
    strategy_name: str = "bb_rsi_atr_meanrev"
    strategy_params: dict[str, Any] = field(default_factory=dict)
    initial_cash: float = 10_000.0
    fee_rate: float = 0.0005
    slippage_bps: float = 2.0
    max_leverage: float = 3.0
    max_notional: float | None = None
    long_only: bool = True
    execution_mode: Literal["same_close", "next_open"] = "next_open"
    run_name: str = ""


@dataclass
class RunResult:
    config: RunConfig
    metrics: dict[str, float]
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    fills: pd.DataFrame
    run_id: int | None = None
