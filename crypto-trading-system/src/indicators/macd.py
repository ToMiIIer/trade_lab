"""Moving Average Convergence Divergence."""

from __future__ import annotations

from src.indicators.ema import calculate_ema


def calculate_macd(
    closes: list[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> dict[str, list[float | None]]:
    if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
        raise ValueError("all periods must be > 0")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be less than slow_period")

    fast = calculate_ema(closes, fast_period)
    slow = calculate_ema(closes, slow_period)

    macd_line: list[float | None] = [None] * len(closes)
    for idx in range(len(closes)):
        if fast[idx] is not None and slow[idx] is not None:
            macd_line[idx] = float(fast[idx]) - float(slow[idx])

    dense_macd = [v for v in macd_line if v is not None]
    signal_dense = calculate_ema([float(v) for v in dense_macd], signal_period)

    signal_line: list[float | None] = [None] * len(closes)
    histogram: list[float | None] = [None] * len(closes)

    macd_indices = [idx for idx, value in enumerate(macd_line) if value is not None]
    for dense_idx, source_idx in enumerate(macd_indices):
        signal_value = signal_dense[dense_idx]
        signal_line[source_idx] = signal_value
        if signal_value is not None and macd_line[source_idx] is not None:
            histogram[source_idx] = float(macd_line[source_idx]) - float(signal_value)

    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }
