"""Average True Range."""

from __future__ import annotations


def calculate_atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> list[float | None]:
    if period <= 0:
        raise ValueError("period must be > 0")
    if not highs or not lows or not closes:
        return []
    if not (len(highs) == len(lows) == len(closes)):
        raise ValueError("highs, lows, closes length mismatch")

    tr_values: list[float] = []
    for idx in range(len(closes)):
        if idx == 0:
            tr = highs[idx] - lows[idx]
        else:
            tr = max(
                highs[idx] - lows[idx],
                abs(highs[idx] - closes[idx - 1]),
                abs(lows[idx] - closes[idx - 1]),
            )
        tr_values.append(tr)

    atr: list[float | None] = [None] * len(closes)
    if len(closes) < period:
        return atr

    initial = sum(tr_values[:period]) / period
    atr[period - 1] = initial
    prev = initial
    for idx in range(period, len(tr_values)):
        prev = ((prev * (period - 1)) + tr_values[idx]) / period
        atr[idx] = prev

    return atr
