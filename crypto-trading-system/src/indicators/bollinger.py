"""Bollinger Bands."""

from __future__ import annotations

import math


def calculate_bollinger_bands(
    closes: list[float],
    period: int = 20,
    std_dev_mult: float = 2.0,
) -> dict[str, list[float | None]]:
    if period <= 0:
        raise ValueError("period must be > 0")
    if std_dev_mult <= 0:
        raise ValueError("std_dev_mult must be > 0")
    if not closes:
        return {"middle": [], "upper": [], "lower": []}

    middle: list[float | None] = [None] * len(closes)
    upper: list[float | None] = [None] * len(closes)
    lower: list[float | None] = [None] * len(closes)

    if len(closes) < period:
        return {"middle": middle, "upper": upper, "lower": lower}

    for idx in range(period - 1, len(closes)):
        window = closes[idx - period + 1 : idx + 1]
        mean = sum(window) / period
        variance = sum((value - mean) ** 2 for value in window) / period
        std_dev = math.sqrt(variance)

        middle[idx] = mean
        upper[idx] = mean + (std_dev_mult * std_dev)
        lower[idx] = mean - (std_dev_mult * std_dev)

    return {"middle": middle, "upper": upper, "lower": lower}
