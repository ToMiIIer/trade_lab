"""Exponential moving average."""

from __future__ import annotations


def calculate_ema(values: list[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period must be > 0")
    if not values:
        return []

    ema_values: list[float | None] = [None] * len(values)
    if len(values) < period:
        return ema_values

    multiplier = 2.0 / (period + 1)
    seed = sum(values[:period]) / period
    ema_values[period - 1] = seed

    prev = seed
    for idx in range(period, len(values)):
        prev = (values[idx] - prev) * multiplier + prev
        ema_values[idx] = prev

    return ema_values
