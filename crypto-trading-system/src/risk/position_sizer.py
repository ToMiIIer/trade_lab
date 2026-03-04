"""Position sizing helper used by the hardcoded risk engine."""

from __future__ import annotations


def calculate_position_pct(
    *,
    max_position_pct: float,
    risk_per_trade_pct: float,
    stop_distance_pct: float,
    max_symbol_exposure_pct: float,
    current_symbol_exposure_pct: float,
) -> float:
    if stop_distance_pct <= 0:
        return 0.0

    risk_based_cap = risk_per_trade_pct / stop_distance_pct
    exposure_headroom = max(0.0, max_symbol_exposure_pct - current_symbol_exposure_pct)

    return max(
        0.0,
        min(max_position_pct, risk_based_cap, exposure_headroom),
    )
