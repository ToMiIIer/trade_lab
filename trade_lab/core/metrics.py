from __future__ import annotations

import numpy as np
import pandas as pd


def add_drawdown(equity_curve: pd.DataFrame) -> pd.DataFrame:
    if equity_curve.empty:
        output = equity_curve.copy()
        output["drawdown"] = []
        return output

    output = equity_curve.copy()
    running_peak = output["equity"].cummax()
    output["drawdown"] = output["equity"] / running_peak - 1.0
    return output


def _periods_per_year(timeframe: str) -> float:
    key = timeframe.lower()
    if key == "4h":
        return 6 * 365
    if key == "1d":
        return 365
    if key == "1h":
        return 24 * 365
    return 365


def calculate_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    timeframe: str,
) -> dict[str, float]:
    if equity_curve.empty:
        return {
            "initial_equity": 0.0,
            "final_equity": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "avg_trade_pnl": 0.0,
            "avg_trade_return": 0.0,
            "num_trades": 0.0,
            "exposure": 0.0,
        }

    initial = float(equity_curve["equity"].iloc[0])
    final = float(equity_curve["equity"].iloc[-1])
    total_return = (final / initial - 1.0) if initial else 0.0

    enriched_curve = add_drawdown(equity_curve)
    max_drawdown = float(enriched_curve["drawdown"].min()) if not enriched_curve.empty else 0.0

    returns = equity_curve["equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = 0.0
    if len(returns) > 1 and returns.std() > 0:
        sharpe = float((returns.mean() / returns.std()) * np.sqrt(_periods_per_year(timeframe)))

    num_trades = int(len(trades))
    win_rate = float((trades["pnl"] > 0).mean()) if num_trades else 0.0
    avg_trade_pnl = float(trades["pnl"].mean()) if num_trades else 0.0
    avg_trade_return = float(trades["pnl_pct"].mean()) if num_trades else 0.0

    exposure = float((equity_curve["position_qty"].abs() > 1e-12).mean())

    return {
        "initial_equity": initial,
        "final_equity": final,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "avg_trade_pnl": avg_trade_pnl,
        "avg_trade_return": avg_trade_return,
        "num_trades": float(num_trades),
        "exposure": exposure,
    }
