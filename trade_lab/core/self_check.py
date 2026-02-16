from __future__ import annotations

from pathlib import Path

import yaml

from core.data_io import load_ohlcv_csv
from core.engine import BacktestEngine
from core.types import RunConfig
from strategies import load_strategy


def build_run_config(defaults: dict, strategy_params: dict) -> RunConfig:
    return RunConfig(
        symbol=defaults.get("symbol", "BTC-PERP"),
        timeframe=defaults.get("timeframe", "4h"),
        strategy_name=defaults.get("strategy", {}).get("name", "bb_rsi_atr_meanrev"),
        strategy_params=strategy_params,
        initial_cash=float(defaults.get("initial_cash", 10_000.0)),
        fee_rate=float(defaults.get("fees", {}).get("fee_rate", 0.0005)),
        slippage_bps=float(defaults.get("fees", {}).get("slippage_bps", 2.0)),
        max_leverage=float(defaults.get("risk", {}).get("max_leverage", 3.0)),
        max_notional=defaults.get("risk", {}).get("max_notional"),
        long_only=True,
        run_name="self-check",
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    defaults = yaml.safe_load((root / "configs/default.yaml").read_text(encoding="utf-8"))

    strategy_name = defaults.get("strategy", {}).get("name", "bb_rsi_atr_meanrev")
    strategy_params = defaults.get("strategy", {}).get("params", {})

    data = load_ohlcv_csv(root / "data/sample_btc_4h.csv")
    config = build_run_config(defaults, strategy_params)

    strategy = load_strategy(strategy_name, strategy_params)
    result = BacktestEngine(config=config).run(data=data, strategy=strategy)

    assert len(result.equity_curve) == len(data), "Equity curve length must match bars"
    assert result.metrics["final_equity"] > 0, "Final equity must be positive"

    print(
        "SELF_CHECK PASS "
        f"bars={len(data)} "
        f"trades={int(result.metrics['num_trades'])} "
        f"total_return={result.metrics['total_return']:.4f} "
        f"max_dd={result.metrics['max_drawdown']:.4f}"
    )


if __name__ == "__main__":
    main()
