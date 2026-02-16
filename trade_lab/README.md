# Perps Strategy Lab (MVP)

Minimal, pluggable backtest core + Streamlit dashboard for BTC perpetual futures on the 4h timeframe.

## What this MVP includes

- Modular backtest engine (`core/engine.py`) with strategy plugin loading by name.
- Strategy contract (`strategies/base.py`) where strategies can only output intent (`TargetPosition` or `OrderIntent`).
- Simulated broker (`core/broker_sim.py`) with fee/slippage models.
- Risk controls (`core/risk.py`) including leverage cap (3x default), optional notional cap, kill-switch hooks, and ATR stop handling.
- Metrics (`core/metrics.py`): total return, max drawdown, Sharpe (approx), win rate, average trade, exposure.
- SQLite persistence (`core/storage.py`) for runs, trades, and equity curve.
- Streamlit dashboard (`app.py`) to run backtests and inspect both fresh and saved runs.

## Strategy implemented

Initial strategy plugin: `strategies/bb_rsi_atr_meanrev.py`

- Indicators:
  - Bollinger Bands (default length=20, stdev=2)
  - RSI (default length=14)
  - ATR (default length=14)
- Regime filter choice:
  - **ATR% filter** (`atr / close`) with configurable threshold (`regime_atr_pct_threshold`, default `0.02`).
- Long entry:
  - `close <= lower_band`
  - `RSI < rsi_entry` (default 30)
  - regime filter passes
- Exit:
  - `close >= upper_band` OR `RSI > rsi_exit` (default 70) OR ATR stop hit
- Stop:
  - `ATR * atr_k` distance (default `2.0`) pushed to engine via `TargetPosition.stop_distance`
- Position sizing:
  - target notional fraction of equity (`target_notional_fraction`, default `0.5`), still capped by leverage risk.

## Setup

```bash
python -m venv .venv; source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run from the `trade_lab/` directory.

## Manual test steps

1. Start Streamlit with `streamlit run app.py`.
2. Keep default sample CSV (`data/sample_btc_4h.csv`) and default strategy.
3. Click **Run Backtest**.
4. Verify metrics appear and equity/drawdown charts render.
5. Verify run appears in **Saved Runs**.
6. Select that run in **Open run** and click **Open Selected Run**.
7. Confirm saved metrics/charts/trades load correctly from SQLite.

## Self-check

A minimal smoke check is included:

```bash
python -m core.self_check
```

Because this environment has no network access, dependency installation is blocked here, so runtime smoke execution cannot be run until dependencies are installed locally.

Dependency-free check run in this environment:

- `python3 -m compileall trade_lab` -> PASS on February 16, 2026 (all project `.py` files compiled successfully).

When dependencies are installed (`pip install -r requirements.txt`), run:

- `python -m core.self_check`

## How to add a new strategy (single new file)

1. Create one file in `strategies/`, for example `strategies/my_new_strategy.py`.
2. Implement a `Strategy` class that extends `BaseStrategy`.
3. Implement:
   - `default_parameters()`
   - `initialize(state)`
   - `on_bar(i, bar, state)` returning `TargetPosition`, `list[OrderIntent]`, or `None`.
4. Launch the app: the new strategy is auto-discovered by filename; no engine changes required.

## Notes

- Long-only execution is enabled for MVP, but engine/risk abstractions are ready for short-side extension.
- No live exchange integration is included in this MVP.
