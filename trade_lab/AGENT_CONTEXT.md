# AGENT CONTEXT (trade_lab)

## Product Source of Truth
- Product: Perps Strategy Lab.
- Scope: backtest core + parameter sweeps + Streamlit UI.
- Data source workflow: local CSV datasets, including Binance downloader outputs.
- Primary UX: run backtests, view metrics/equity/trades, run sweeps, save runs.

## Strategy Source of Truth
- Current strategy plugin: `bb_rsi_atr_meanrev`.
- Logic family: mean-reversion using Bollinger Bands + RSI.
- Risk exit: ATR-based stop.
- Regime gate: optional low-vol/ranging filter (ATR% threshold).
- Trend gate: optional ADX filter.
- Leverage policy: risk cap 3x exists; strategy sizing uses target notional fraction.

## Backtest Correctness Rules (Critical)
1. Fills execute at next bar open in realistic mode (`execution_mode = "next_open"`).
2. Strategy must not access future bars during `on_bar`; pass only history slice `df.iloc[:i+1]`.
3. ATR stop must not trigger on the same bar as entry.

## Data Conventions
- Candle timestamps are UTC.
- Streamlit dataset selector reads CSV files from `trade_lab/data`.
- UI should clearly show selected file and load summary:
  - "Loaded N bars from <first_ts> to <last_ts>"

## Evaluation Process
1. Find candidate edge on train range: 2019-2024.
2. Validate out-of-sample on 2025.
3. Run 2D sweeps on key parameters.
4. Prefer results with adequate sample size (minimum ~100 trades) to reduce noise.

## Known Pitfalls (Already Hit)
- SSL certificates on macOS can fail without proper cert handling.
- Virtual environment path confusion (repo root `.venv` usage matters).
- Path handling must use stable `DATA_DIR`; otherwise page refresh can falsely show missing files and trigger repeated download prompts.
