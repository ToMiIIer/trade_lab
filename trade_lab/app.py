from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

from core.data_io import filter_date_range, load_ohlcv_csv
from core.engine import BacktestEngine
from core.storage import SQLiteStorage
from core.types import RunConfig
from strategies import discover_strategy_names, get_default_parameters, load_strategy
from tools.download_binance_klines import download_binance_klines, download_btcusdt_4h_last_3y

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SWEEPS_DIR = DATA_DIR / "sweeps"
DEFAULT_CONFIG_PATH = BASE_DIR / "configs/default.yaml"
DEFAULT_DATA_PATH = DATA_DIR / "sample_btc_4h.csv"
DOWNLOADED_DATA_PATH = DATA_DIR / "btcusdt_4h_3y.csv"
DOWNLOADED_1H_2019_2025_PATH = DATA_DIR / "btcusdt_1h_2019_2025.csv"
DEFAULT_DB_PATH = BASE_DIR / "runs.sqlite3"

SWEEP_OPTIONS: dict[str, dict[str, Any]] = {
    "ATR Stop Multiplier (k)": {
        "slug": "k",
        "target": "strategy",
        "keys": ["atr_k", "atr_stop_mult"],
        "as_int": False,
        "allow_zero": False,
        "default_values": "1.5,2.0,2.5,3.0,3.5",
    },
    "RSI Entry Threshold": {
        "slug": "rsi_entry",
        "target": "strategy",
        "keys": ["rsi_entry"],
        "as_int": False,
        "allow_zero": False,
        "default_values": "15,20,25,30,35",
    },
    "RSI Exit Threshold": {
        "slug": "rsi_exit",
        "target": "strategy",
        "keys": ["rsi_exit"],
        "as_int": False,
        "allow_zero": False,
        "default_values": "60,65,70,75,80",
    },
    "Bollinger Length": {
        "slug": "bb_length",
        "target": "strategy",
        "keys": ["bb_length"],
        "as_int": True,
        "allow_zero": False,
        "default_values": "10,15,20,25,30",
    },
    "Bollinger StdDev (multiplier)": {
        "slug": "bb_std",
        "target": "strategy",
        "keys": ["bb_std"],
        "as_int": False,
        "allow_zero": False,
        "default_values": "1.5,2.0,2.5,3.0",
    },
    "ATR Length": {
        "slug": "atr_length",
        "target": "strategy",
        "keys": ["atr_length"],
        "as_int": True,
        "allow_zero": False,
        "default_values": "7,10,14,21",
    },
    "Regime ATR % Threshold": {
        "slug": "regime_atr_pct_threshold",
        "target": "strategy",
        "keys": ["regime_atr_pct_threshold"],
        "as_int": False,
        "allow_zero": False,
        "default_values": "0.005,0.01,0.015,0.02",
    },
    "Fees (fee_rate)": {
        "slug": "fee_rate",
        "target": "config",
        "keys": ["fee_rate"],
        "as_int": False,
        "allow_zero": True,
        "default_values": "0.0002,0.0005,0.0008,0.001",
    },
    "Slippage (bps)": {
        "slug": "slippage_bps",
        "target": "config",
        "keys": ["slippage_bps"],
        "as_int": False,
        "allow_zero": True,
        "default_values": "0.5,1,2,3,5",
    },
}


def load_defaults() -> dict[str, Any]:
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_strategy_parameters(widget_prefix: str, defaults: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, value in defaults.items():
        label = key.replace("_", " ").title()
        widget_key = f"{widget_prefix}_{key}"
        if isinstance(value, bool):
            params[key] = st.checkbox(label, value=value, key=widget_key)
        elif isinstance(value, int) and not isinstance(value, bool):
            params[key] = int(st.number_input(label, value=value, step=1, key=widget_key))
        elif isinstance(value, float):
            params[key] = float(
                st.number_input(label, value=float(value), format="%.6f", key=widget_key)
            )
        else:
            params[key] = st.text_input(label, value=str(value), key=widget_key)
    return params


def render_metrics(metrics: dict[str, Any]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return", f"{metrics.get('total_return', 0.0) * 100:.2f}%")
    c2.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0.0) * 100:.2f}%")
    c3.metric("Sharpe (approx)", f"{metrics.get('sharpe', 0.0):.2f}")
    c4.metric("Win Rate", f"{metrics.get('win_rate', 0.0) * 100:.2f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Final Equity", f"{metrics.get('final_equity', 0.0):,.2f}")
    c6.metric("Avg Trade PnL", f"{metrics.get('avg_trade_pnl', 0.0):,.2f}")
    c7.metric("Trades", f"{int(metrics.get('num_trades', 0))}")
    c8.metric("Exposure", f"{metrics.get('exposure', 0.0) * 100:.2f}%")


def render_result(
    title: str,
    metrics: dict[str, Any],
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    config: dict[str, Any],
    timeframe_display: str | None = None,
) -> None:
    st.subheader(title)
    if timeframe_display:
        st.caption(f"Timeframe: {timeframe_display}")
    render_metrics(metrics)

    if equity_curve.empty:
        st.warning("No equity curve data available")
        return

    curve = equity_curve.copy()
    curve["timestamp"] = pd.to_datetime(curve["timestamp"], errors="coerce")
    curve = curve.dropna(subset=["timestamp"])

    st.markdown("### Equity Curve")
    st.line_chart(curve.set_index("timestamp")["equity"], use_container_width=True)

    if "drawdown" in curve.columns:
        st.markdown("### Drawdown")
        st.line_chart(curve.set_index("timestamp")["drawdown"], use_container_width=True)

    st.markdown("### Recent Trades")
    if trades.empty:
        st.info("No closed trades in this run.")
    else:
        display = trades.copy()
        if "entry_time" in display.columns:
            display["entry_time"] = pd.to_datetime(display["entry_time"], errors="coerce")
        if "exit_time" in display.columns:
            display["exit_time"] = pd.to_datetime(display["exit_time"], errors="coerce")
        st.dataframe(display.tail(30), use_container_width=True)

    with st.expander("Run Configuration"):
        st.json(config)


def load_data(csv_path: str) -> pd.DataFrame:
    return load_ohlcv_csv(csv_path)


def list_local_csv_files() -> list[Path]:
    files = sorted(DATA_DIR.glob("*.csv"), key=lambda p: p.name.lower())
    if not files and DEFAULT_DATA_PATH.exists():
        files = [DEFAULT_DATA_PATH]
    return files


def infer_timeframe_label(timestamps: pd.Series) -> str:
    series = pd.to_datetime(timestamps, errors="coerce").dropna().sort_values()
    if len(series) < 2:
        return "unknown"

    deltas_sec = series.diff().dt.total_seconds().dropna()
    if deltas_sec.empty:
        return "unknown"

    median_delta = float(deltas_sec.median())
    known = {
        5 * 60: "5m",
        15 * 60: "15m",
        30 * 60: "30m",
        60 * 60: "1h",
        2 * 60 * 60: "2h",
        4 * 60 * 60: "4h",
        24 * 60 * 60: "1d",
    }
    closest_sec = min(known.keys(), key=lambda x: abs(x - median_delta))
    tolerance = max(60.0, closest_sec * 0.10)
    if abs(closest_sec - median_delta) <= tolerance:
        return known[closest_sec]

    minutes = max(1, int(round(median_delta / 60)))
    if minutes % (24 * 60) == 0:
        return f"{minutes // (24 * 60)}d"
    if minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}m"


def parse_sweep_values(
    raw_values: str,
    *,
    as_int: bool = False,
    allow_zero: bool = False,
) -> list[float]:
    tokens = [token.strip() for token in raw_values.split(",")]
    if not tokens or all(not token for token in tokens):
        raise ValueError("Sweep values are empty. Use comma-separated numbers like: 1.5,2.0,2.5")
    if any(token == "" for token in tokens):
        raise ValueError("Sweep values include an empty item. Remove extra commas.")

    parsed: list[float] = []
    for token in tokens:
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"Invalid number '{token}'. Use comma-separated floats.") from exc
        if as_int:
            if not value.is_integer():
                raise ValueError(f"Value '{token}' must be a whole number.")
            value = float(int(value))
        if allow_zero:
            if value < 0:
                raise ValueError(f"All sweep values must be >= 0. Invalid value: {value}")
        else:
            if value <= 0:
                raise ValueError(f"All sweep values must be > 0. Invalid value: {value}")
        parsed.append(value)

    unique_values: list[float] = []
    for value in parsed:
        if value not in unique_values:
            unique_values.append(value)
    return unique_values


def get_sweep_option(parameter_label: str) -> dict[str, Any]:
    option = SWEEP_OPTIONS.get(parameter_label)
    if option is None:
        raise ValueError(f"Unsupported sweep parameter: {parameter_label}")
    return option


def default_sweep_values(parameter_label: str) -> str:
    return str(get_sweep_option(parameter_label)["default_values"])


def sweep_file_prefix(parameter_label: str) -> str:
    return str(get_sweep_option(parameter_label)["slug"])


def apply_sweep_value(
    *,
    strategy_params: dict[str, Any],
    fee_rate: float,
    slippage_bps: float,
    parameter_label: str,
    sweep_value: float,
) -> tuple[dict[str, Any], float, float]:
    option = get_sweep_option(parameter_label)
    value: float | int = int(sweep_value) if option["as_int"] else float(sweep_value)

    updated_strategy_params = dict(strategy_params)
    updated_fee_rate = float(fee_rate)
    updated_slippage_bps = float(slippage_bps)

    if option["target"] == "strategy":
        for key in option["keys"]:
            if key == "atr_stop_mult":
                if key in updated_strategy_params:
                    updated_strategy_params[key] = value
            else:
                updated_strategy_params[key] = value
    elif option["target"] == "config":
        config_key = option["keys"][0]
        if config_key == "fee_rate":
            updated_fee_rate = float(value)
        elif config_key == "slippage_bps":
            updated_slippage_bps = float(value)
    else:
        raise ValueError(f"Unsupported sweep option target: {option['target']}")

    return updated_strategy_params, updated_fee_rate, updated_slippage_bps


def summarize_exit_reasons(trades: pd.DataFrame) -> dict[str, int]:
    if trades.empty or "reason" not in trades.columns:
        return {"atr_stop_hit": 0, "meanrev_exit": 0, "other_exit": 0}

    reasons = trades["reason"].fillna("").astype(str)
    atr_stop_hit = int((reasons == "atr_stop_hit").sum())
    meanrev_exit = int((reasons == "meanrev_exit").sum())
    other_exit = int(len(reasons) - atr_stop_hit - meanrev_exit)

    return {
        "atr_stop_hit": atr_stop_hit,
        "meanrev_exit": meanrev_exit,
        "other_exit": other_exit,
    }


def build_sweep_csv_path(prefix: str = "k") -> Path:
    SWEEPS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    base = SWEEPS_DIR / f"sweep_{prefix}_{stamp}.csv"
    if not base.exists():
        return base

    for i in range(1, 100):
        candidate = SWEEPS_DIR / f"sweep_{prefix}_{stamp}_{i:02d}.csv"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Unable to allocate unique sweep CSV filename.")


def build_grid_sweep_csv_path(param_a_label: str, param_b_label: str) -> Path:
    SWEEPS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    slug_a = sweep_file_prefix(param_a_label)
    slug_b = sweep_file_prefix(param_b_label)
    base = SWEEPS_DIR / f"grid_{slug_a}_{slug_b}_{stamp}.csv"
    if not base.exists():
        return base

    for i in range(1, 100):
        candidate = SWEEPS_DIR / f"grid_{slug_a}_{slug_b}_{stamp}_{i:02d}.csv"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Unable to allocate unique grid sweep CSV filename.")


def main() -> None:
    st.set_page_config(page_title="Perps Strategy Lab", layout="wide")
    st.title("Perps Strategy Lab")
    st.caption("MVP core backtester for BTC perpetual on 4h candles")

    defaults = load_defaults()
    storage = SQLiteStorage(DEFAULT_DB_PATH)

    strategy_names = discover_strategy_names()
    if not strategy_names:
        st.error("No strategies found in /strategies")
        return

    default_strategy = defaults.get("strategy", {}).get("name", strategy_names[0])
    if default_strategy not in strategy_names:
        default_strategy = strategy_names[0]

    with st.sidebar:
        st.header("Backtest Inputs")

        symbol = st.text_input("Symbol", value=defaults.get("symbol", "BTC-PERP"))
        timeframe = "4h"
        initial_cash = float(
            st.number_input(
                "Initial Equity",
                min_value=100.0,
                value=float(defaults.get("initial_cash", 10_000.0)),
                step=100.0,
            )
        )

        fee_rate = float(
            st.number_input(
                "Fee Rate",
                min_value=0.0,
                value=float(defaults.get("fees", {}).get("fee_rate", 0.0005)),
                format="%.6f",
            )
        )
        slippage_bps = float(
            st.number_input(
                "Slippage (bps)",
                min_value=0.0,
                value=float(defaults.get("fees", {}).get("slippage_bps", 2.0)),
                format="%.4f",
            )
        )

        max_leverage = float(
            st.number_input(
                "Max Leverage",
                min_value=1.0,
                value=float(defaults.get("risk", {}).get("max_leverage", 3.0)),
                format="%.2f",
            )
        )
        max_notional_input = float(
            st.number_input(
                "Max Notional (0 = disabled)",
                min_value=0.0,
                value=float(defaults.get("risk", {}).get("max_notional", 0.0) or 0.0),
                step=100.0,
            )
        )

        st.markdown("### Data Download")
        st.caption(f"Data directory: {DATA_DIR.resolve()}")
        if st.button("Download BTC 4h (3 years) from Binance"):
            with st.spinner("Downloading BTCUSDT 4h candles from Binance..."):
                try:
                    summary = download_btcusdt_4h_last_3y(out_path=DOWNLOADED_DATA_PATH)
                    st.session_state["local_csv_path"] = str(summary.saved_path)
                    st.success(
                        "Download complete: "
                        f"{summary.rows} rows ({summary.first_timestamp} -> {summary.last_timestamp})."
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Download failed: {exc}")

        if st.button("Download BTC 1h (2019-2025) from Binance"):
            with st.spinner("Downloading BTCUSDT 1h candles from Binance..."):
                try:
                    summary = download_binance_klines(
                        symbol="BTCUSDT",
                        interval="1h",
                        start_date="2019-01-01",
                        end_date="2025-12-31",
                        out_path=DOWNLOADED_1H_2019_2025_PATH,
                    )
                    st.session_state["local_csv_path"] = str(summary.saved_path)
                    st.success(
                        "Download complete: "
                        f"{summary.rows} rows ({summary.first_timestamp} -> {summary.last_timestamp})."
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Download failed: {exc}")

        st.markdown("### Local CSV File")
        local_csv_files = list_local_csv_files()
        csv_path = str(DEFAULT_DATA_PATH)
        if not local_csv_files:
            st.error(f"No CSV files found in: {DATA_DIR.resolve()}")
        else:
            local_csv_options = [str(path) for path in local_csv_files]
            preferred_csv = st.session_state.get("local_csv_path")
            if preferred_csv not in local_csv_options:
                preferred_csv = (
                    str(DOWNLOADED_1H_2019_2025_PATH)
                    if str(DOWNLOADED_1H_2019_2025_PATH) in local_csv_options
                    else local_csv_options[0]
                )
            default_index = local_csv_options.index(preferred_csv)
            csv_path = st.selectbox(
                "Choose dataset",
                options=local_csv_options,
                index=default_index,
                format_func=lambda p: Path(p).name,
            )
            st.session_state["local_csv_path"] = csv_path

        strategy_name = st.selectbox(
            "Strategy",
            options=strategy_names,
            index=strategy_names.index(default_strategy),
        )

        param_defaults = get_default_parameters(strategy_name)
        yaml_param_overrides = defaults.get("strategy", {}).get("params", {})
        merged_defaults = {**param_defaults, **yaml_param_overrides}

        st.markdown("### Strategy Parameters")
        strategy_params = render_strategy_parameters(
            widget_prefix=f"params_{strategy_name}",
            defaults=merged_defaults,
        )

        run_name = st.text_input("Run Name (optional)", value="")

        st.markdown("### Parameter Sweep")
        sweep_mode = st.radio(
            "Sweep mode",
            options=["Single Parameter Sweep", "Two-Parameter Grid Sweep"],
            index=0,
        )
        sweep_options = list(SWEEP_OPTIONS.keys())

        single_sweep_parameter = ""
        single_sweep_values_raw = ""
        grid_param_a = ""
        grid_values_a_raw = ""
        grid_param_b = ""
        grid_values_b_raw = ""
        grid_allow_over_limit = False

        if sweep_mode == "Single Parameter Sweep":
            single_sweep_parameter = st.selectbox(
                "Sweep parameter",
                options=sweep_options,
                index=0,
                key="single_sweep_parameter",
            )
            if "single_sweep_values_input" not in st.session_state:
                st.session_state["single_sweep_values_input"] = default_sweep_values(single_sweep_parameter)
            if st.session_state.get("single_sweep_values_parameter") != single_sweep_parameter:
                st.session_state["single_sweep_values_input"] = default_sweep_values(single_sweep_parameter)
                st.session_state["single_sweep_values_parameter"] = single_sweep_parameter
            single_sweep_values_raw = st.text_input(
                "Sweep values (comma-separated)",
                key="single_sweep_values_input",
            )
        else:
            grid_param_a = st.selectbox(
                "Param A",
                options=sweep_options,
                index=sweep_options.index("Regime ATR % Threshold")
                if "Regime ATR % Threshold" in sweep_options
                else 0,
                key="grid_param_a",
            )
            grid_param_b_options = [option for option in sweep_options if option != grid_param_a]
            default_b = "RSI Entry Threshold" if "RSI Entry Threshold" in grid_param_b_options else grid_param_b_options[0]
            grid_param_b = st.selectbox(
                "Param B",
                options=grid_param_b_options,
                index=grid_param_b_options.index(default_b),
                key="grid_param_b",
            )

            if "grid_values_a_input" not in st.session_state:
                st.session_state["grid_values_a_input"] = default_sweep_values(grid_param_a)
            if st.session_state.get("grid_values_a_parameter") != grid_param_a:
                st.session_state["grid_values_a_input"] = default_sweep_values(grid_param_a)
                st.session_state["grid_values_a_parameter"] = grid_param_a

            if "grid_values_b_input" not in st.session_state:
                st.session_state["grid_values_b_input"] = default_sweep_values(grid_param_b)
            if st.session_state.get("grid_values_b_parameter") != grid_param_b:
                st.session_state["grid_values_b_input"] = default_sweep_values(grid_param_b)
                st.session_state["grid_values_b_parameter"] = grid_param_b

            grid_values_a_raw = st.text_input(
                "Values A (comma-separated)",
                key="grid_values_a_input",
            )
            grid_values_b_raw = st.text_input(
                "Values B (comma-separated)",
                key="grid_values_b_input",
            )

            preview_message = "Grid combinations: invalid values"
            try:
                parsed_a = parse_sweep_values(
                    grid_values_a_raw,
                    as_int=bool(get_sweep_option(grid_param_a)["as_int"]),
                    allow_zero=bool(get_sweep_option(grid_param_a)["allow_zero"]),
                )
                parsed_b = parse_sweep_values(
                    grid_values_b_raw,
                    as_int=bool(get_sweep_option(grid_param_b)["as_int"]),
                    allow_zero=bool(get_sweep_option(grid_param_b)["allow_zero"]),
                )
                preview_message = f"Grid combinations: {len(parsed_a) * len(parsed_b)}"
            except ValueError:
                pass
            st.caption(preview_message)

            grid_allow_over_limit = st.checkbox(
                "I understand (allow > 60 combinations)",
                value=False,
                key="grid_allow_over_limit",
            )

    try:
        data = load_data(csv_path=csv_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load data: {exc}")
        return

    min_date = data["timestamp"].min().date()
    max_date = data["timestamp"].max().date()

    with st.sidebar:
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        run_clicked = st.button("Run Backtest", type="primary")
        run_sweep_clicked = sweep_mode == "Single Parameter Sweep" and st.button("Run Sweep")
        run_grid_sweep_clicked = sweep_mode == "Two-Parameter Grid Sweep" and st.button("Run Grid Sweep")

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    filtered_data = filter_date_range(data, start=start_date, end=end_date)
    inferred_timeframe = infer_timeframe_label(filtered_data["timestamp"])
    timeframe_display = f"{inferred_timeframe} (inferred)"

    st.markdown(
        f"Loaded **{len(filtered_data)}** bars from `{pd.Timestamp(filtered_data['timestamp'].min())}` to `{pd.Timestamp(filtered_data['timestamp'].max())}`"
    )
    st.caption(f"Selected file: `{Path(csv_path).name}`")
    st.caption(f"Timeframe: {timeframe_display}")

    if run_clicked:
        if filtered_data.empty:
            st.error("No bars in selected date range")
        else:
            config = RunConfig(
                symbol=symbol,
                timeframe=timeframe,
                strategy_name=strategy_name,
                strategy_params=strategy_params,
                initial_cash=initial_cash,
                fee_rate=fee_rate,
                slippage_bps=slippage_bps,
                max_leverage=max_leverage,
                max_notional=max_notional_input if max_notional_input > 0 else None,
                long_only=True,
                run_name=run_name,
            )
            strategy = load_strategy(strategy_name, strategy_params)
            engine = BacktestEngine(config=config)
            result = engine.run(data=filtered_data, strategy=strategy)
            run_id = storage.save_run(result)
            config_display = asdict(config)
            config_display["timeframe"] = timeframe_display

            st.session_state["active_view"] = {
                "source": "new",
                "run_id": run_id,
                "metrics": result.metrics,
                "equity_curve": result.equity_curve,
                "trades": result.trades,
                "config": config_display,
                "timeframe_display": timeframe_display,
            }
            st.success(f"Backtest complete. Saved as run #{run_id}.")

    if run_sweep_clicked:
        if filtered_data.empty:
            st.error("No bars in selected date range")
        else:
            try:
                sweep_option = get_sweep_option(single_sweep_parameter)
                sweep_values = parse_sweep_values(
                    single_sweep_values_raw,
                    as_int=bool(sweep_option["as_int"]),
                    allow_zero=bool(sweep_option["allow_zero"]),
                )
            except ValueError as exc:
                st.error(f"Sweep input error: {exc}")
            else:
                progress = st.progress(0.0, text="Starting parameter sweep...")
                sweep_rows: list[dict[str, Any]] = []
                total = len(sweep_values)

                for idx, sweep_value in enumerate(sweep_values, start=1):
                    run_params, run_fee_rate, run_slippage_bps = apply_sweep_value(
                        strategy_params=strategy_params,
                        fee_rate=fee_rate,
                        slippage_bps=slippage_bps,
                        parameter_label=single_sweep_parameter,
                        sweep_value=sweep_value,
                    )
                    run_label = run_name.strip() or "sweep"
                    run_config = RunConfig(
                        symbol=symbol,
                        timeframe=timeframe,
                        strategy_name=strategy_name,
                        strategy_params=run_params,
                        initial_cash=initial_cash,
                        fee_rate=run_fee_rate,
                        slippage_bps=run_slippage_bps,
                        max_leverage=max_leverage,
                        max_notional=max_notional_input if max_notional_input > 0 else None,
                        long_only=True,
                        run_name=f"{run_label}_{sweep_value:g}",
                    )
                    strategy = load_strategy(strategy_name, run_params)
                    engine = BacktestEngine(config=run_config)
                    result = engine.run(data=filtered_data, strategy=strategy)
                    run_id = storage.save_run(result)

                    exits = summarize_exit_reasons(result.trades)
                    row: dict[str, Any] = {
                        "run_id": run_id,
                        "parameter": single_sweep_parameter,
                        "sweep_value": sweep_value,
                        "total_return": result.metrics.get("total_return", 0.0),
                        "max_drawdown": result.metrics.get("max_drawdown", 0.0),
                        "sharpe_approx": result.metrics.get("sharpe", 0.0),
                        "win_rate": result.metrics.get("win_rate", 0.0),
                        "trades": int(result.metrics.get("num_trades", 0)),
                        "exposure": result.metrics.get("exposure", 0.0),
                        **exits,
                    }
                    if single_sweep_parameter == "ATR Stop Multiplier (k)":
                        row["k"] = sweep_value
                    elif single_sweep_parameter == "RSI Entry Threshold":
                        row["rsi_entry"] = sweep_value
                    sweep_rows.append(
                        row
                    )

                    progress.progress(
                        idx / total,
                        text=f"Sweep progress: {idx}/{total} runs completed",
                    )

                progress.empty()

                sweep_df = pd.DataFrame(sweep_rows).sort_values(
                    by=["sharpe_approx", "total_return"],
                    ascending=[False, False],
                )
                sweep_df = sweep_df.reset_index(drop=True)

                sweep_csv_path = build_sweep_csv_path(prefix=sweep_file_prefix(single_sweep_parameter))
                sweep_df.to_csv(sweep_csv_path, index=False)

                st.session_state["sweep_results"] = {
                    "table": sweep_df,
                    "path": str(sweep_csv_path),
                }
                st.success(
                    f"Sweep complete: {len(sweep_df)} runs. "
                    f"Saved CSV: {sweep_csv_path}"
                )

    if run_grid_sweep_clicked:
        if filtered_data.empty:
            st.error("No bars in selected date range")
        elif grid_param_a == grid_param_b:
            st.error("Grid parameters must be different.")
        else:
            try:
                option_a = get_sweep_option(grid_param_a)
                option_b = get_sweep_option(grid_param_b)
                values_a = parse_sweep_values(
                    grid_values_a_raw,
                    as_int=bool(option_a["as_int"]),
                    allow_zero=bool(option_a["allow_zero"]),
                )
                values_b = parse_sweep_values(
                    grid_values_b_raw,
                    as_int=bool(option_b["as_int"]),
                    allow_zero=bool(option_b["allow_zero"]),
                )
            except ValueError as exc:
                st.error(f"Grid sweep input error: {exc}")
            else:
                combinations = len(values_a) * len(values_b)
                if combinations > 60 and not grid_allow_over_limit:
                    st.error(
                        f"Grid sweep has {combinations} combinations (> 60). "
                        "Enable 'I understand' to proceed."
                    )
                else:
                    progress = st.progress(0.0, text="Starting grid sweep...")
                    rows: list[dict[str, Any]] = []
                    total = combinations
                    completed = 0

                    for value_a in values_a:
                        for value_b in values_b:
                            run_params = dict(strategy_params)
                            run_fee_rate = fee_rate
                            run_slippage_bps = slippage_bps

                            run_params, run_fee_rate, run_slippage_bps = apply_sweep_value(
                                strategy_params=run_params,
                                fee_rate=run_fee_rate,
                                slippage_bps=run_slippage_bps,
                                parameter_label=grid_param_a,
                                sweep_value=value_a,
                            )
                            run_params, run_fee_rate, run_slippage_bps = apply_sweep_value(
                                strategy_params=run_params,
                                fee_rate=run_fee_rate,
                                slippage_bps=run_slippage_bps,
                                parameter_label=grid_param_b,
                                sweep_value=value_b,
                            )

                            run_label = run_name.strip() or "grid"
                            run_config = RunConfig(
                                symbol=symbol,
                                timeframe=timeframe,
                                strategy_name=strategy_name,
                                strategy_params=run_params,
                                initial_cash=initial_cash,
                                fee_rate=run_fee_rate,
                                slippage_bps=run_slippage_bps,
                                max_leverage=max_leverage,
                                max_notional=max_notional_input if max_notional_input > 0 else None,
                                long_only=True,
                                run_name=(
                                    f"{run_label}_"
                                    f"{sweep_file_prefix(grid_param_a)}{value_a:g}_"
                                    f"{sweep_file_prefix(grid_param_b)}{value_b:g}"
                                ),
                            )
                            strategy = load_strategy(strategy_name, run_params)
                            engine = BacktestEngine(config=run_config)
                            result = engine.run(data=filtered_data, strategy=strategy)
                            run_id = storage.save_run(result)

                            exits = summarize_exit_reasons(result.trades)
                            rows.append(
                                {
                                    "run_id": run_id,
                                    "param_a_name": grid_param_a,
                                    "param_a_value": value_a,
                                    "param_b_name": grid_param_b,
                                    "param_b_value": value_b,
                                    "total_return": result.metrics.get("total_return", 0.0),
                                    "max_drawdown": result.metrics.get("max_drawdown", 0.0),
                                    "sharpe_approx": result.metrics.get("sharpe", 0.0),
                                    "win_rate": result.metrics.get("win_rate", 0.0),
                                    "trades": int(result.metrics.get("num_trades", 0)),
                                    "exposure": result.metrics.get("exposure", 0.0),
                                    "atr_stop_hit": exits["atr_stop_hit"],
                                    "meanrev_exit": exits["meanrev_exit"],
                                }
                            )

                            completed += 1
                            progress.progress(
                                completed / total,
                                text=f"Grid sweep progress: {completed}/{total} runs completed",
                            )

                    progress.empty()
                    grid_df = pd.DataFrame(rows).sort_values(
                        by=["sharpe_approx", "total_return"],
                        ascending=[False, False],
                    )
                    grid_df = grid_df.reset_index(drop=True)
                    grid_csv_path = build_grid_sweep_csv_path(
                        param_a_label=grid_param_a,
                        param_b_label=grid_param_b,
                    )
                    grid_df.to_csv(grid_csv_path, index=False)

                    st.session_state["sweep_results"] = {
                        "table": grid_df,
                        "path": str(grid_csv_path),
                    }
                    st.success(
                        f"Grid sweep complete: {len(grid_df)} runs. "
                        f"Saved CSV: {grid_csv_path}"
                    )

    sweep_results = st.session_state.get("sweep_results")
    if sweep_results:
        st.markdown("## Parameter Sweep Results")
        st.caption(
            "Sorted best-first by Sharpe (approx), then Total Return. "
            f"Saved CSV: `{sweep_results['path']}`"
        )
        st.dataframe(sweep_results["table"], use_container_width=True)

    st.markdown("## Saved Runs")
    runs = storage.list_runs(limit=100)
    if runs.empty:
        st.info("No saved runs yet.")
    else:
        runs_display = runs.copy()
        if "timeframe" in runs_display.columns:
            runs_display["timeframe"] = timeframe_display
        st.dataframe(runs_display, use_container_width=True)
        selected_run_id = st.selectbox(
            "Open run",
            options=runs["id"].astype(int).tolist(),
            key="open_run_id",
        )
        if st.button("Open Selected Run"):
            loaded = storage.load_run(int(selected_run_id))
            loaded_timeframe_display = "unknown (inferred)"
            if not loaded["equity_curve"].empty and "timestamp" in loaded["equity_curve"].columns:
                loaded_timeframe_display = (
                    f"{infer_timeframe_label(loaded['equity_curve']['timestamp'])} (inferred)"
                )
            loaded_config = dict(loaded["config"])
            loaded_config["timeframe"] = loaded_timeframe_display
            st.session_state["active_view"] = {
                "source": "saved",
                "run_id": int(selected_run_id),
                "metrics": loaded["metrics"],
                "equity_curve": loaded["equity_curve"],
                "trades": loaded["trades"],
                "config": loaded_config,
                "timeframe_display": loaded_timeframe_display,
            }

    active_view = st.session_state.get("active_view")
    if active_view:
        render_result(
            title=f"Run #{active_view['run_id']} ({active_view['source']})",
            metrics=active_view["metrics"],
            equity_curve=active_view["equity_curve"],
            trades=active_view["trades"],
            config=active_view["config"],
            timeframe_display=active_view.get("timeframe_display"),
        )


if __name__ == "__main__":
    main()
