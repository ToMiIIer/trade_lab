from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

from core.data_io import filter_date_range, load_ohlcv_csv, prepare_ohlcv_dataframe
from core.engine import BacktestEngine
from core.storage import SQLiteStorage
from core.types import RunConfig
from strategies import discover_strategy_names, get_default_parameters, load_strategy
from tools.download_binance_klines import download_btcusdt_4h_last_3y

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs/default.yaml"
DEFAULT_DATA_PATH = ROOT_DIR / "data/sample_btc_4h.csv"
DOWNLOADED_DATA_PATH = ROOT_DIR / "data/btcusdt_4h_3y.csv"
DEFAULT_DB_PATH = ROOT_DIR / "runs.sqlite3"


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
) -> None:
    st.subheader(title)
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


def load_data(csv_mode: str, csv_path: str, upload) -> pd.DataFrame:
    if csv_mode == "Upload CSV" and upload is not None:
        uploaded_df = pd.read_csv(upload)
        return prepare_ohlcv_dataframe(uploaded_df)
    return load_ohlcv_csv(csv_path)


def list_local_csv_files() -> list[str]:
    data_dir = ROOT_DIR / "data"
    files = sorted(str(path) for path in data_dir.glob("*.csv"))
    if str(DEFAULT_DATA_PATH) not in files:
        files.insert(0, str(DEFAULT_DATA_PATH))
    return files


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
        timeframe = st.selectbox("Timeframe", options=["4h"], index=0)
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

        st.markdown("### Data")
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

        csv_mode = st.radio("CSV Input", options=["Local Path", "Upload CSV"], index=0)
        csv_path = str(DEFAULT_DATA_PATH)
        upload = None
        if csv_mode == "Local Path":
            local_csv_options = list_local_csv_files()
            preferred_csv = st.session_state.get("local_csv_path", str(DEFAULT_DATA_PATH))
            if preferred_csv in local_csv_options:
                default_index = local_csv_options.index(preferred_csv)
            else:
                default_index = 0
            csv_path = st.selectbox(
                "Local CSV File",
                options=local_csv_options,
                index=default_index,
            )
            st.session_state["local_csv_path"] = csv_path
        else:
            upload = st.file_uploader("Upload OHLCV CSV", type=["csv"])

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

    try:
        data = load_data(csv_mode=csv_mode, csv_path=csv_path, upload=upload)
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

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    filtered_data = filter_date_range(data, start=start_date, end=end_date)

    st.markdown(
        f"Loaded **{len(filtered_data)}** bars from `{pd.Timestamp(filtered_data['timestamp'].min())}` to `{pd.Timestamp(filtered_data['timestamp'].max())}`"
    )

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

            st.session_state["active_view"] = {
                "source": "new",
                "run_id": run_id,
                "metrics": result.metrics,
                "equity_curve": result.equity_curve,
                "trades": result.trades,
                "config": asdict(config),
            }
            st.success(f"Backtest complete. Saved as run #{run_id}.")

    st.markdown("## Saved Runs")
    runs = storage.list_runs(limit=100)
    if runs.empty:
        st.info("No saved runs yet.")
    else:
        st.dataframe(runs, use_container_width=True)
        selected_run_id = st.selectbox(
            "Open run",
            options=runs["id"].astype(int).tolist(),
            key="open_run_id",
        )
        if st.button("Open Selected Run"):
            loaded = storage.load_run(int(selected_run_id))
            st.session_state["active_view"] = {
                "source": "saved",
                "run_id": int(selected_run_id),
                "metrics": loaded["metrics"],
                "equity_curve": loaded["equity_curve"],
                "trades": loaded["trades"],
                "config": loaded["config"],
            }

    active_view = st.session_state.get("active_view")
    if active_view:
        render_result(
            title=f"Run #{active_view['run_id']} ({active_view['source']})",
            metrics=active_view["metrics"],
            equity_curve=active_view["equity_curve"],
            trades=active_view["trades"],
            config=active_view["config"],
        )


if __name__ == "__main__":
    main()
