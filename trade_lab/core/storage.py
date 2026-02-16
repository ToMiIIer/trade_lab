from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from core.types import RunResult


class SQLiteStorage:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    run_name TEXT,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    start_ts TEXT,
                    end_ts TEXT,
                    num_trades INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    entry_time TEXT,
                    exit_time TEXT,
                    side TEXT,
                    qty REAL,
                    entry_price REAL,
                    exit_price REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    bars_held INTEGER,
                    reason TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                );

                CREATE TABLE IF NOT EXISTS equity_curve (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    equity REAL NOT NULL,
                    cash REAL NOT NULL,
                    market_value REAL NOT NULL,
                    unrealized REAL NOT NULL,
                    position_qty REAL NOT NULL,
                    close REAL NOT NULL,
                    drawdown REAL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                );
                """
            )

    def save_run(self, result: RunResult) -> int:
        if result.equity_curve.empty:
            raise ValueError("Cannot save run with empty equity curve")

        created_at = pd.Timestamp.utcnow().isoformat()
        config_json = json.dumps(asdict(result.config), default=str)
        metrics_json = json.dumps(result.metrics)
        num_trades = int(len(result.trades))
        start_ts = pd.Timestamp(result.equity_curve["timestamp"].iloc[0]).isoformat()
        end_ts = pd.Timestamp(result.equity_curve["timestamp"].iloc[-1]).isoformat()

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO runs (
                    created_at, run_name, strategy_name, symbol, timeframe,
                    config_json, metrics_json, start_ts, end_ts, num_trades
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    result.config.run_name,
                    result.config.strategy_name,
                    result.config.symbol,
                    result.config.timeframe,
                    config_json,
                    metrics_json,
                    start_ts,
                    end_ts,
                    num_trades,
                ),
            )
            run_id = int(cursor.lastrowid)

            if not result.trades.empty:
                trade_rows = [
                    (
                        run_id,
                        _to_iso(row.get("entry_time")),
                        _to_iso(row.get("exit_time")),
                        str(row.get("side", "")),
                        float(row.get("qty", 0.0)),
                        float(row.get("entry_price", 0.0)),
                        float(row.get("exit_price", 0.0)),
                        float(row.get("pnl", 0.0)),
                        float(row.get("pnl_pct", 0.0)),
                        int(row.get("bars_held", 0)),
                        str(row.get("reason", "")),
                    )
                    for _, row in result.trades.iterrows()
                ]
                cursor.executemany(
                    """
                    INSERT INTO trades (
                        run_id, entry_time, exit_time, side, qty,
                        entry_price, exit_price, pnl, pnl_pct, bars_held, reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    trade_rows,
                )

            equity_rows = [
                (
                    run_id,
                    _to_iso(row["timestamp"]),
                    float(row["equity"]),
                    float(row["cash"]),
                    float(row["market_value"]),
                    float(row["unrealized"]),
                    float(row["position_qty"]),
                    float(row["close"]),
                    float(row.get("drawdown", 0.0)),
                )
                for _, row in result.equity_curve.iterrows()
            ]
            cursor.executemany(
                """
                INSERT INTO equity_curve (
                    run_id, timestamp, equity, cash, market_value,
                    unrealized, position_qty, close, drawdown
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                equity_rows,
            )

        return run_id

    def list_runs(self, limit: int = 50) -> pd.DataFrame:
        with self._connect() as conn:
            df = pd.read_sql_query(
                """
                SELECT id, created_at, run_name, strategy_name, symbol, timeframe, num_trades, metrics_json
                FROM runs
                ORDER BY id DESC
                LIMIT ?
                """,
                conn,
                params=(limit,),
            )

        if df.empty:
            return df

        totals: list[dict[str, Any]] = []
        for metrics_str in df["metrics_json"]:
            metrics = json.loads(metrics_str)
            totals.append(
                {
                    "total_return": metrics.get("total_return", 0.0),
                    "max_drawdown": metrics.get("max_drawdown", 0.0),
                    "final_equity": metrics.get("final_equity", 0.0),
                }
            )

        totals_df = pd.DataFrame(totals)
        merged = pd.concat([df.drop(columns=["metrics_json"]), totals_df], axis=1)
        return merged

    def load_run(self, run_id: int) -> dict[str, Any]:
        with self._connect() as conn:
            run_df = pd.read_sql_query("SELECT * FROM runs WHERE id = ?", conn, params=(run_id,))
            if run_df.empty:
                raise ValueError(f"Run id {run_id} not found")
            run_row = run_df.iloc[0].to_dict()

            trades = pd.read_sql_query(
                """
                SELECT entry_time, exit_time, side, qty, entry_price, exit_price, pnl, pnl_pct, bars_held, reason
                FROM trades
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                conn,
                params=(run_id,),
            )
            equity = pd.read_sql_query(
                """
                SELECT timestamp, equity, cash, market_value, unrealized, position_qty, close, drawdown
                FROM equity_curve
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                conn,
                params=(run_id,),
            )

        for col in ["entry_time", "exit_time"]:
            if col in trades.columns and not trades.empty:
                trades[col] = pd.to_datetime(trades[col], errors="coerce")
        if not equity.empty:
            equity["timestamp"] = pd.to_datetime(equity["timestamp"], errors="coerce")

        return {
            "run": run_row,
            "config": json.loads(run_row["config_json"]),
            "metrics": json.loads(run_row["metrics_json"]),
            "trades": trades,
            "equity_curve": equity,
        }


def _to_iso(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return pd.Timestamp(value).isoformat()
