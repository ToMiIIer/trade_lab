from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def prepare_ohlcv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("OHLCV input is empty")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    output = df[REQUIRED_COLUMNS].copy()

    timestamps = pd.to_datetime(output["timestamp"], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError("Invalid timestamps found in CSV")
    output["timestamp"] = timestamps.dt.tz_convert(None)

    for col in ["open", "high", "low", "close", "volume"]:
        output[col] = pd.to_numeric(output[col], errors="coerce")
        if output[col].isna().any():
            raise ValueError(f"Column '{col}' has non-numeric values")

    output = output.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return output


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    raw_df = pd.read_csv(csv_path)
    return prepare_ohlcv_dataframe(raw_df)


def filter_date_range(
    df: pd.DataFrame,
    start: pd.Timestamp | str | None,
    end: pd.Timestamp | str | None,
) -> pd.DataFrame:
    output = df.copy()

    if start is not None:
        start_ts = pd.Timestamp(start)
        output = output[output["timestamp"] >= start_ts]

    if end is not None:
        end_ts = pd.Timestamp(end)
        if end_ts == end_ts.normalize():
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        output = output[output["timestamp"] <= end_ts]

    return output.reset_index(drop=True)
