from __future__ import annotations

import argparse
import csv
import json
import os
import ssl
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_URL = "https://api.binance.com/api/v3/klines"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "4h"
DEFAULT_YEARS = 3
LIMIT = 1000
REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 1.0
DEFAULT_REQUEST_GAP_SECONDS = 0.25

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT_PATH = ROOT_DIR / "data" / "btcusdt_4h_3y.csv"


@dataclass(slots=True)
class DownloadSummary:
    saved_path: Path
    rows: int
    first_timestamp: str
    last_timestamp: str


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _build_ssl_context() -> ssl.SSLContext:
    """Use certifi bundle when available, otherwise default system trust store."""
    try:
        import certifi
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _parse_yyyy_mm_dd(date_str: str) -> datetime:
    try:
        parsed = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"Invalid date '{date_str}'. Expected YYYY-MM-DD format."
        ) from exc
    return parsed.replace(tzinfo=timezone.utc)


def _resolve_time_window(
    start_date: str | None,
    end_date: str | None,
    default_years: int,
) -> tuple[datetime, datetime]:
    if start_date is None and end_date is None:
        end_dt = _utc_now()
        start_dt = end_dt - timedelta(days=default_years * 365)
        return start_dt, end_dt

    if start_date is None or end_date is None:
        raise ValueError("Both start_date and end_date must be provided together.")

    start_dt = _parse_yyyy_mm_dd(start_date)
    # Include entire end day in UTC.
    end_dt = _parse_yyyy_mm_dd(end_date) + timedelta(days=1) - timedelta(milliseconds=1)

    if end_dt <= start_dt:
        raise ValueError("end_date must be later than start_date.")

    return start_dt, end_dt


def _fetch_klines(
    *,
    symbol: str,
    interval: str,
    start_time_ms: int,
    end_time_ms: int,
    retries: int,
    backoff_seconds: float,
    ssl_context: ssl.SSLContext,
) -> list[list[Any]]:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time_ms,
        "endTime": end_time_ms,
        "limit": LIMIT,
    }
    url = f"{API_URL}?{urlencode(params)}"

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "trade-lab/1.0"})
            with urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS, context=ssl_context) as resp:
                raw = resp.read().decode("utf-8")
            payload = json.loads(raw)
            if not isinstance(payload, list):
                raise RuntimeError(f"Unexpected Binance response: {payload}")
            return payload
        except (HTTPError, URLError, TimeoutError, ssl.SSLError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(backoff_seconds * (2**attempt))

    raise RuntimeError(f"Failed fetching Binance klines after {retries + 1} attempts: {last_error}")


def _write_rows_atomic(out_path: Path, rows: list[list[Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        delete=False,
        dir=out_path.parent,
        prefix=f".{out_path.stem}_",
        suffix=".tmp",
    )

    tmp_path = Path(tmp_file.name)
    try:
        with tmp_file:
            writer = csv.writer(tmp_file)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            writer.writerows(rows)
        os.replace(tmp_path, out_path)
    except Exception:  # noqa: BLE001
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

def download_binance_klines(
    *,
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    start_date: str | None = None,
    end_date: str | None = None,
    out_path: str | Path = DEFAULT_OUT_PATH,
    retries: int = DEFAULT_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    request_gap_seconds: float = DEFAULT_REQUEST_GAP_SECONDS,
) -> DownloadSummary:
    if not symbol:
        raise ValueError("symbol cannot be empty")
    if not interval:
        raise ValueError("interval cannot be empty")

    output_path = Path(out_path)
    start_dt, end_dt = _resolve_time_window(
        start_date=start_date,
        end_date=end_date,
        default_years=DEFAULT_YEARS,
    )

    start_ms = _to_ms(start_dt)
    end_ms = _to_ms(end_dt)

    rows: list[list[Any]] = []
    cursor = start_ms
    last_open_time: int | None = None
    ssl_context = _build_ssl_context()

    while True:
        batch = _fetch_klines(
            symbol=symbol,
            interval=interval,
            start_time_ms=cursor,
            end_time_ms=end_ms,
            retries=retries,
            backoff_seconds=backoff_seconds,
            ssl_context=ssl_context,
        )
        if not batch:
            break

        for k in batch:
            open_time = int(k[0])
            if open_time < start_ms or open_time > end_ms:
                continue
            if last_open_time is not None and open_time <= last_open_time:
                continue

            ts = datetime.fromtimestamp(open_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            rows.append(
                [
                    ts,
                    float(k[1]),
                    float(k[2]),
                    float(k[3]),
                    float(k[4]),
                    float(k[5]),
                ]
            )
            last_open_time = open_time

        next_cursor = int(batch[-1][0]) + 1
        if next_cursor <= cursor or next_cursor > end_ms or len(batch) < LIMIT:
            break

        cursor = next_cursor
        time.sleep(request_gap_seconds)

    if not rows:
        raise RuntimeError("No rows downloaded from Binance. Check network or API availability.")

    _write_rows_atomic(out_path=output_path, rows=rows)

    return DownloadSummary(
        saved_path=output_path,
        rows=len(rows),
        first_timestamp=rows[0][0],
        last_timestamp=rows[-1][0],
    )


def download_btcusdt_4h_last_3y(
    out_path: Path = DEFAULT_OUT_PATH,
    retries: int = DEFAULT_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    request_gap_seconds: float = DEFAULT_REQUEST_GAP_SECONDS,
) -> DownloadSummary:
    return download_binance_klines(
        symbol=DEFAULT_SYMBOL,
        interval=DEFAULT_INTERVAL,
        out_path=out_path,
        retries=retries,
        backoff_seconds=backoff_seconds,
        request_gap_seconds=request_gap_seconds,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Binance Spot OHLCV klines to CSV")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Trading symbol, e.g. BTCUSDT")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="Kline interval, e.g. 1h, 4h, 1d")
    parser.add_argument(
        "--start",
        dest="start_date",
        default=None,
        help="Start date (YYYY-MM-DD). Must be used together with --end.",
    )
    parser.add_argument(
        "--end",
        dest="end_date",
        default=None,
        help="End date (YYYY-MM-DD). Must be used together with --start.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT_PATH),
        help="Output CSV path. Default: trade_lab/data/btcusdt_4h_3y.csv",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if bool(args.start_date) != bool(args.end_date):
        parser.error("--start and --end must be provided together.")

    if args.start_date and args.end_date:
        range_text = f"from {args.start_date} to {args.end_date} (UTC)"
    else:
        range_text = f"last {DEFAULT_YEARS} years (UTC)"

    print(
        f"Downloading {args.symbol} {args.interval} {range_text} from Binance...",
        flush=True,
    )

    try:
        summary = download_binance_klines(
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start_date,
            end_date=args.end_date,
            out_path=args.out,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Download failed: {exc}") from exc

    print(
        "Saved: "
        f"{summary.saved_path}  "
        f"rows={summary.rows}  "
        f"first={summary.first_timestamp}  "
        f"last={summary.last_timestamp}"
    )


if __name__ == "__main__":
    main()
    
def download_btcusdt_4h_last_3y(*args, **kwargs):
    """
    Backwards-compatible wrapper for the original UI button.
    Calls download_binance_klines with BTCUSDT 4h and a 3-year lookback
    if your implementation supports date ranges; otherwise uses existing defaults.
    """
    return download_binance_klines(*args, **kwargs)
