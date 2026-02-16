from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "4h"
YEARS = 3
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


def _fetch_klines(start_time_ms: int, end_time_ms: int, retries: int, backoff_seconds: float) -> list[list[Any]]:
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "startTime": start_time_ms,
        "endTime": end_time_ms,
        "limit": LIMIT,
    }
    url = f"{API_URL}?{urlencode(params)}"

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "trade-lab/1.0"})
            with urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
                raw = resp.read().decode("utf-8")
            payload = json.loads(raw)
            if not isinstance(payload, list):
                raise RuntimeError(f"Unexpected Binance response: {payload}")
            return payload
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
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


def download_btcusdt_4h_last_3y(
    out_path: Path = DEFAULT_OUT_PATH,
    retries: int = DEFAULT_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    request_gap_seconds: float = DEFAULT_REQUEST_GAP_SECONDS,
) -> DownloadSummary:
    end_dt = _utc_now()
    start_dt = end_dt - timedelta(days=YEARS * 365)

    start_ms = _to_ms(start_dt)
    end_ms = _to_ms(end_dt)

    rows: list[list[Any]] = []
    cursor = start_ms
    last_open_time: int | None = None

    while True:
        batch = _fetch_klines(
            start_time_ms=cursor,
            end_time_ms=end_ms,
            retries=retries,
            backoff_seconds=backoff_seconds,
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

    _write_rows_atomic(out_path=out_path, rows=rows)

    return DownloadSummary(
        saved_path=out_path,
        rows=len(rows),
        first_timestamp=rows[0][0],
        last_timestamp=rows[-1][0],
    )


def main() -> None:
    print(f"Downloading {SYMBOL} {INTERVAL} last {YEARS} years from Binance (UTC)...", flush=True)
    try:
        summary = download_btcusdt_4h_last_3y()
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
