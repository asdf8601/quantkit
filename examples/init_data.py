"""Download the tour's historical prices once and reuse the local snapshot."""

import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data" / "market"
TICKERS = ["SPY", "QQQ", "TLT", "GLD"]
START = "2018-01-01"
END = "2026-01-01"  # Yahoo's end date is exclusive.
FIELDS = {
    "Close": "close",
    "Adj Close": "adjusted_close",
    "Dividends": "dividends",
    "Stock Splits": "stock_splits",
}


def prepare_prices(raw):
    """Validate the download and return one row per date and ticker."""
    if raw is None or raw.empty:
        raise ValueError("Yahoo returned no data; no snapshot was saved.")
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise ValueError("Expected a datetime index.")
    if raw.index.has_duplicates or raw.index.hasnans:
        raise ValueError("Dates must be unique and non-missing.")
    frames = []
    for ticker in TICKERS:
        columns = [(field, ticker) for field in FIELDS]
        if any(column not in raw.columns for column in columns):
            raise ValueError(f"Missing required data for {ticker}.")
        frame = raw.loc[:, columns].copy()
        frame.columns = list(FIELDS.values())
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Missing or non-finite data for {ticker}.")
        if (frame[["close", "adjusted_close"]] <= 0).any().any():
            raise ValueError(f"Non-positive prices for {ticker}.")
        frame.index = frame.index.tz_localize(None)
        if ((frame.index < START) | (frame.index >= END)).any():
            raise ValueError("Dates outside the requested range.")
        frame.index.name = "date"
        frames.append(frame.reset_index().assign(ticker=ticker))
    return pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"])


def download_prices():
    """Fetch unadjusted and adjusted closes plus corporate actions."""
    import yfinance as yf

    raw = yf.download(
        TICKERS,
        start=START,
        end=END,
        interval="1d",
        auto_adjust=False,
        back_adjust=False,
        actions=True,
        repair=False,
        keepna=True,
        group_by="column",
        multi_level_index=True,
        threads=False,
        progress=False,
    )
    return prepare_prices(raw), yf.__version__


def initialize(destination=DATA_DIR):
    """Create a complete snapshot, or verify and reuse the existing one."""
    destination = Path(destination)
    csv_path = destination / "prices.csv"
    metadata_path = destination / "metadata.json"
    if destination.exists():
        if not csv_path.is_file() or not metadata_path.is_file():
            raise ValueError(f"Incomplete snapshot at {destination}.")
        metadata = json.loads(metadata_path.read_text())
        digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        if digest != metadata["sha256"]:
            raise ValueError(f"Snapshot checksum mismatch at {destination}.")
        print(f"Reusing {csv_path} (no download).")
        return csv_path

    prices, version = download_prices()
    metadata = {
        "source": "Yahoo Finance via yfinance",
        "source_url": "https://finance.yahoo.com/",
        "yfinance_version": version,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "tickers": TICKERS,
        "currency": "USD",
        "interval": "1d",
        "start_inclusive": START,
        "end_exclusive": END,
        "auto_adjust": False,
        "back_adjust": False,
        "repair": False,
        "first_date": str(prices["date"].min().date()),
        "last_date": str(prices["date"].max().date()),
        "rows": len(prices),
        "fields": FIELDS,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish both files together; failed downloads never become cached data.
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary) / "market"
        staging.mkdir()
        staged_csv = staging / csv_path.name
        prices.to_csv(staged_csv, index=False, date_format="%Y-%m-%d")
        metadata["sha256"] = hashlib.sha256(
            staged_csv.read_bytes()
        ).hexdigest()
        (staging / metadata_path.name).write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        staging.rename(destination)
    print(f"Saved {len(prices)} rows to {csv_path}.")
    return csv_path


if __name__ == "__main__":
    initialize()
