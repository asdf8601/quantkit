"""Load the tour's fixed local snapshot without accessing the network."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "market"


def load_prices():
    """Return validated long-form prices and their source metadata."""
    csv_path = DATA_DIR / "prices.csv"
    metadata_path = DATA_DIR / "metadata.json"
    if not csv_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Run `uv run --group examples python examples/init_data.py` "
            "from the repository root to initialize the local data."
        )
    metadata = json.loads(metadata_path.read_text())
    if hashlib.sha256(csv_path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("The example data does not match its checksum.")
    data = pd.read_csv(csv_path, parse_dates=["date"])
    if data.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker observations.")
    prices = data.pivot(
        index="date", columns="ticker", values=["close", "adjusted_close"]
    )
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Prices must be finite, positive and aligned.")
    return data, metadata
