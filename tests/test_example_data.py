"""Checks for offline reuse and safe initialization of example data."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "init_data.py"
spec = importlib.util.spec_from_file_location("init_data", SCRIPT)
init_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(init_data)


@pytest.fixture
def raw_prices():
    columns = pd.MultiIndex.from_product([init_data.FIELDS, init_data.TICKERS])
    return pd.DataFrame(
        100.0,
        index=pd.to_datetime(["2018-01-02", "2018-01-03"]),
        columns=columns,
    )


def test_initialize_reuses_identical_files_without_downloading(
    tmp_path, monkeypatch, raw_prices
):
    monkeypatch.setattr(
        init_data,
        "download_prices",
        lambda: (init_data.prepare_prices(raw_prices), "test"),
    )
    destination = tmp_path / "market"
    csv = init_data.initialize(destination)
    before = {p.name: p.read_bytes() for p in destination.iterdir()}

    def no_network():
        pytest.fail("Existing data must be reused without downloading")

    monkeypatch.setattr(init_data, "download_prices", no_network)
    assert init_data.initialize(destination) == csv
    assert before == {p.name: p.read_bytes() for p in destination.iterdir()}
    restored = pd.read_csv(csv)
    assert len(restored) == 8
    assert set(restored["ticker"]) == set(init_data.TICKERS)
    assert json.loads(before["metadata.json"])["rows"] == 8

    csv.write_text("damaged")
    with pytest.raises(ValueError, match="checksum mismatch"):
        init_data.initialize(destination)


def test_incomplete_snapshot_is_not_overwritten(tmp_path):
    with pytest.raises(ValueError, match="Incomplete snapshot"):
        init_data.initialize(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("problem", ["empty", "missing_ticker", "nan", "zero"])
def test_bad_download_is_not_published(
    tmp_path, monkeypatch, raw_prices, problem
):
    if problem == "empty":
        raw_prices = raw_prices.iloc[:0]
    elif problem == "missing_ticker":
        raw_prices = raw_prices.drop(columns="GLD", level=1)
    elif problem == "nan":
        raw_prices.iloc[0, 0] = np.nan
    else:
        raw_prices.iloc[0, 0] = 0
    monkeypatch.setattr(
        init_data,
        "download_prices",
        lambda: (init_data.prepare_prices(raw_prices), "test"),
    )
    destination = tmp_path / "market"
    with pytest.raises(ValueError):
        init_data.initialize(destination)
    assert not destination.exists()
