"""Regression tests for real numeric input validation and nullable pandas."""

import numpy as np
import pandas as pd
import pytest

from quantkit.portfolio import exposure, returns, valuation


@pytest.mark.parametrize("value", [np.timedelta64(1, "D"), np.timedelta64("NaT", "D")])
def test_temporal_values_are_not_money_or_shares(value):
    with pytest.raises(TypeError, match="real numeric"):
        exposure.net_exposure(np.array([value]))
    with pytest.raises(TypeError, match="real numeric"):
        valuation.net_asset_value(np.array([100.]), cash=value)
    with pytest.raises(TypeError, match="real numeric"):
        valuation.nav_per_share(value, 2.)
    with pytest.raises(TypeError, match="real numeric"):
        valuation.nav_per_share(100., value)
    with pytest.raises(TypeError, match="real numeric"):
        valuation.position_values(np.array([1.]), np.array([value]))


@pytest.mark.parametrize("dtype", ["Int64", "Float64"])
def test_nullable_numeric_histories_preserve_missing_values_and_labels(dtype):
    values = pd.DataFrame({"a": [100, 40, None], "b": [-40, 10, 20]}, dtype=dtype)
    values.index = pd.Index(["t1", "t2", "t3"], name="time")
    original = values.copy(deep=True)
    expected = pd.Series([60., 50., np.nan], index=values.index)
    pd.testing.assert_series_equal(exposure.net_exposure(values), expected)
    pd.testing.assert_series_equal(valuation.net_asset_value(values), expected)
    pd.testing.assert_frame_equal(values, original)


def test_nullable_paired_frames_are_float_after_both_axes_are_aligned():
    quantities = pd.DataFrame(
        {"a": [2, 3], "b": [-1, -2]}, index=["t1", "t2"], dtype="Int64"
    )
    prices = pd.DataFrame(
        {"b": [None, 20.], "a": [30., 10.]},
        index=["t2", "t1"], dtype="Float64",
    )
    original = prices.copy(deep=True)
    expected = pd.DataFrame(
        {"a": [20., 90.], "b": [-20., np.nan]}, index=quantities.index
    )
    pd.testing.assert_frame_equal(valuation.position_values(quantities, prices), expected)
    pd.testing.assert_frame_equal(prices, original)


def test_nullable_weights_broadcast_and_preserve_missing_returns():
    data = pd.DataFrame({"a": [.1, None], "b": [-.05, .1]}, dtype="Float64")
    weights = pd.Series([.4, .6], index=["b", "a"], dtype="Float64")
    expected = pd.Series([.04, np.nan])
    pd.testing.assert_series_equal(returns.portfolio_returns(data, weights), expected)


def test_nullable_nav_and_shares_remain_float_after_alignment():
    nav = pd.Series([100, 200], index=["t1", "t2"], dtype="Int64", name="NAV")
    shares = pd.Series([None, 10], index=["t2", "t1"], dtype="Int64")
    expected = pd.Series([10., np.nan], index=nav.index, name=nav.name)
    pd.testing.assert_series_equal(valuation.nav_per_share(nav, shares), expected)


@pytest.mark.parametrize("column", [
    pd.Series([True, None], dtype="boolean"),
    pd.Series(["1", None], dtype="string"),
    pd.Series([1., 2.], dtype=object),
    pd.Series([1j, 2j]),
    pd.Series(pd.to_timedelta([1, None], unit="D")),
    pd.Series(pd.to_datetime(["2026-01-01", None])),
])
def test_nonnumeric_pandas_columns_cannot_be_coerced_to_money(column):
    frame = pd.DataFrame({"valid": pd.Series([1, 2], dtype="Int64"), "invalid": column})
    with pytest.raises(TypeError, match="real numeric"):
        exposure.net_exposure(frame)
    with pytest.raises(TypeError, match="real numeric"):
        valuation.nav_per_share(column, 2.)


def test_nullable_infinity_still_rejected():
    frame = pd.DataFrame({"a": [1., np.inf]}, dtype="Float64")
    with pytest.raises(ValueError, match="infinity"):
        exposure.net_exposure(frame)
