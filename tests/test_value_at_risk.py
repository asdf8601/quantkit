"""Tests for the historical value at risk.

Historical VaR at confidence ``c`` is the empirical ``(1 - c)`` quantile of
the returns, computed with numpy's default linear interpolation. It keeps
the sign of the return: a loss is a negative number, as drawdown does in
this library.
"""
import warnings

from quantkit import stats

import numpy as np
import pandas as pd
import pytest


# sorted: -0.05, -0.02, 0.0, 0.01, 0.03
RETURNS = np.array([-0.05, -0.02, 0.0, 0.01, 0.03])


def test_value_at_risk_is_the_lower_quantile_of_returns():
    # confidence 0.75 -> 25th percentile
    # position = 0.25 * (5 - 1) = 1.0 -> exactly the 2nd sorted value
    obtained = stats.value_at_risk(RETURNS, confidence=0.75)
    expected = -0.02
    np.testing.assert_almost_equal(obtained, expected)


def test_value_at_risk_defaults_to_95_percent_confidence():
    # 5th percentile: position = 0.05 * 4 = 0.2
    # -0.05 + 0.2 * (-0.02 - -0.05) = -0.05 + 0.006 = -0.044
    obtained = stats.value_at_risk(RETURNS)
    expected = stats.value_at_risk(RETURNS, confidence=0.95)
    np.testing.assert_almost_equal(obtained, -0.044)
    np.testing.assert_almost_equal(obtained, expected)


def test_value_at_risk_interpolates_linearly_between_order_statistics():
    # sorted: -0.04, -0.01, 0.02, 0.05 (n = 4)
    # confidence 0.9 -> 10th percentile
    # numpy linear rule: position = q * (n - 1) = 0.10 * 3 = 0.3
    # lower = sorted[0] = -0.04, upper = sorted[1] = -0.01, fraction = 0.3
    # value = -0.04 + 0.3 * (-0.01 - -0.04) = -0.04 + 0.009 = -0.031
    rets = np.array([0.02, -0.04, 0.05, -0.01])
    obtained = stats.value_at_risk(rets, confidence=0.9)
    expected = -0.031
    np.testing.assert_almost_equal(obtained, expected)


def test_value_at_risk_at_confidence_one_half_is_the_median():
    # sorted: -0.04, -0.01, 0.01, 0.02, 0.03; median = 0.01
    rets = np.array([0.02, -0.01, 0.03, -0.04, 0.01])
    obtained = stats.value_at_risk(rets, confidence=0.5)
    np.testing.assert_almost_equal(obtained, 0.01)
    np.testing.assert_almost_equal(obtained, np.median(rets))


def test_value_at_risk_reports_a_loss_as_a_negative_number():
    # the quantile is returned as is: no negation, unlike Morningstar
    assert stats.value_at_risk(RETURNS, confidence=0.75) < 0

    # sorted: 0.01, 0.02, 0.03, 0.04, 0.05; 25th percentile = 0.02 (a gain)
    gains = np.array([0.03, 0.01, 0.05, 0.02, 0.04])
    obtained = stats.value_at_risk(gains, confidence=0.75)
    np.testing.assert_almost_equal(obtained, 0.02)


def test_value_at_risk_returns_a_float_for_one_dimensional_input():
    obtained = stats.value_at_risk(RETURNS, confidence=0.75)
    assert isinstance(obtained, float)
    assert np.ndim(obtained) == 0


@pytest.mark.parametrize("confidence", [0, 1, 0.0, 1.0, -0.5, 1.5])
def test_value_at_risk_confidence_outside_open_unit_interval_raises(
    confidence,
):
    with pytest.raises(ValueError):
        stats.value_at_risk(RETURNS, confidence=confidence)


def test_value_at_risk_accepts_confidence_just_inside_the_bounds():
    # confidence 0.99 -> 1st percentile: position = 0.01 * 4 = 0.04
    # -0.05 + 0.04 * 0.03 = -0.0488
    obtained = stats.value_at_risk(RETURNS, confidence=0.99)
    np.testing.assert_almost_equal(obtained, -0.0488)

    # confidence 0.01 -> 99th percentile: position = 0.99 * 4 = 3.96
    # 0.01 + 0.96 * (0.03 - 0.01) = 0.01 + 0.0192 = 0.0292
    obtained = stats.value_at_risk(RETURNS, confidence=0.01)
    np.testing.assert_almost_equal(obtained, 0.0292)


def test_value_at_risk_is_non_increasing_in_confidence():
    # sorted: -0.03, -0.02, -0.01, 0.01, 0.02, 0.04
    rets = np.array([0.02, -0.03, 0.01, -0.01, 0.04, -0.02])
    confidences = [0.5, 0.75, 0.9, 0.95, 0.99]
    values = [stats.value_at_risk(rets, confidence=c) for c in confidences]
    assert np.all(np.diff(values) <= 0), values


def test_value_at_risk_ignores_nan_at_the_start_and_in_the_middle():
    # valid values are RETURNS, so the answer is the same: -0.02
    rets = np.array([np.nan, -0.05, -0.02, np.nan, 0.0, 0.01, 0.03])
    obtained = stats.value_at_risk(rets, confidence=0.75)
    expected = stats.value_at_risk(RETURNS, confidence=0.75)
    np.testing.assert_almost_equal(obtained, -0.02)
    np.testing.assert_almost_equal(obtained, expected)


def test_value_at_risk_of_empty_input_is_nan():
    obtained = stats.value_at_risk(np.array([]))
    assert np.isnan(obtained)


def test_value_at_risk_of_all_nan_input_is_nan():
    obtained = stats.value_at_risk(np.array([np.nan, np.nan, np.nan]))
    assert np.isnan(obtained)


def test_value_at_risk_does_not_warn_on_empty_or_all_nan_input():
    # numpy warns "All-NaN slice encountered" / "Mean of empty slice";
    # the function must swallow both so a test run stays clean
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stats.value_at_risk(np.array([np.nan, np.nan]))
        stats.value_at_risk(np.array([]))
        stats.value_at_risk(np.array([[np.nan, 0.01], [np.nan, 0.02]]))


def test_value_at_risk_2d_reduces_each_column():
    # column 1 is column 0 shifted by +0.10, so its 25th percentile is
    # -0.02 + 0.10 = 0.08
    rets = np.column_stack([RETURNS, RETURNS + 0.10])
    obtained = stats.value_at_risk(rets, confidence=0.75)
    expected = np.array([-0.02, 0.08])
    assert obtained.shape == (2,)
    np.testing.assert_almost_equal(obtained, expected)


def test_value_at_risk_2d_ignores_nan_per_column():
    # col 0 has a NaN at the start, col 1 in the middle; valid values are
    # RETURNS and RETURNS + 0.10 respectively -> [-0.02, 0.08]
    rets = np.array(
        [
            [np.nan, 0.05],
            [-0.05, 0.08],
            [-0.02, np.nan],
            [0.0, 0.10],
            [0.01, 0.11],
            [0.03, 0.13],
        ]
    )
    obtained = stats.value_at_risk(rets, confidence=0.75)
    expected = np.array([-0.02, 0.08])
    np.testing.assert_almost_equal(obtained, expected)


def test_value_at_risk_2d_all_nan_column_is_nan_only_for_that_column():
    rets = np.column_stack([RETURNS, np.full(5, np.nan)])
    obtained = stats.value_at_risk(rets, confidence=0.75)
    expected = np.array([-0.02, np.nan])
    np.testing.assert_almost_equal(obtained, expected)


def test_value_at_risk_series_reduces_to_a_float():
    # a reduction drops the Series axis, so the name cannot survive; the
    # index and name must not get in the way of the result
    rets = pd.Series(
        RETURNS, index=pd.date_range("2020", periods=5), name="asset"
    )
    obtained = stats.value_at_risk(rets, confidence=0.75)
    assert isinstance(obtained, float)
    np.testing.assert_almost_equal(obtained, -0.02)


def test_value_at_risk_series_with_nan_matches_series_without_them():
    with_nan = pd.Series([np.nan, -0.05, -0.02, np.nan, 0.0, 0.01, 0.03])
    without_nan = pd.Series(RETURNS)
    obtained = stats.value_at_risk(with_nan, confidence=0.75)
    expected = stats.value_at_risk(without_nan, confidence=0.75)
    np.testing.assert_almost_equal(obtained, expected)


def test_value_at_risk_empty_series_is_nan():
    obtained = stats.value_at_risk(pd.Series([], dtype=float))
    assert np.isnan(obtained)


def test_value_at_risk_dataframe_returns_a_series_indexed_by_columns():
    rets = pd.DataFrame(
        {"a": RETURNS, "b": RETURNS + 0.10},
        index=pd.date_range("2020", periods=5),
    )
    obtained = stats.value_at_risk(rets, confidence=0.75)
    expected = pd.Series([-0.02, 0.08], index=["a", "b"])
    pd.testing.assert_series_equal(obtained, expected)


def test_value_at_risk_dataframe_ignores_nan_per_column():
    # a: NaN at the start, valid values are RETURNS        -> -0.02
    # b: NaN in the middle, valid values are RETURNS + 0.1 -> 0.08
    rets = pd.DataFrame(
        {
            "a": [np.nan, -0.05, -0.02, 0.0, 0.01, 0.03],
            "b": [0.05, 0.08, np.nan, 0.10, 0.11, 0.13],
        }
    )
    obtained = stats.value_at_risk(rets, confidence=0.75)
    expected = pd.Series([-0.02, 0.08], index=["a", "b"])
    pd.testing.assert_series_equal(obtained, expected)


def test_value_at_risk_dataframe_all_nan_column_is_nan_for_that_column():
    rets = pd.DataFrame({"a": RETURNS, "b": np.full(5, np.nan)})
    obtained = stats.value_at_risk(rets, confidence=0.75)
    expected = pd.Series([-0.02, np.nan], index=["a", "b"])
    pd.testing.assert_series_equal(obtained, expected)
