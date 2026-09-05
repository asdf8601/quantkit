"""Tests for annualized_return and calmar_ratio.

Every test is a small worked example: the expected value is derived by hand
in a comment next to the data, so the file doubles as the specification of
both statistics.
"""
from quantkit import stats
from quantkit.conventions import BYEAR

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# annualized_return: geometric


def test_annualized_return_compounds_two_periods_into_one_year():
    # 1.1 * 1.1 = 1.21; with periods_per_year=2 the two returns span a year
    # 1.21 ** (2 / 2) - 1 = 0.21
    returns = np.array([0.1, 0.1])
    obtained = stats.annualized_return(returns, periods_per_year=2)
    np.testing.assert_almost_equal(obtained, 0.21)


def test_annualized_return_extrapolates_half_year_to_full_year():
    # 1.21 ** (4 / 2) - 1 = 1.4641 - 1 = 0.4641
    returns = np.array([0.1, 0.1])
    obtained = stats.annualized_return(returns, periods_per_year=4)
    np.testing.assert_almost_equal(obtained, 0.4641)


def test_annualized_return_deannualizes_a_two_year_span():
    # four returns with periods_per_year=2 span two years
    # (1.1 ** 4) ** (2 / 4) - 1 = 1.1 ** 2 - 1 = 0.21
    returns = np.array([0.1, 0.1, 0.1, 0.1])
    obtained = stats.annualized_return(returns, periods_per_year=2)
    np.testing.assert_almost_equal(obtained, 0.21)


def test_annualized_return_single_return_with_one_period_per_year_is_itself():
    # periods_per_year at its lower bound: 1.05 ** (1 / 1) - 1 = 0.05
    returns = np.array([0.05])
    obtained = stats.annualized_return(returns, periods_per_year=1)
    np.testing.assert_almost_equal(obtained, 0.05)


def test_annualized_return_defaults_to_business_year():
    # BYEAR = 261 business days: 1.001 * 1.001 = 1.002001
    # 1.002001 ** (261 / 2) - 1 = 0.2981...; 252 days would give 0.2861...
    returns = np.array([0.001, 0.001])
    obtained = stats.annualized_return(returns)
    expected = 1.002001 ** (BYEAR / 2) - 1
    np.testing.assert_almost_equal(obtained, expected)


def test_annualized_return_flat_returns_give_zero():
    # 1.0 * 1.0 = 1; 1 ** (4 / 2) - 1 = 0
    returns = np.array([0.0, 0.0])
    obtained = stats.annualized_return(returns, periods_per_year=4)
    np.testing.assert_almost_equal(obtained, 0.0)


def test_annualized_return_total_loss_is_exactly_minus_one():
    # 1.1 * 0.0 * 1.2 = 0; 0 ** (4 / 3) - 1 = -1, no rounding involved
    returns = np.array([0.1, -1.0, 0.2])
    obtained = stats.annualized_return(returns, periods_per_year=4)
    assert obtained == -1.0


# ---------------------------------------------------------------------------
# annualized_return: NaN handling


def test_annualized_return_ignores_leading_nan_in_product_and_count():
    # nan is dropped from both the product and n: same as [0.1, 0.1]
    # 1.21 ** (4 / 2) - 1 = 0.4641
    returns = np.array([np.nan, 0.1, 0.1])
    obtained = stats.annualized_return(returns, periods_per_year=4)
    np.testing.assert_almost_equal(obtained, 0.4641)


def test_annualized_return_ignores_nan_in_the_middle():
    # 1.21 ** (4 / 2) - 1 = 0.4641
    returns = np.array([0.1, np.nan, 0.1])
    obtained = stats.annualized_return(returns, periods_per_year=4)
    np.testing.assert_almost_equal(obtained, 0.4641)


def test_annualized_return_empty_input_is_nan():
    # n = 0: nothing to annualize
    returns = np.array([])
    obtained = stats.annualized_return(returns, periods_per_year=4)
    assert np.isnan(obtained)


def test_annualized_return_all_nan_input_is_nan():
    # n = 0 after dropping nan
    returns = np.array([np.nan, np.nan])
    obtained = stats.annualized_return(returns, periods_per_year=4)
    assert np.isnan(obtained)


# ---------------------------------------------------------------------------
# annualized_return: containers


def test_annualized_return_2d_reduces_each_column_with_its_own_count():
    # col 0: 1.1 * 1.1 = 1.21, n = 2 -> 1.21 ** (2 / 2) - 1 = 0.21
    # col 1: 1.2 (nan dropped), n = 1 -> 1.2 ** (2 / 1) - 1 = 0.44
    returns = np.array([[0.1, 0.2], [0.1, np.nan]])
    obtained = stats.annualized_return(returns, periods_per_year=2)
    np.testing.assert_almost_equal(obtained, np.array([0.21, 0.44]))


def test_annualized_return_2d_all_nan_column_is_nan_only_for_that_column():
    # col 0: 1.21 ** (2 / 2) - 1 = 0.21; col 1: n = 0 -> nan
    returns = np.array([[0.1, np.nan], [0.1, np.nan]])
    obtained = stats.annualized_return(returns, periods_per_year=2)
    np.testing.assert_almost_equal(obtained, np.array([0.21, np.nan]))


def test_annualized_return_series_reduces_to_a_float():
    # a 1-d input reduces to a scalar, there is no container left to name
    # 1.21 ** (4 / 2) - 1 = 0.4641
    returns = pd.Series([0.1, 0.1], name="asset")
    obtained = stats.annualized_return(returns, periods_per_year=4)
    assert isinstance(obtained, float)
    np.testing.assert_almost_equal(obtained, 0.4641)


def test_annualized_return_dataframe_returns_series_indexed_by_columns():
    # a: 1.21 ** (2 / 2) - 1 = 0.21; b: 1.2 ** (2 / 1) - 1 = 0.44
    returns = pd.DataFrame({"a": [0.1, 0.1], "b": [0.2, np.nan]})
    obtained = stats.annualized_return(returns, periods_per_year=2)
    expected = pd.Series([0.21, 0.44], index=["a", "b"])
    pd.testing.assert_series_equal(obtained, expected)


def test_annualized_return_all_nan_dataframe_is_nan_per_column():
    returns = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
    obtained = stats.annualized_return(returns, periods_per_year=2)
    expected = pd.Series([np.nan, np.nan], index=["a", "b"])
    pd.testing.assert_series_equal(obtained, expected)


# ---------------------------------------------------------------------------
# calmar_ratio: definition


def test_calmar_ratio_divides_annualized_return_by_max_drawdown():
    # prices: 1.1, 0.55, 0.66; max drawdown = 0.55 / 1.1 - 1 = -0.5
    # annualized: 1.1 * 0.5 * 1.2 - 1 = -0.34 (periods_per_year = n = 3)
    # calmar = -0.34 / 0.5 = -0.68
    returns = np.array([0.1, -0.5, 0.2])
    obtained = stats.calmar_ratio(returns, periods_per_year=3)
    np.testing.assert_almost_equal(obtained, -0.68)


def test_calmar_ratio_sign_follows_the_annualized_return():
    # prices: 1.5, 0.75, 1.2; max drawdown = 0.75 / 1.5 - 1 = -0.5
    # annualized: 1.5 * 0.5 * 1.6 - 1 = 0.2
    # calmar = 0.2 / 0.5 = 0.4, positive because the return is positive
    returns = np.array([0.5, -0.5, 0.6])
    obtained = stats.calmar_ratio(returns, periods_per_year=3)
    np.testing.assert_almost_equal(obtained, 0.4)


def test_calmar_ratio_annualizes_only_the_numerator():
    # same prices as above, max drawdown stays -0.5
    # periods_per_year=6 makes the 3 returns half a year:
    # annualized: 0.66 ** (6 / 3) - 1 = 0.4356 - 1 = -0.5644
    # calmar = -0.5644 / 0.5 = -1.1288
    returns = np.array([0.1, -0.5, 0.2])
    obtained = stats.calmar_ratio(returns, periods_per_year=6)
    np.testing.assert_almost_equal(obtained, -1.1288)


def test_calmar_ratio_defaults_to_business_year():
    # prices: 1 (start), 1.01, 0.9999, 1.009899
    # max drawdown = 0.9999 / 1.01 - 1 = -0.01
    # annualized: (1.01 * 0.99 * 1.01) ** (261 / 3) - 1 = 1.009899 ** 87 - 1
    # calmar = 1.356... / 0.01 = 135.6...; 252 days would give 128.7...
    returns = np.array([0.01, -0.01, 0.01])
    obtained = stats.calmar_ratio(returns)
    expected = (1.009899 ** (BYEAR / 3) - 1) / 0.01
    np.testing.assert_almost_equal(obtained, expected)


def test_calmar_ratio_counts_drawdown_from_initial_capital():
    # the first return already loses half of the starting capital of 1
    # prices: 1 (start), 0.5, 0.5; max drawdown = 0.5 / 1 - 1 = -0.5
    # annualized: 0.5 * 1.0 - 1 = -0.5; calmar = -0.5 / 0.5 = -1
    returns = np.array([-0.5, 0.0])
    obtained = stats.calmar_ratio(returns, periods_per_year=2)
    np.testing.assert_almost_equal(obtained, -1.0)


def test_calmar_ratio_total_loss_is_minus_one():
    # prices: 1 (start), 0; max drawdown = -1; annualized = -1
    # calmar = -1 / 1 = -1
    returns = np.array([-1.0])
    obtained = stats.calmar_ratio(returns, periods_per_year=1)
    np.testing.assert_almost_equal(obtained, -1.0)


# ---------------------------------------------------------------------------
# calmar_ratio: zero denominator


def test_calmar_ratio_without_drawdown_is_nan():
    # prices 1.1, 1.32 never fall below the running max: max drawdown = 0
    returns = np.array([0.1, 0.2])
    obtained = stats.calmar_ratio(returns, periods_per_year=2)
    assert np.isnan(obtained)


def test_calmar_ratio_flat_returns_are_nan():
    # prices 1, 1: annualized return 0 over a drawdown of 0
    returns = np.array([0.0, 0.0])
    obtained = stats.calmar_ratio(returns, periods_per_year=2)
    assert np.isnan(obtained)


# ---------------------------------------------------------------------------
# calmar_ratio: NaN handling


def test_calmar_ratio_ignores_leading_nan():
    # nan adds nothing to the product, the count or the price path
    # same as [0.1, -0.5, 0.2] -> -0.68
    returns = np.array([np.nan, 0.1, -0.5, 0.2])
    obtained = stats.calmar_ratio(returns, periods_per_year=3)
    np.testing.assert_almost_equal(obtained, -0.68)


def test_calmar_ratio_ignores_nan_in_the_middle():
    # prices carry forward over the gap: 1.1, 1.1, 0.55, 0.66 -> -0.68
    returns = np.array([0.1, np.nan, -0.5, 0.2])
    obtained = stats.calmar_ratio(returns, periods_per_year=3)
    np.testing.assert_almost_equal(obtained, -0.68)


def test_calmar_ratio_empty_input_is_nan():
    returns = np.array([])
    obtained = stats.calmar_ratio(returns, periods_per_year=3)
    assert np.isnan(obtained)


def test_calmar_ratio_all_nan_input_is_nan():
    returns = np.array([np.nan, np.nan])
    obtained = stats.calmar_ratio(returns, periods_per_year=3)
    assert np.isnan(obtained)


# ---------------------------------------------------------------------------
# calmar_ratio: containers


def test_calmar_ratio_2d_reduces_each_column_independently():
    # col 0: [0.1, -0.5, 0.2] -> -0.34 / 0.5 = -0.68
    # col 1: [0.5, -0.5, 0.6] -> 0.2 / 0.5 = 0.4
    returns = np.array([[0.1, 0.5], [-0.5, -0.5], [0.2, 0.6]])
    obtained = stats.calmar_ratio(returns, periods_per_year=3)
    np.testing.assert_almost_equal(obtained, np.array([-0.68, 0.4]))


def test_calmar_ratio_2d_column_without_drawdown_is_nan_only_there():
    # col 0: -0.68; col 1: prices 1.1, 1.32, 1.716 never fall -> nan
    returns = np.array([[0.1, 0.1], [-0.5, 0.2], [0.2, 0.3]])
    obtained = stats.calmar_ratio(returns, periods_per_year=3)
    np.testing.assert_almost_equal(obtained, np.array([-0.68, np.nan]))


def test_calmar_ratio_series_reduces_to_a_float():
    # a 1-d input reduces to a scalar, there is no container left to name
    returns = pd.Series([0.1, -0.5, 0.2], name="asset")
    obtained = stats.calmar_ratio(returns, periods_per_year=3)
    assert isinstance(obtained, float)
    np.testing.assert_almost_equal(obtained, -0.68)


def test_calmar_ratio_dataframe_returns_series_indexed_by_columns():
    # a: -0.68; b: 0.4
    returns = pd.DataFrame({"a": [0.1, -0.5, 0.2], "b": [0.5, -0.5, 0.6]})
    obtained = stats.calmar_ratio(returns, periods_per_year=3)
    expected = pd.Series([-0.68, 0.4], index=["a", "b"])
    pd.testing.assert_series_equal(obtained, expected)


def test_calmar_ratio_all_nan_dataframe_is_nan_per_column():
    returns = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
    obtained = stats.calmar_ratio(returns, periods_per_year=3)
    expected = pd.Series([np.nan, np.nan], index=["a", "b"])
    pd.testing.assert_series_equal(obtained, expected)
