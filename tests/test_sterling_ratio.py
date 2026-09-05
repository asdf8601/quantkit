"""Tests for sterling_ratio.

Every test is a small worked example: the annualized return, the average
drawdown and the division are derived by hand in a comment next to the data,
so the file doubles as the specification of the statistic.

The price path is the one :func:`quantkit.stats.calmar_ratio` uses: the
initial capital of 1 followed by ``cum_returns(returns, first_price=1)``.
That leading 1 is an observation like any other, so a series of ``n`` returns
produces ``n + 1`` prices and the blocks of :func:`average_drawdown` are cut
over those ``n + 1`` observations.
"""
from quantkit import stats
from quantkit.conventions import BYEAR

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# definition


def test_sterling_ratio_adds_the_excess_to_the_average_drawdown():
    # prices: 1 (start), 1.1, 0.55, 0.66
    # blocks of 3: [1, 1.1, 0.55] -> 0.55 / 1.1 - 1 = -0.5 and [0.66] -> 0
    # average drawdown = -0.5 / (4 / 3) = -0.375
    # annualized: 1.1 * 0.5 * 1.2 - 1 = -0.34 (periods_per_year = n = 3)
    # sterling = -0.34 / (0.375 + 0.10) = -0.7157894736842105
    returns = np.array([0.1, -0.5, 0.2])
    obtained = stats.sterling_ratio(returns, periods_per_year=3)
    np.testing.assert_almost_equal(obtained, -0.34 / 0.475)


def test_sterling_ratio_over_a_single_block_uses_the_maximum_drawdown():
    # 3 returns make 4 prices: 1 (start), 1.1, 0.55, 0.66
    # periods_per_year=4 is one block covering exactly one year, so the
    # average drawdown is the maximum drawdown: 0.55 / 1.1 - 1 = -0.5
    # annualized: 0.66 ** (4 / 3) - 1 = -0.4253652123825415
    # sterling = -0.4253652123825415 / (0.5 + 0.10) = -0.7089420206375693
    returns = np.array([0.1, -0.5, 0.2])
    obtained = stats.sterling_ratio(returns, periods_per_year=4)
    expected = (0.66 ** (4 / 3) - 1) / 0.6
    np.testing.assert_almost_equal(obtained, expected)
    np.testing.assert_almost_equal(
        stats.average_drawdown(np.array([1.0, 1.1, 0.55, 0.66]), 4), -0.5
    )


def test_sterling_ratio_averages_two_yearly_blocks_that_differ():
    # prices: 1 (start), 1, 0.5, 0.5, 1, 0.8, 0.8, 0.8
    # blocks of 4: [1, 1, 0.5, 0.5] -> 0.5 / 1 - 1 = -0.5
    #              [1, 0.8, 0.8, 0.8] -> 0.8 / 1 - 1 = -0.2
    # average drawdown = (-0.5 - 0.2) / (8 / 4) = -0.7 / 2 = -0.35
    # annualized: growth 0.8 over n = 7 -> 0.8 ** (4 / 7) - 1 = -0.1197159
    # sterling = -0.1197159 / (0.35 + 0.10) = -0.2660353686391224
    returns = np.array([0.0, -0.5, 0.0, 1.0, -0.2, 0.0, 0.0])
    obtained = stats.sterling_ratio(returns, periods_per_year=4)
    expected = (0.8 ** (4 / 7) - 1) / 0.45
    np.testing.assert_almost_equal(obtained, expected)


def test_sterling_ratio_with_zero_excess_reproduces_the_calmar_ratio():
    # one block of 4 prices makes the average drawdown the maximum drawdown,
    # so without the excess the denominator is Calmar's: |-0.5| = 0.5
    # sterling = calmar = (0.66 ** (4 / 3) - 1) / 0.5 = -0.8507304247650831
    returns = np.array([0.1, -0.5, 0.2])
    obtained = stats.sterling_ratio(returns, periods_per_year=4, excess=0.0)
    expected = (0.66 ** (4 / 3) - 1) / 0.5
    np.testing.assert_almost_equal(obtained, expected)
    np.testing.assert_almost_equal(
        obtained, stats.calmar_ratio(returns, periods_per_year=4)
    )


def test_sterling_ratio_counts_the_drawdown_from_the_initial_capital():
    # the first return already loses half of the starting capital of 1
    # prices: 1 (start), 0.5, 0.5; one block of 3 -> 0.5 / 1 - 1 = -0.5
    # annualized: 0.5 ** (3 / 2) - 1 = -0.6464466094067263
    # sterling = -0.6464466094067263 / (0.5 + 0.10) = -1.0774110156778771
    returns = np.array([-0.5, 0.0])
    obtained = stats.sterling_ratio(returns, periods_per_year=3)
    expected = (0.5 ** (3 / 2) - 1) / 0.6
    np.testing.assert_almost_equal(obtained, expected)


def test_sterling_ratio_defaults_to_a_ten_percent_excess():
    # the default denominator is |average drawdown| + 0.10, nothing else
    returns = np.array([0.1, -0.5, 0.2])
    obtained = stats.sterling_ratio(returns, periods_per_year=4)
    expected = stats.sterling_ratio(
        returns, periods_per_year=4, excess=0.10
    )
    np.testing.assert_almost_equal(obtained, expected)
    np.testing.assert_almost_equal(obtained, (0.66 ** (4 / 3) - 1) / 0.6)


def test_sterling_ratio_defaults_to_business_year():
    # prices: 1 (start), 1.01, 0.9999, 1.009899; 4 observations, one block
    # max drawdown = 0.9999 / 1.01 - 1 = -0.01 over 4 / 261 years
    # average drawdown = -0.01 * 261 / 4 = -0.6525
    # annualized: 1.009899 ** (261 / 3) - 1 = 1.3560308...
    # sterling = 1.3560308 / (0.6525 + 0.10) = 1.8020343185504588
    returns = np.array([0.01, -0.01, 0.01])
    obtained = stats.sterling_ratio(returns)
    expected = (1.009899 ** (BYEAR / 3) - 1) / (0.01 * BYEAR / 4 + 0.10)
    np.testing.assert_almost_equal(obtained, expected)


# ---------------------------------------------------------------------------
# zero denominator and parameter bounds


def test_sterling_ratio_of_rising_prices_divides_by_the_excess_alone():
    # prices 1, 1.1, 1.32 never fall below the running max: average
    # drawdown = 0, so the denominator is the excess
    # annualized: 1.1 * 1.2 - 1 = 0.32; sterling = 0.32 / 0.10 = 3.2
    returns = np.array([0.1, 0.2])
    obtained = stats.sterling_ratio(returns, periods_per_year=2)
    np.testing.assert_almost_equal(obtained, 3.2)


def test_sterling_ratio_without_drawdown_and_zero_excess_is_nan():
    # excess at its lower bound: 0 drawdown + 0 excess is a zero denominator
    returns = np.array([0.1, 0.2])
    obtained = stats.sterling_ratio(
        returns, periods_per_year=2, excess=0.0
    )
    assert np.isnan(obtained)


def test_sterling_ratio_with_one_period_per_year_has_no_drawdown_left():
    # periods_per_year at its lower bound: every block is a single price, so
    # no block has a drawdown and the average drawdown is 0
    # annualized: 0.66 ** (1 / 3) - 1 = -0.1293412308826388
    # sterling = -0.1293412308826388 / 0.10 = -1.2934123088263882
    returns = np.array([0.1, -0.5, 0.2])
    obtained = stats.sterling_ratio(returns, periods_per_year=1)
    expected = (0.66 ** (1 / 3) - 1) / 0.10
    np.testing.assert_almost_equal(obtained, expected)


def test_sterling_ratio_with_one_period_per_year_and_zero_excess_is_nan():
    # both parameters at their bound: 0 drawdown + 0 excess -> nan, not inf
    returns = np.array([0.1, -0.5, 0.2])
    obtained = stats.sterling_ratio(
        returns, periods_per_year=1, excess=0.0
    )
    assert np.isnan(obtained)


def test_sterling_ratio_is_nan_when_the_excess_cancels_the_drawdown():
    # a negative excess of exactly -|average drawdown| empties the
    # denominator: -0.5 + 0.5 = 0 -> nan, never inf
    returns = np.array([0.1, -0.5, 0.2])
    obtained = stats.sterling_ratio(
        returns, periods_per_year=4, excess=-0.5
    )
    assert np.isnan(obtained)


# ---------------------------------------------------------------------------
# NaN handling


def test_sterling_ratio_ignores_leading_nan_in_the_annualized_return():
    # the nan adds nothing to the product or to the count, n = 3
    # prices: 1 (start), 1, 1.1, 0.55, 0.66; one block of 5
    # max drawdown = 0.55 / 1.1 - 1 = -0.5 over 5 / 5 = 1 year
    # annualized: 0.66 ** (5 / 3) - 1 = -0.4996891831209674
    # sterling = -0.4996891831209674 / 0.6 = -0.8328153052016124
    returns = np.array([np.nan, 0.1, -0.5, 0.2])
    obtained = stats.sterling_ratio(returns, periods_per_year=5)
    expected = (0.66 ** (5 / 3) - 1) / 0.6
    np.testing.assert_almost_equal(obtained, expected)


def test_sterling_ratio_ignores_nan_in_the_middle():
    # prices carry the previous level forward over the gap:
    # 1 (start), 1.1, 1.1, 0.55, 0.66; the same block and the same n = 3
    # sterling = (0.66 ** (5 / 3) - 1) / 0.6 = -0.8328153052016124
    returns = np.array([0.1, np.nan, -0.5, 0.2])
    obtained = stats.sterling_ratio(returns, periods_per_year=5)
    expected = (0.66 ** (5 / 3) - 1) / 0.6
    np.testing.assert_almost_equal(obtained, expected)


def test_sterling_ratio_empty_input_is_nan():
    # no return to annualize: the numerator is nan whatever the denominator
    returns = np.array([])
    obtained = stats.sterling_ratio(returns, periods_per_year=3)
    assert np.isnan(obtained)


def test_sterling_ratio_all_nan_input_is_nan():
    # n = 0 after dropping nan, so the annualized return is nan
    returns = np.array([np.nan, np.nan])
    obtained = stats.sterling_ratio(returns, periods_per_year=3)
    assert np.isnan(obtained)


# ---------------------------------------------------------------------------
# containers


def test_sterling_ratio_2d_reduces_each_column_independently():
    # col 0: prices 1, 1.1, 0.55, 0.66 -> average drawdown -0.5
    #        (0.66 ** (4 / 3) - 1) / 0.6 = -0.7089420206375693
    # col 1: prices 1, 1.1, 1.32, 1.716 never fall -> average drawdown 0
    #        (1.716 ** (4 / 3) - 1) / 0.10 = 10.544222566153731
    returns = np.array([[0.1, 0.1], [-0.5, 0.2], [0.2, 0.3]])
    obtained = stats.sterling_ratio(returns, periods_per_year=4)
    expected = np.array(
        [(0.66 ** (4 / 3) - 1) / 0.6, (1.716 ** (4 / 3) - 1) / 0.10]
    )
    np.testing.assert_almost_equal(obtained, expected)


def test_sterling_ratio_series_reduces_to_a_float():
    # a 1-d input reduces to a scalar, there is no container left to name
    returns = pd.Series([0.1, -0.5, 0.2], name="asset")
    obtained = stats.sterling_ratio(returns, periods_per_year=4)
    assert isinstance(obtained, float)
    np.testing.assert_almost_equal(obtained, (0.66 ** (4 / 3) - 1) / 0.6)


def test_sterling_ratio_dataframe_returns_series_indexed_by_columns():
    # a: -0.7089420206375693; b: 10.544222566153731, as in the 2d case
    returns = pd.DataFrame(
        {"a": [0.1, -0.5, 0.2], "b": [0.1, 0.2, 0.3]}
    )
    obtained = stats.sterling_ratio(returns, periods_per_year=4)
    expected = pd.Series(
        [(0.66 ** (4 / 3) - 1) / 0.6, (1.716 ** (4 / 3) - 1) / 0.10],
        index=["a", "b"],
    )
    pd.testing.assert_series_equal(obtained, expected)


def test_sterling_ratio_all_nan_dataframe_is_nan_per_column():
    returns = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
    obtained = stats.sterling_ratio(returns, periods_per_year=3)
    expected = pd.Series([np.nan, np.nan], index=["a", "b"])
    pd.testing.assert_series_equal(obtained, expected)
