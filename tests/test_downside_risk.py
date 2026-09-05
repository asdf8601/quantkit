"""Downside risk statistics: deviations, kappa, omega and sortino.

Each test doubles as the specification of the statistic: the expected value
is worked out by hand in a comment next to the data.
"""
from quantkit import stats

import numpy as np
import pandas as pd
import pytest


# Two gains and two losses, used across the module so the results can be
# cross-checked against each other (e.g. omega == 1 + kappa(order=1)).
#
#   r            : 0.1, -0.1, 0.3, -0.2   -> mean = 0.1 / 4 = 0.025
#   gains  (> 0) : 0.1, 0.3   -> sum 0.4, squares 0.01 + 0.09 = 0.10
#   losses (< 0) : 0.1, 0.2   -> sum 0.3, squares 0.01 + 0.04 = 0.05
RETS = np.array([0.1, -0.1, 0.3, -0.2])

# Same data with a NaN at the start and in the middle. Every statistic must
# ignore the NaN both in the sums and in N (N stays 4, not 5).
RETS_WITH_NAN = [
    np.array([np.nan, 0.1, -0.1, 0.3, -0.2]),
    np.array([0.1, -0.1, np.nan, 0.3, -0.2]),
]

# Inputs without a single valid observation: every statistic returns NaN.
NO_VALID_OBS = [
    np.array([]),
    np.array([np.nan, np.nan, np.nan]),
    pd.Series([], dtype=float),
    pd.Series([np.nan, np.nan, np.nan]),
]

ALL_GAINS = np.array([0.1, 0.2, 0.3, 0.4])
ALL_LOSSES = np.array([-0.1, -0.2, -0.3, -0.4])

# http://www.redrockcapital.com/Sortino__A__Sharper__Ratio_Red_Rock_Capital.pdf
# mean = 0.80 / 8 = 0.1
# shortfalls below 0: 0.05, 0.04 -> sqrt((0.0025 + 0.0016) / 8) = 0.0226385
# sortino = 0.1 / 0.0226385 = 4.4172610
RED_ROCK = np.array([0.17, 0.15, 0.23, -0.05, 0.12, 0.09, 0.13, -0.04])
RED_ROCK_SORTINO = 4.417261042993862


# ---------------------------------------------------------------------------
# downside_deviation


def test_downside_deviation_divides_by_all_observations_not_only_losses():
    # shortfalls below 0: 0.1, 0.2 over N=4 -> sqrt((0.01 + 0.04) / 4)
    # = 0.1118034 (dividing by the 2 losses would give sqrt(0.025) = 0.158)
    obtained = stats.downside_deviation(RETS)
    np.testing.assert_almost_equal(obtained, 0.1118034)


def test_downside_deviation_measures_shortfall_below_mar():
    # r - 0.1: 0, -0.2, 0.2, -0.3 -> shortfalls 0.2, 0.3
    # sqrt((0.04 + 0.09) / 4) = sqrt(0.0325) = 0.1802776
    obtained = stats.downside_deviation(RETS, mar=0.1)
    np.testing.assert_almost_equal(obtained, 0.1802776)


def test_downside_deviation_is_zero_when_no_return_is_below_mar():
    obtained = stats.downside_deviation(ALL_GAINS, mar=0.0)
    np.testing.assert_almost_equal(obtained, 0.0)


def test_downside_deviation_factor_multiplies_the_result():
    # 2 * 0.1118034 = 0.2236068
    obtained = stats.downside_deviation(RETS, factor=2)
    np.testing.assert_almost_equal(obtained, 0.2236068)


def test_downside_deviation_default_factor_is_one():
    obtained = stats.downside_deviation(RETS)
    expected = stats.downside_deviation(RETS, factor=1)
    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("rets", RETS_WITH_NAN)
def test_downside_deviation_ignores_nan_in_the_mean_and_in_n(rets):
    # same as RETS: N=4 -> 0.1118034 (counting the NaN as N=5 would give
    # sqrt(0.05 / 5) = 0.1)
    obtained = stats.downside_deviation(rets)
    np.testing.assert_almost_equal(obtained, 0.1118034)


@pytest.mark.parametrize("rets", NO_VALID_OBS)
def test_downside_deviation_empty_or_all_nan_returns_nan(rets):
    obtained = stats.downside_deviation(rets)
    assert np.isnan(obtained)


def test_downside_deviation_2d_reduces_each_column():
    # column 0 = RETS -> 0.1118034 ; column 1 has no losses -> 0
    rets = np.column_stack([RETS, ALL_GAINS])
    obtained = stats.downside_deviation(rets)
    np.testing.assert_almost_equal(obtained, [0.1118034, 0.0])


def test_downside_deviation_of_a_named_series_is_a_float():
    # a reduction of a Series is a scalar: same value as the numpy input
    rets = pd.Series(RETS, name="fund")
    obtained = stats.downside_deviation(rets)
    assert isinstance(obtained, float)
    np.testing.assert_almost_equal(obtained, 0.1118034)


def test_downside_deviation_of_a_dataframe_is_a_series_indexed_by_columns():
    rets = pd.DataFrame(
        np.column_stack([RETS, ALL_GAINS]), columns=["a", "b"]
    )
    obtained = stats.downside_deviation(rets)
    expected = pd.Series([0.1118034, 0.0], index=rets.columns)
    pd.testing.assert_series_equal(obtained, expected, check_exact=False)


# ---------------------------------------------------------------------------
# upside_deviation


def test_upside_deviation_divides_by_all_observations_not_only_gains():
    # gains above 0: 0.1, 0.3 over N=4 -> sqrt((0.01 + 0.09) / 4) = 0.1581139
    obtained = stats.upside_deviation(RETS)
    np.testing.assert_almost_equal(obtained, 0.1581139)


def test_upside_deviation_measures_excess_above_mar():
    # r - 0.1: 0, -0.2, 0.2, -0.3 -> gains 0.2 -> sqrt(0.04 / 4) = 0.1
    obtained = stats.upside_deviation(RETS, mar=0.1)
    np.testing.assert_almost_equal(obtained, 0.1)


def test_upside_deviation_is_zero_when_all_returns_are_below_mar():
    obtained = stats.upside_deviation(ALL_LOSSES, mar=0.0)
    np.testing.assert_almost_equal(obtained, 0.0)


def test_upside_deviation_factor_multiplies_the_result():
    # 2 * 0.1581139 = 0.3162278
    obtained = stats.upside_deviation(RETS, factor=2)
    np.testing.assert_almost_equal(obtained, 0.3162278)


@pytest.mark.parametrize("rets", RETS_WITH_NAN)
def test_upside_deviation_ignores_nan_in_the_mean_and_in_n(rets):
    # same as RETS: N=4 -> 0.1581139 (N=5 would give sqrt(0.02) = 0.1414)
    obtained = stats.upside_deviation(rets)
    np.testing.assert_almost_equal(obtained, 0.1581139)


@pytest.mark.parametrize("rets", NO_VALID_OBS)
def test_upside_deviation_empty_or_all_nan_returns_nan(rets):
    obtained = stats.upside_deviation(rets)
    assert np.isnan(obtained)


def test_upside_deviation_2d_reduces_each_column():
    # column 0 = RETS -> 0.1581139 ; column 1 has no gains -> 0
    rets = np.column_stack([RETS, ALL_LOSSES])
    obtained = stats.upside_deviation(rets)
    np.testing.assert_almost_equal(obtained, [0.1581139, 0.0])


def test_upside_deviation_of_a_named_series_is_a_float():
    rets = pd.Series(RETS, name="fund")
    obtained = stats.upside_deviation(rets)
    assert isinstance(obtained, float)
    np.testing.assert_almost_equal(obtained, 0.1581139)


def test_upside_deviation_of_a_dataframe_is_a_series_indexed_by_columns():
    rets = pd.DataFrame(
        np.column_stack([RETS, ALL_LOSSES]), columns=["a", "b"]
    )
    obtained = stats.upside_deviation(rets)
    expected = pd.Series([0.1581139, 0.0], index=rets.columns)
    pd.testing.assert_series_equal(obtained, expected, check_exact=False)


# ---------------------------------------------------------------------------
# kappa


@pytest.mark.parametrize(
    "order, expected",
    [
        # lpm1 = (0.1 + 0.2) / 4 = 0.075
        # 0.025 / 0.075 = 0.3333333
        (1, 0.3333333),
        # lpm2 = (0.01 + 0.04) / 4 = 0.0125 -> sqrt = 0.1118034
        # 0.025 / 0.1118034 = 0.2236068
        (2, 0.2236068),
        # lpm3 = (0.001 + 0.008) / 4 = 0.00225 -> cbrt = 0.1310371
        # 0.025 / 0.1310371 = 0.1907857
        (3, 0.1907857),
    ],
)
def test_kappa_order_one_two_and_three_on_the_same_data(order, expected):
    obtained = stats.kappa(RETS, mar=0.0, order=order)
    np.testing.assert_almost_equal(obtained, expected)


def test_kappa_measures_excess_over_mar():
    # r - 0.1: 0, -0.2, 0.2, -0.3 -> mean = -0.3 / 4 = -0.075
    # lpm2 = (0.04 + 0.09) / 4 = 0.0325 -> sqrt = 0.1802776
    # kappa = -0.075 / 0.1802776 = -0.4160251
    obtained = stats.kappa(RETS, mar=0.1, order=2)
    np.testing.assert_almost_equal(obtained, -0.4160251)


def test_kappa_is_nan_when_no_return_is_below_mar():
    # lpm == 0 -> zero denominator -> NaN, never inf
    obtained = stats.kappa(ALL_GAINS, mar=0.0)
    assert np.isnan(obtained)


def test_kappa_is_zero_when_mar_equals_the_mean_return():
    # mean(RETS) = 0.025 -> numerator 0 ; shortfalls 0.125, 0.225 -> lpm > 0
    obtained = stats.kappa(RETS, mar=0.025)
    np.testing.assert_almost_equal(obtained, 0.0)


def test_kappa_factor_multiplies_the_result():
    # 2 * 0.2236068 = 0.4472136
    obtained = stats.kappa(RETS, order=2, factor=2)
    np.testing.assert_almost_equal(obtained, 0.4472136)


@pytest.mark.parametrize("order", [0, -1])
def test_kappa_raises_for_non_positive_order(order):
    with pytest.raises(ValueError):
        stats.kappa(RETS, order=order)


@pytest.mark.parametrize("rets", RETS_WITH_NAN)
def test_kappa_ignores_nan_in_the_mean_and_in_n(rets):
    # same as RETS with order 2 -> 0.2236068
    obtained = stats.kappa(rets, order=2)
    np.testing.assert_almost_equal(obtained, 0.2236068)


@pytest.mark.parametrize("rets", NO_VALID_OBS)
def test_kappa_empty_or_all_nan_returns_nan(rets):
    obtained = stats.kappa(rets)
    assert np.isnan(obtained)


def test_kappa_2d_reduces_each_column():
    # column 0 = RETS -> 0.2236068 ; column 1 has no losses -> NaN
    rets = np.column_stack([RETS, ALL_GAINS])
    obtained = stats.kappa(rets, order=2)
    np.testing.assert_almost_equal(obtained, [0.2236068, np.nan])


def test_kappa_of_a_named_series_is_a_float():
    rets = pd.Series(RETS, name="fund")
    obtained = stats.kappa(rets, order=2)
    assert isinstance(obtained, float)
    np.testing.assert_almost_equal(obtained, 0.2236068)


def test_kappa_of_a_dataframe_is_a_series_indexed_by_columns():
    rets = pd.DataFrame(
        np.column_stack([RETS, ALL_GAINS]), columns=["a", "b"]
    )
    obtained = stats.kappa(rets, order=2)
    expected = pd.Series([0.2236068, np.nan], index=rets.columns)
    pd.testing.assert_series_equal(obtained, expected, check_exact=False)


# ---------------------------------------------------------------------------
# omega_ratio


def test_omega_ratio_is_sum_of_gains_over_sum_of_losses():
    # gains 0.1 + 0.3 = 0.4 ; losses 0.1 + 0.2 = 0.3 -> 0.4 / 0.3 = 1.3333333
    obtained = stats.omega_ratio(RETS)
    np.testing.assert_almost_equal(obtained, 1.3333333)


def test_omega_ratio_measures_gains_and_losses_relative_to_mar():
    # r - 0.1: 0, -0.2, 0.2, -0.3 -> gains 0.2 ; losses 0.2 + 0.3 = 0.5
    # 0.2 / 0.5 = 0.4
    obtained = stats.omega_ratio(RETS, mar=0.1)
    np.testing.assert_almost_equal(obtained, 0.4)


def test_omega_ratio_is_nan_when_no_return_is_below_mar():
    # sum of losses == 0 -> zero denominator -> NaN, never inf
    obtained = stats.omega_ratio(ALL_GAINS)
    assert np.isnan(obtained)


def test_omega_ratio_is_zero_when_all_returns_are_below_mar():
    # gains 0 / losses 1.0 = 0
    obtained = stats.omega_ratio(ALL_LOSSES)
    np.testing.assert_almost_equal(obtained, 0.0)


@pytest.mark.parametrize("rets", [RETS, RED_ROCK])
def test_omega_ratio_equals_one_plus_kappa_of_order_one(rets):
    # sum(gains) / sum(losses) = (mean(r - mar) + lpm1) / lpm1 = kappa1 + 1
    obtained = stats.omega_ratio(rets, mar=0.02)
    expected = 1 + stats.kappa(rets, mar=0.02, order=1)
    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("rets", RETS_WITH_NAN)
def test_omega_ratio_ignores_nan(rets):
    # same as RETS -> 1.3333333
    obtained = stats.omega_ratio(rets)
    np.testing.assert_almost_equal(obtained, 1.3333333)


@pytest.mark.parametrize("rets", NO_VALID_OBS)
def test_omega_ratio_empty_or_all_nan_returns_nan(rets):
    # both sums are 0 -> 0 / 0 -> NaN
    obtained = stats.omega_ratio(rets)
    assert np.isnan(obtained)


def test_omega_ratio_2d_reduces_each_column():
    # column 0 = RETS -> 1.3333333 ; column 1 has no losses -> NaN
    rets = np.column_stack([RETS, ALL_GAINS])
    obtained = stats.omega_ratio(rets)
    np.testing.assert_almost_equal(obtained, [1.3333333, np.nan])


def test_omega_ratio_of_a_named_series_is_a_float():
    rets = pd.Series(RETS, name="fund")
    obtained = stats.omega_ratio(rets)
    assert isinstance(obtained, float)
    np.testing.assert_almost_equal(obtained, 1.3333333)


def test_omega_ratio_of_a_dataframe_is_a_series_indexed_by_columns():
    rets = pd.DataFrame(
        np.column_stack([RETS, ALL_GAINS]), columns=["a", "b"]
    )
    obtained = stats.omega_ratio(rets)
    expected = pd.Series([1.3333333, np.nan], index=rets.columns)
    pd.testing.assert_series_equal(obtained, expected, check_exact=False)


# ---------------------------------------------------------------------------
# sortino_ratio


def test_sortino_ratio_reproduces_the_red_rock_capital_example():
    obtained = stats.sortino_ratio(RED_ROCK, mar=0.0, factor=None)
    np.testing.assert_almost_equal(obtained, RED_ROCK_SORTINO)


def test_sortino_ratio_of_constant_losses_is_minus_one():
    # mean = -10 ; downside deviation = sqrt(mean(100)) = 10 -> -10 / 10 = -1
    rets = np.array([-10.0, -10.0, -10.0, -10.0])
    obtained = stats.sortino_ratio(rets, mar=0.0)
    np.testing.assert_almost_equal(obtained, -1.0)


def test_sortino_ratio_is_mean_excess_over_downside_deviation():
    # mean 0.025 / downside deviation 0.1118034 = 0.2236068
    obtained = stats.sortino_ratio(RETS)
    np.testing.assert_almost_equal(obtained, 0.2236068)
    expected = np.mean(RETS) / stats.downside_deviation(RETS)
    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("rets", [RETS, RED_ROCK])
def test_sortino_ratio_equals_kappa_of_order_two(rets):
    obtained = stats.sortino_ratio(rets, mar=0.02)
    expected = stats.kappa(rets, mar=0.02, order=2)
    np.testing.assert_almost_equal(obtained, expected)


def test_sortino_ratio_is_zero_when_mar_equals_the_mean_return():
    # mean(RED_ROCK) = 0.1 -> numerator 0
    obtained = stats.sortino_ratio(RED_ROCK, mar=0.1)
    np.testing.assert_almost_equal(obtained, 0.0)


def test_sortino_ratio_is_nan_when_no_return_is_below_mar():
    obtained = stats.sortino_ratio(ALL_GAINS)
    assert np.isnan(obtained)


def test_sortino_ratio_factor_multiplies_the_result():
    # 2 * 4.4172610 = 8.8345221
    obtained = stats.sortino_ratio(RED_ROCK, factor=2)
    np.testing.assert_almost_equal(obtained, 2 * RED_ROCK_SORTINO)


@pytest.mark.parametrize(
    "rets",
    [
        np.insert(RED_ROCK, 0, np.nan),
        np.insert(RED_ROCK, 4, np.nan),
    ],
)
def test_sortino_ratio_ignores_nan_in_the_mean_and_in_n(rets):
    obtained = stats.sortino_ratio(rets)
    np.testing.assert_almost_equal(obtained, RED_ROCK_SORTINO)


@pytest.mark.parametrize("rets", NO_VALID_OBS)
def test_sortino_ratio_empty_or_all_nan_returns_nan(rets):
    obtained = stats.sortino_ratio(rets)
    assert np.isnan(obtained)


def test_sortino_ratio_2d_reduces_each_column():
    # column 0 = RED_ROCK -> 4.4172610 ; column 1 constant -10 -> -1
    rets = np.column_stack([RED_ROCK, np.full(8, -10.0)])
    obtained = stats.sortino_ratio(rets)
    np.testing.assert_almost_equal(obtained, [RED_ROCK_SORTINO, -1.0])


def test_sortino_ratio_of_a_named_series_is_a_float():
    rets = pd.Series(RED_ROCK, name="fund")
    obtained = stats.sortino_ratio(rets)
    assert isinstance(obtained, float)
    np.testing.assert_almost_equal(obtained, RED_ROCK_SORTINO)


def test_sortino_ratio_of_a_dataframe_is_a_series_indexed_by_columns():
    rets = pd.DataFrame(
        np.column_stack([RED_ROCK, np.full(8, -10.0)]), columns=["a", "b"]
    )
    obtained = stats.sortino_ratio(rets)
    expected = pd.Series([RED_ROCK_SORTINO, -1.0], index=rets.columns)
    pd.testing.assert_series_equal(obtained, expected, check_exact=False)
