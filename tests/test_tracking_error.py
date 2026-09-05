"""Tests of the tracking error and the information ratio.

Both statistics summarise the *active return* ``r - b``, the difference
between the fund and its benchmark: the tracking error is its sample
standard deviation and the information ratio its mean over that deviation.
Inputs follow the benchmark contract of the module: ``returns`` is 1D or 2D
(numpy or pandas), ``benchmark`` is 1D, and reductions are column-wise on
the rows where both the column and the benchmark are non-NaN.
"""
from quantkit import stats

import pytest
import numpy as np
import pandas as pd


# Six observations shared by the contract tests below.
BENCH = np.array([0.02, -0.01, 0.03, -0.02, 0.01, -0.03])

# FUND - BENCH = 0.04, -0.02, 0.02, 0.00, 0.01, 0.01
#   -> mean 0.01; deviations 0.03, -0.03, 0.01, -0.01, 0.00, 0.00
#   -> sum of squares 0.002, / (6 - 1) -> variance 0.0004
#   -> tracking error 0.02, information ratio 0.01 / 0.02 = 0.5
FUND = np.array([0.06, -0.03, 0.05, -0.02, 0.02, -0.02])

# LAGGARD - BENCH = 0.05, -0.07, 0.01, -0.03, -0.01, -0.01
#   -> mean -0.01; deviations 0.06, -0.06, 0.02, -0.02, 0.00, 0.00
#   -> sum of squares 0.008, / (6 - 1) -> variance 0.0016
#   -> tracking error 0.04, information ratio -0.01 / 0.04 = -0.25
LAGGARD = np.array([0.07, -0.08, 0.04, -0.05, 0.00, -0.04])

TRACKING_FUNCTIONS = [
    "tracking_error",
    "information_ratio",
]

# Expected statistic of FUND against BENCH, per function.
FUND_EXPECTED = [
    ("tracking_error", 0.02),
    ("information_ratio", 0.5),
]

# Expected statistic of the two columns [FUND, LAGGARD] against BENCH.
TWO_COLUMN_EXPECTED = [
    ("tracking_error", [0.02, 0.04]),
    ("information_ratio", [0.5, -0.25]),
]

# Both statistics are multiplied by the factor: 0.02 * 2 and 0.5 * 2.
FACTOR_EXPECTED = [
    ("tracking_error", 0.04),
    ("information_ratio", 1.0),
]


# ---------------------------------------------------------------------------
# Shared contract of both functions


@pytest.mark.parametrize("name, expected", FUND_EXPECTED)
def test_series_returns_reduce_to_a_float(name, expected):
    returns = pd.Series(FUND, name="fund")
    benchmark = pd.Series(BENCH, name="index")

    obtained = getattr(stats, name)(returns, benchmark)

    assert isinstance(obtained, float)
    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", FUND_EXPECTED)
def test_one_dimensional_numpy_returns_reduce_to_a_float(name, expected):
    obtained = getattr(stats, name)(FUND, BENCH)

    assert np.ndim(obtained) == 0
    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", TWO_COLUMN_EXPECTED)
def test_two_column_numpy_returns_reduce_to_one_value_per_column(
    name, expected
):
    returns = np.column_stack([FUND, LAGGARD])

    obtained = getattr(stats, name)(returns, BENCH)

    assert isinstance(obtained, np.ndarray)
    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", TWO_COLUMN_EXPECTED)
def test_dataframe_returns_reduce_to_a_series_indexed_by_columns(
    name, expected
):
    returns = pd.DataFrame({"fund": FUND, "laggard": LAGGARD})
    benchmark = pd.Series(BENCH)

    obtained = getattr(stats, name)(returns, benchmark)

    expected = pd.Series(expected, index=["fund", "laggard"])
    pd.testing.assert_series_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", FUND_EXPECTED)
def test_nan_at_the_start_of_returns_is_dropped(name, expected):
    # the first row is incomplete; on the other three the active return is
    # 0.01, -0.01, 0.03 -> mean 0.01, std(ddof=1) 0.02 -> IR 0.5
    returns = np.array([np.nan, 0.02, 0.00, 0.05])
    benchmark = np.array([0.05, 0.01, 0.01, 0.02])

    obtained = getattr(stats, name)(returns, benchmark)

    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", FUND_EXPECTED)
def test_nan_in_the_middle_of_returns_is_dropped(name, expected):
    # active on the complete rows: 0.01, -0.01, 0.03 -> TE 0.02, IR 0.5
    returns = np.array([0.02, np.nan, 0.00, 0.05])
    benchmark = np.array([0.01, 0.05, 0.01, 0.02])

    obtained = getattr(stats, name)(returns, benchmark)

    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", FUND_EXPECTED)
def test_nan_in_the_benchmark_drops_that_row(name, expected):
    # the return on the NaN benchmark row is wild: it must be ignored, so
    # the active return is again 0.01, -0.01, 0.03 -> TE 0.02, IR 0.5
    returns = np.array([0.02, 0.99, 0.00, 0.05])
    benchmark = np.array([0.01, np.nan, 0.01, 0.02])

    obtained = getattr(stats, name)(returns, benchmark)

    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", TWO_COLUMN_EXPECTED)
def test_nan_in_one_column_drops_the_row_only_for_that_column(
    name, expected
):
    benchmark = np.array([0.01, 0.05, 0.01, 0.02])
    returns = np.column_stack(
        [
            # active on rows 0, 2, 3: 0.01, -0.01, 0.03 -> mean 0.01,
            # variance 0.0008 / 2 = 0.0004 -> TE 0.02, IR 0.5
            [0.02, np.nan, 0.00, 0.05],
            # active on the four rows: 0.05, -0.03, -0.03, -0.03 -> mean
            # -0.01, variance 0.0048 / 3 = 0.0016 -> TE 0.04, IR -0.25
            [0.06, 0.02, -0.02, -0.01],
        ]
    )

    obtained = getattr(stats, name)(returns, benchmark)

    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", FUND_EXPECTED)
def test_misaligned_pandas_indexes_use_the_overlap(name, expected):
    dates = pd.date_range("2020-01-01", periods=8)
    # benchmark is known on d0..d6 and returns on d1..d7; the overlap
    # d1..d6 carries BENCH and FUND. The values outside the overlap are
    # wild so that a positional match would be visibly wrong.
    benchmark = pd.Series(np.append(1.0, BENCH), index=dates[:7])
    returns = pd.Series(np.append(FUND, -1.0), index=dates[1:])

    obtained = getattr(stats, name)(returns, benchmark)

    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", FACTOR_EXPECTED)
def test_factor_multiplies_the_result(name, expected):
    obtained = getattr(stats, name)(FUND, BENCH, factor=2)

    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", FUND_EXPECTED)
def test_factor_none_leaves_the_result_unchanged(name, expected):
    obtained = getattr(stats, name)(FUND, BENCH, factor=None)

    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", TWO_COLUMN_EXPECTED)
def test_factor_applies_to_every_column(name, expected):
    returns = pd.DataFrame({"fund": FUND, "laggard": LAGGARD})
    benchmark = pd.Series(BENCH)

    obtained = getattr(stats, name)(returns, benchmark, factor=2)

    expected = pd.Series(
        [2 * value for value in expected], index=["fund", "laggard"]
    )
    pd.testing.assert_series_equal(obtained, expected)


@pytest.mark.parametrize("name", TRACKING_FUNCTIONS)
def test_empty_numpy_returns_give_nan(name):
    obtained = getattr(stats, name)(np.array([]), np.array([]))

    assert np.isnan(obtained)


@pytest.mark.parametrize("name", TRACKING_FUNCTIONS)
def test_empty_series_returns_give_nan(name):
    returns = pd.Series([], dtype=float)
    benchmark = pd.Series([], dtype=float)

    obtained = getattr(stats, name)(returns, benchmark)

    assert np.isnan(obtained)


@pytest.mark.parametrize("name", TRACKING_FUNCTIONS)
def test_all_nan_returns_give_nan(name):
    returns = np.array([np.nan, np.nan, np.nan, np.nan])

    obtained = getattr(stats, name)(returns, BENCH[:4])

    assert np.isnan(obtained)


@pytest.mark.parametrize("name", TRACKING_FUNCTIONS)
def test_all_nan_dataframe_gives_nan_per_column(name):
    returns = pd.DataFrame(
        np.full((4, 2), np.nan), columns=["fund", "laggard"]
    )
    benchmark = pd.Series(BENCH[:4])

    obtained = getattr(stats, name)(returns, benchmark)

    expected = pd.Series([np.nan, np.nan], index=["fund", "laggard"])
    pd.testing.assert_series_equal(obtained, expected)


@pytest.mark.parametrize("name", TRACKING_FUNCTIONS)
def test_a_single_complete_row_gives_nan(name):
    # only the last row is complete: no deviation can be estimated
    returns = np.array([np.nan, np.nan, 0.05])
    benchmark = np.array([0.02, -0.01, 0.03])

    obtained = getattr(stats, name)(returns, benchmark)

    assert np.isnan(obtained)


@pytest.mark.parametrize("name", TRACKING_FUNCTIONS)
def test_numpy_of_different_length_is_rejected(name):
    returns = np.array([0.01, 0.02, 0.03])
    benchmark = np.array([0.02, -0.01])

    with pytest.raises(ValueError, match="length"):
        getattr(stats, name)(returns, benchmark)


@pytest.mark.parametrize("name", TRACKING_FUNCTIONS)
def test_two_dimensional_numpy_benchmark_is_rejected(name):
    returns = np.array([0.01, 0.02, 0.03])
    benchmark = np.array([[0.02], [-0.01], [0.03]])

    with pytest.raises(ValueError, match="1D"):
        getattr(stats, name)(returns, benchmark)


@pytest.mark.parametrize("name", TRACKING_FUNCTIONS)
def test_dataframe_benchmark_is_rejected(name):
    returns = pd.Series([0.01, 0.02, 0.03])
    benchmark = pd.DataFrame([[0.02], [-0.01], [0.03]])

    with pytest.raises(ValueError, match="1D"):
        getattr(stats, name)(returns, benchmark)


# ---------------------------------------------------------------------------
# tracking_error


def test_tracking_error_is_the_deviation_of_the_active_return():
    # active: 0.01, -0.01, 0.03 -> mean 0.01, deviations 0, -0.02, 0.02
    # -> sum of squares 0.0008, / (3 - 1) -> variance 0.0004 -> TE 0.02
    returns = np.array([0.02, 0.00, 0.05])
    benchmark = np.array([0.01, 0.01, 0.02])

    obtained = stats.tracking_error(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 0.02)


def test_tracking_error_uses_one_degree_of_freedom():
    # the same active return divided by 3 instead of 2 would give
    # sqrt(0.0008 / 3) = 0.01633, so the sample deviation is the larger one
    returns = np.array([0.02, 0.00, 0.05])
    benchmark = np.array([0.01, 0.01, 0.02])

    obtained = stats.tracking_error(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 0.02)
    assert obtained > np.sqrt(0.0008 / 3)


def test_tracking_error_is_zero_when_returns_equal_the_benchmark():
    obtained = stats.tracking_error(BENCH, BENCH)

    assert obtained == 0.0


def test_tracking_error_is_zero_for_a_constant_active_return():
    # r = b + 0.01 -> the active return never moves, so there is nothing to
    # track: the deviation is exactly zero, not the float noise of b + 0.01
    returns = BENCH + 0.01

    obtained = stats.tracking_error(returns, BENCH)

    assert obtained == 0.0


def test_tracking_error_of_double_the_benchmark_is_its_own_deviation():
    # r = 2b -> active return = b = 0.01, 0.02, 0.03 -> mean 0.02,
    # deviations -0.01, 0, 0.01 -> variance 0.0002 / 2 = 0.0001 -> 0.01
    benchmark = np.array([0.01, 0.02, 0.03])
    returns = 2 * benchmark

    obtained = stats.tracking_error(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 0.01)
    np.testing.assert_almost_equal(obtained, np.std(benchmark, ddof=1))


def test_tracking_error_of_a_laggard_is_positive():
    obtained = stats.tracking_error(LAGGARD, BENCH)

    np.testing.assert_almost_equal(obtained, 0.04)


def test_tracking_error_is_zero_only_for_the_column_that_tracks_exactly():
    returns = pd.DataFrame({"fund": FUND, "clone": BENCH + 0.01})
    benchmark = pd.Series(BENCH)

    obtained = stats.tracking_error(returns, benchmark)

    expected = pd.Series([0.02, 0.0], index=["fund", "clone"])
    pd.testing.assert_series_equal(obtained, expected)


def test_tracking_error_annualizes_with_the_square_root_of_the_periods():
    # a deviation is annualized by sqrt(periods per year), here 4 quarters
    obtained = stats.tracking_error(FUND, BENCH, factor=np.sqrt(4))

    np.testing.assert_almost_equal(obtained, 0.04)


# ---------------------------------------------------------------------------
# information_ratio


def test_information_ratio_is_the_mean_active_return_over_its_deviation():
    # active: 0.01, -0.01, 0.03 -> mean 0.01, std(ddof=1) 0.02 -> IR 0.5
    returns = np.array([0.02, 0.00, 0.05])
    benchmark = np.array([0.01, 0.01, 0.02])

    obtained = stats.information_ratio(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 0.5)


def test_information_ratio_of_double_the_benchmark():
    # r = 2b -> active return = b = 0.01, 0.02, 0.03 -> mean 0.02,
    # std(ddof=1) 0.01 -> IR 2.0
    benchmark = np.array([0.01, 0.02, 0.03])
    returns = 2 * benchmark

    obtained = stats.information_ratio(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 2.0)


def test_information_ratio_is_negative_when_the_fund_lags():
    # LAGGARD active mean -0.01 over a tracking error of 0.04 -> -0.25
    obtained = stats.information_ratio(LAGGARD, BENCH)

    np.testing.assert_almost_equal(obtained, -0.25)


def test_information_ratio_is_nan_when_returns_equal_the_benchmark():
    # active return is zero everywhere: 0 / 0 is undefined, not inf
    obtained = stats.information_ratio(BENCH, BENCH)

    assert np.isnan(obtained)


def test_information_ratio_is_nan_for_a_constant_positive_active_return():
    # r = b + 0.01 beats the benchmark every period with a zero tracking
    # error: the ratio is undefined, and must not come out as inf or 1e16
    returns = BENCH + 0.01

    obtained = stats.information_ratio(returns, BENCH)

    assert np.isnan(obtained)


def test_information_ratio_is_nan_for_a_constant_negative_active_return():
    returns = BENCH - 0.01

    obtained = stats.information_ratio(returns, BENCH)

    assert np.isnan(obtained)


def test_information_ratio_is_nan_only_for_the_column_that_tracks_exactly():
    returns = pd.DataFrame({"fund": FUND, "clone": BENCH + 0.01})
    benchmark = pd.Series(BENCH)

    obtained = stats.information_ratio(returns, benchmark)

    expected = pd.Series([0.5, np.nan], index=["fund", "clone"])
    pd.testing.assert_series_equal(obtained, expected)


def test_information_ratio_is_the_mean_active_return_over_tracking_error():
    mean_active = np.mean(FUND - BENCH)
    tracking_error = stats.tracking_error(FUND, BENCH)

    obtained = stats.information_ratio(FUND, BENCH)

    np.testing.assert_almost_equal(obtained, mean_active / tracking_error)
