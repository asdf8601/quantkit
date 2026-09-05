"""Tests of the statistics computed against a benchmark.

Every function receives ``(returns, benchmark)`` where ``returns`` is 1D or
2D (numpy or pandas) and ``benchmark`` is 1D. Reductions are column-wise on
the rows where both the column and the benchmark are non-NaN.
"""
from quantkit import stats, utils

import pytest
import numpy as np
import pandas as pd


# Six observations with three positive and three negative benchmark values,
# so bull and bear beta both keep at least two rows after dropping one NaN.
BENCH = np.array([0.02, -0.01, 0.03, -0.02, 0.01, -0.03])

# LINEAR = 2 * BENCH + 0.01 -> beta 2, alpha 0.01, correlation 1
LINEAR = np.array([0.05, -0.01, 0.07, -0.03, 0.03, -0.05])

# INVERTED = -BENCH -> beta -1, alpha 0, correlation -1
INVERTED = np.array([-0.02, 0.01, -0.03, 0.02, -0.01, 0.03])

BENCHMARK_FUNCTIONS = [
    "beta",
    "alpha",
    "correlation",
    "r_squared",
    "bull_beta",
    "bear_beta",
]

# Expected statistic of LINEAR against BENCH, per function.
LINEAR_EXPECTED = [
    ("beta", 2.0),
    ("alpha", 0.01),
    ("correlation", 1.0),
    ("r_squared", 1.0),
    ("bull_beta", 2.0),
    ("bear_beta", 2.0),
]

# Expected statistic of the two columns [LINEAR, INVERTED] against BENCH.
TWO_COLUMN_EXPECTED = [
    ("beta", [2.0, -1.0]),
    ("alpha", [0.01, 0.0]),
    ("correlation", [1.0, -1.0]),
    ("r_squared", [1.0, 1.0]),
    ("bull_beta", [2.0, -1.0]),
    ("bear_beta", [2.0, -1.0]),
]


# ---------------------------------------------------------------------------
# utils.align


def test_align_returns_numpy_inputs_as_ndarrays():
    returns = np.array([0.1, 0.2, 0.3])
    benchmark = np.array([0.4, 0.5, 0.6])

    arr_returns, arr_benchmark = utils.align(returns, benchmark)

    assert isinstance(arr_returns, np.ndarray)
    assert isinstance(arr_benchmark, np.ndarray)
    np.testing.assert_array_equal(arr_returns, returns)
    np.testing.assert_array_equal(arr_benchmark, benchmark)


def test_align_keeps_two_dimensional_returns():
    returns = np.array([[0.1, 1.0], [0.2, 2.0], [0.3, 3.0]])
    benchmark = np.array([0.4, 0.5, 0.6])

    arr_returns, arr_benchmark = utils.align(returns, benchmark)

    assert arr_returns.shape == (3, 2)
    assert arr_benchmark.shape == (3,)


def test_align_inner_joins_pandas_series_on_the_index():
    dates = pd.date_range("2020-01-01", periods=6)
    # returns is known on d1..d5, benchmark on d0..d4: the overlap is d1..d4
    returns = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=dates[1:])
    benchmark = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=dates[:5])

    arr_returns, arr_benchmark = utils.align(returns, benchmark)

    np.testing.assert_array_equal(arr_returns, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(arr_benchmark, [20.0, 30.0, 40.0, 50.0])


def test_align_inner_joins_dataframe_and_series_on_the_index():
    dates = pd.date_range("2020-01-01", periods=6)
    returns = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [5.0, 4.0, 3.0, 2.0, 1.0]},
        index=dates[1:],
    )
    benchmark = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=dates[:5])

    arr_returns, arr_benchmark = utils.align(returns, benchmark)

    np.testing.assert_array_equal(
        arr_returns, [[1.0, 5.0], [2.0, 4.0], [3.0, 3.0], [4.0, 2.0]]
    )
    np.testing.assert_array_equal(arr_benchmark, [20.0, 30.0, 40.0, 50.0])


def test_align_matches_pandas_and_numpy_by_position():
    # the pandas index is ignored when the other side has none
    returns = pd.Series([1.0, 2.0, 3.0], index=[10, 11, 12])
    benchmark = np.array([4.0, 5.0, 6.0])

    arr_returns, arr_benchmark = utils.align(returns, benchmark)

    np.testing.assert_array_equal(arr_returns, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(arr_benchmark, [4.0, 5.0, 6.0])


def test_align_raises_on_numpy_of_different_length():
    returns = np.array([0.1, 0.2, 0.3])
    benchmark = np.array([0.4, 0.5])

    with pytest.raises(ValueError, match="length"):
        utils.align(returns, benchmark)


def test_align_raises_on_pandas_and_numpy_of_different_length():
    returns = pd.Series([0.1, 0.2, 0.3])
    benchmark = np.array([0.4, 0.5])

    with pytest.raises(ValueError, match="length"):
        utils.align(returns, benchmark)


def test_align_raises_on_two_dimensional_numpy_benchmark():
    returns = np.array([0.1, 0.2, 0.3])
    benchmark = np.array([[0.4], [0.5], [0.6]])

    with pytest.raises(ValueError, match="1D"):
        utils.align(returns, benchmark)


def test_align_raises_on_dataframe_benchmark():
    returns = pd.Series([0.1, 0.2, 0.3])
    benchmark = pd.DataFrame([[0.4], [0.5], [0.6]])

    with pytest.raises(ValueError, match="1D"):
        utils.align(returns, benchmark)


# ---------------------------------------------------------------------------
# Shared contract of every benchmark function


@pytest.mark.parametrize("name, expected", LINEAR_EXPECTED)
def test_series_returns_reduce_to_a_float(name, expected):
    returns = pd.Series(LINEAR, name="fund")
    benchmark = pd.Series(BENCH, name="index")

    obtained = getattr(stats, name)(returns, benchmark)

    assert isinstance(obtained, float)
    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", TWO_COLUMN_EXPECTED)
def test_two_column_numpy_returns_reduce_to_one_value_per_column(
    name, expected
):
    returns = np.column_stack([LINEAR, INVERTED])

    obtained = getattr(stats, name)(returns, BENCH)

    assert isinstance(obtained, np.ndarray)
    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", TWO_COLUMN_EXPECTED)
def test_dataframe_returns_reduce_to_a_series_indexed_by_columns(
    name, expected
):
    returns = pd.DataFrame({"fund": LINEAR, "short": INVERTED})
    benchmark = pd.Series(BENCH)

    obtained = getattr(stats, name)(returns, benchmark)

    expected = pd.Series(expected, index=["fund", "short"])
    pd.testing.assert_series_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", LINEAR_EXPECTED)
def test_nan_at_the_start_of_returns_is_dropped(name, expected):
    returns = LINEAR.copy()
    returns[0] = np.nan

    obtained = getattr(stats, name)(returns, BENCH)

    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", LINEAR_EXPECTED)
def test_nan_in_the_middle_of_returns_is_dropped(name, expected):
    returns = LINEAR.copy()
    returns[3] = np.nan

    obtained = getattr(stats, name)(returns, BENCH)

    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", LINEAR_EXPECTED)
def test_nan_in_the_benchmark_drops_that_row(name, expected):
    # the returns value on the NaN benchmark row is wild: it must be ignored
    returns = LINEAR.copy()
    returns[2] = 100.0
    benchmark = BENCH.copy()
    benchmark[2] = np.nan

    obtained = getattr(stats, name)(returns, benchmark)

    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", TWO_COLUMN_EXPECTED)
def test_nan_in_one_column_drops_the_row_only_for_that_column(
    name, expected
):
    returns = np.column_stack([LINEAR, INVERTED])
    returns[2, 0] = np.nan  # only the first column loses a row

    obtained = getattr(stats, name)(returns, BENCH)

    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", LINEAR_EXPECTED)
def test_misaligned_pandas_indexes_use_the_overlap(name, expected):
    dates = pd.date_range("2020-01-01", periods=8)
    # benchmark is known on d0..d6 and returns on d1..d7; on the overlap
    # d1..d6 returns = 2 * benchmark + 0.01. The values outside the overlap
    # are wild so that a positional match would be visibly wrong.
    benchmark = pd.Series(np.append(1.0, BENCH), index=dates[:7])
    returns = pd.Series(np.append(LINEAR, -1.0), index=dates[1:])

    obtained = getattr(stats, name)(returns, benchmark)

    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("name", BENCHMARK_FUNCTIONS)
def test_empty_numpy_returns_give_nan(name):
    obtained = getattr(stats, name)(np.array([]), np.array([]))

    assert np.isnan(obtained)


@pytest.mark.parametrize("name", BENCHMARK_FUNCTIONS)
def test_empty_series_returns_give_nan(name):
    returns = pd.Series([], dtype=float)
    benchmark = pd.Series([], dtype=float)

    obtained = getattr(stats, name)(returns, benchmark)

    assert np.isnan(obtained)


@pytest.mark.parametrize("name", BENCHMARK_FUNCTIONS)
def test_all_nan_returns_give_nan(name):
    returns = np.array([np.nan, np.nan, np.nan, np.nan])

    obtained = getattr(stats, name)(returns, BENCH[:4])

    assert np.isnan(obtained)


@pytest.mark.parametrize("name", BENCHMARK_FUNCTIONS)
def test_all_nan_dataframe_gives_nan_per_column(name):
    returns = pd.DataFrame(
        np.full((4, 2), np.nan), columns=["fund", "short"]
    )
    benchmark = pd.Series(BENCH[:4])

    obtained = getattr(stats, name)(returns, benchmark)

    expected = pd.Series([np.nan, np.nan], index=["fund", "short"])
    pd.testing.assert_series_equal(obtained, expected)


@pytest.mark.parametrize("name", BENCHMARK_FUNCTIONS)
def test_a_single_complete_row_gives_nan(name):
    # only the last row is complete: no variance can be estimated
    returns = np.array([np.nan, np.nan, 0.05])
    benchmark = np.array([0.02, -0.01, 0.03])

    obtained = getattr(stats, name)(returns, benchmark)

    assert np.isnan(obtained)


@pytest.mark.parametrize("name", BENCHMARK_FUNCTIONS)
def test_constant_benchmark_gives_nan(name):
    # np.var of three 0.05 is 7e-35 rather than 0 because the mean is
    # rounded, so the zero-variance check must not trust the computed
    # variance: a constant benchmark has to give NaN, not 1e30.
    returns = np.array([0.01, 0.02, 0.03])
    benchmark = np.array([0.05, 0.05, 0.05])

    obtained = getattr(stats, name)(returns, benchmark)

    assert np.isnan(obtained)


@pytest.mark.parametrize("name", BENCHMARK_FUNCTIONS)
def test_numpy_of_different_length_is_rejected(name):
    returns = np.array([0.01, 0.02, 0.03])
    benchmark = np.array([0.02, -0.01])

    with pytest.raises(ValueError, match="length"):
        getattr(stats, name)(returns, benchmark)


@pytest.mark.parametrize("name", BENCHMARK_FUNCTIONS)
def test_two_dimensional_numpy_benchmark_is_rejected(name):
    returns = np.array([0.01, 0.02, 0.03])
    benchmark = np.array([[0.02], [-0.01], [0.03]])

    with pytest.raises(ValueError, match="1D"):
        getattr(stats, name)(returns, benchmark)


@pytest.mark.parametrize("name", BENCHMARK_FUNCTIONS)
def test_dataframe_benchmark_is_rejected(name):
    returns = pd.Series([0.01, 0.02, 0.03])
    benchmark = pd.DataFrame([[0.02], [-0.01], [0.03]])

    with pytest.raises(ValueError, match="1D"):
        getattr(stats, name)(returns, benchmark)


# ---------------------------------------------------------------------------
# beta


def test_beta_reproduces_the_reference_value():
    returns = np.array([0.11, 0.17, 0.21, 0.18, -0.08, -0.12])
    benchmark = np.array([0.08, 0.10, 0.13, 0.11, -0.03, -0.05])

    obtained = stats.beta(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 1.8510158013544014)


def test_beta_reproduces_the_reference_value_for_a_series():
    returns = pd.Series([0.11, 0.17, 0.21, 0.18, -0.08, -0.12])
    benchmark = pd.Series([0.08, 0.10, 0.13, 0.11, -0.03, -0.05])

    obtained = stats.beta(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 1.8510158013544014)


def test_beta_reproduces_the_reference_values_for_two_columns():
    returns = np.array(
        [
            [0.5, 0.2],
            [0.2, 0.4],
            [-0.3, 0.6],
            [-0.1, 0.1],
            [0.8, 0.67],
            [0.9, 0.43],
        ]
    )
    benchmark = np.array([0.08, 0.10, 0.13, 0.11, -0.03, -0.05])

    obtained = stats.beta(returns, benchmark)

    expected = [-5.835214446952595, -1.1038374717832957]
    np.testing.assert_almost_equal(obtained, expected)


def test_beta_reproduces_the_reference_values_for_a_dataframe():
    returns = pd.DataFrame(
        [
            [0.5, 0.2],
            [0.2, 0.4],
            [-0.3, 0.6],
            [-0.1, 0.1],
            [0.8, 0.67],
            [0.9, 0.43],
        ]
    )
    benchmark = pd.Series([0.08, 0.10, 0.13, 0.11, -0.03, -0.05])

    obtained = stats.beta(returns, benchmark)

    expected = pd.Series([-5.835214446952595, -1.1038374717832957])
    pd.testing.assert_series_equal(obtained, expected)


def test_beta_of_benchmark_against_itself_is_one():
    obtained = stats.beta(BENCH, BENCH)

    np.testing.assert_almost_equal(obtained, 1.0)


def test_beta_of_linear_returns_is_the_slope():
    # LINEAR = 2 * BENCH + 0.01 -> beta 2
    obtained = stats.beta(LINEAR, BENCH)

    np.testing.assert_almost_equal(obtained, 2.0)


def test_beta_is_covariance_over_benchmark_variance():
    # benchmark deviations [-0.01, 0, 0.01], returns deviations
    # [-0.02, -0.01, 0.03] -> cov = 0.00025, var = 0.0001 -> beta 2.5
    returns = np.array([0.01, 0.02, 0.06])
    benchmark = np.array([0.01, 0.02, 0.03])

    obtained = stats.beta(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 2.5)


# ---------------------------------------------------------------------------
# alpha


def test_alpha_of_benchmark_against_itself_is_zero():
    obtained = stats.alpha(BENCH, BENCH)

    np.testing.assert_almost_equal(obtained, 0.0)


def test_alpha_of_linear_returns_is_the_intercept():
    # LINEAR = 2 * BENCH + 0.01 -> alpha 0.01 with risk_free 0
    obtained = stats.alpha(LINEAR, BENCH)

    np.testing.assert_almost_equal(obtained, 0.01)


def test_alpha_is_mean_return_minus_beta_times_mean_benchmark():
    # beta 2.5 (see test_beta_is_covariance_over_benchmark_variance),
    # mean returns 0.03, mean benchmark 0.02 -> 0.03 - 2.5 * 0.02 = -0.02
    returns = np.array([0.01, 0.02, 0.06])
    benchmark = np.array([0.01, 0.02, 0.03])

    obtained = stats.alpha(returns, benchmark)

    np.testing.assert_almost_equal(obtained, -0.02)


def test_alpha_subtracts_the_risk_free_rate_from_both_sides():
    # mean(r - rf) - beta * mean(b - rf) = 0.01 - rf * (1 - beta)
    # with beta 2 and rf 0.005 -> 0.01 + 0.005 = 0.015
    obtained = stats.alpha(LINEAR, BENCH, risk_free=0.005)

    np.testing.assert_almost_equal(obtained, 0.015)


def test_alpha_is_multiplied_by_factor():
    # Morningstar annualizes a monthly alpha by 12 -> 0.01 * 12 = 0.12
    obtained = stats.alpha(LINEAR, BENCH, factor=12)

    np.testing.assert_almost_equal(obtained, 0.12)


def test_alpha_factor_applies_to_every_column():
    returns = pd.DataFrame({"fund": LINEAR, "short": INVERTED})
    benchmark = pd.Series(BENCH)

    obtained = stats.alpha(returns, benchmark, factor=12)

    expected = pd.Series([0.12, 0.0], index=["fund", "short"])
    pd.testing.assert_series_equal(obtained, expected)


# ---------------------------------------------------------------------------
# correlation and r_squared


def test_correlation_of_benchmark_against_itself_is_one():
    obtained = stats.correlation(BENCH, BENCH)

    np.testing.assert_almost_equal(obtained, 1.0)


def test_correlation_of_linear_returns_is_one():
    obtained = stats.correlation(LINEAR, BENCH)

    np.testing.assert_almost_equal(obtained, 1.0)


def test_correlation_of_inverted_returns_is_minus_one():
    obtained = stats.correlation(INVERTED, BENCH)

    np.testing.assert_almost_equal(obtained, -1.0)


def test_correlation_of_symmetric_returns_is_zero():
    # returns are even in the benchmark: the covariance cancels out
    returns = np.array([0.01, 0.0, 0.01])
    benchmark = np.array([-0.01, 0.0, 0.01])

    obtained = stats.correlation(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 0.0)


def test_correlation_is_covariance_over_the_product_of_deviations():
    # deviations [-0.01, 0.01, 0] and [-0.01, 0, 0.01]: sum of products
    # 0.0001, sum of squares 0.0002 each -> 0.0001 / 0.0002 = 0.5
    returns = np.array([0.01, 0.03, 0.02])
    benchmark = np.array([0.01, 0.02, 0.03])

    obtained = stats.correlation(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 0.5)


def test_correlation_of_constant_returns_is_nan():
    returns = np.array([0.05, 0.05, 0.05])
    benchmark = np.array([0.01, 0.02, 0.03])

    obtained = stats.correlation(returns, benchmark)

    assert np.isnan(obtained)


def test_r_squared_is_the_square_of_the_correlation():
    # correlation 0.5 (see above) -> r_squared 0.25
    returns = np.array([0.01, 0.03, 0.02])
    benchmark = np.array([0.01, 0.02, 0.03])

    obtained = stats.r_squared(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 0.25)


def test_r_squared_is_explained_variance_ratio():
    # cov 0.00025, var_b 0.0001, var_r 0.0007 -> 0.00025^2 / (0.0001 *
    # 0.0007) = 25 / 28
    returns = np.array([0.01, 0.02, 0.06])
    benchmark = np.array([0.01, 0.02, 0.03])

    obtained = stats.r_squared(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 25 / 28)


def test_r_squared_of_inverted_returns_is_one():
    obtained = stats.r_squared(INVERTED, BENCH)

    np.testing.assert_almost_equal(obtained, 1.0)


def test_r_squared_of_constant_returns_is_nan():
    returns = np.array([0.05, 0.05, 0.05])
    benchmark = np.array([0.01, 0.02, 0.03])

    obtained = stats.r_squared(returns, benchmark)

    assert np.isnan(obtained)


# ---------------------------------------------------------------------------
# bull_beta and bear_beta

# returns = 2 * BENCH where BENCH > 0 and 0.5 * BENCH where BENCH < 0
# -> bull beta 2, bear beta 0.5
SPLIT = np.array([0.04, -0.005, 0.06, -0.01, 0.02, -0.015])


def test_bull_beta_uses_only_the_rows_with_positive_benchmark():
    obtained = stats.bull_beta(SPLIT, BENCH)

    np.testing.assert_almost_equal(obtained, 2.0)


def test_bear_beta_uses_only_the_rows_with_negative_benchmark():
    obtained = stats.bear_beta(SPLIT, BENCH)

    np.testing.assert_almost_equal(obtained, 0.5)


def test_full_sample_beta_lies_between_bull_and_bear_beta():
    obtained = stats.beta(SPLIT, BENCH)

    assert 0.5 < obtained < 2.0


def test_bull_and_bear_beta_exclude_rows_with_zero_benchmark():
    # the second row has a zero benchmark and a wild return: it belongs to
    # neither the bull nor the bear sample
    returns = np.array([0.04, 1.0, 0.06, -0.01, 0.02, -0.015])
    benchmark = np.array([0.02, 0.0, 0.03, -0.02, 0.01, -0.03])

    np.testing.assert_almost_equal(stats.bull_beta(returns, benchmark), 2.0)
    np.testing.assert_almost_equal(stats.bear_beta(returns, benchmark), 0.5)


def test_bull_beta_of_benchmark_against_itself_is_one():
    obtained = stats.bull_beta(BENCH, BENCH)

    np.testing.assert_almost_equal(obtained, 1.0)


def test_bear_beta_of_benchmark_against_itself_is_one():
    obtained = stats.bear_beta(BENCH, BENCH)

    np.testing.assert_almost_equal(obtained, 1.0)


def test_bull_beta_with_a_single_positive_row_is_nan():
    returns = np.array([0.04, -0.005, -0.01, -0.015])
    benchmark = np.array([0.02, -0.01, -0.02, -0.03])

    assert np.isnan(stats.bull_beta(returns, benchmark))
    np.testing.assert_almost_equal(stats.bear_beta(returns, benchmark), 0.5)


def test_bear_beta_with_a_single_negative_row_is_nan():
    returns = np.array([0.04, -0.005, 0.06, 0.02])
    benchmark = np.array([0.02, -0.01, 0.03, 0.01])

    assert np.isnan(stats.bear_beta(returns, benchmark))
    np.testing.assert_almost_equal(stats.bull_beta(returns, benchmark), 2.0)


def test_bull_beta_of_constant_positive_benchmark_is_nan():
    returns = np.array([0.01, 0.02, 0.03, -0.005, -0.01])
    benchmark = np.array([0.05, 0.05, 0.05, -0.01, -0.02])

    assert np.isnan(stats.bull_beta(returns, benchmark))
    np.testing.assert_almost_equal(stats.bear_beta(returns, benchmark), 0.5)


def test_bear_beta_of_constant_negative_benchmark_is_nan():
    returns = np.array([0.04, 0.02, 0.01, 0.02, 0.03])
    benchmark = np.array([0.02, 0.01, -0.05, -0.05, -0.05])

    assert np.isnan(stats.bear_beta(returns, benchmark))
    np.testing.assert_almost_equal(stats.bull_beta(returns, benchmark), 2.0)
