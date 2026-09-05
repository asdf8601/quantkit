"""Tests of ``stats.treynor_ratio``.

Treynor is Sharpe with systematic risk in the denominator: the mean excess
return of ``returns`` divided by their :func:`quantkit.stats.beta` against
``benchmark``, both estimated on the rows where the column and the
benchmark are non-NaN.
"""
from quantkit import stats

import pytest
import numpy as np
import pandas as pd


# Six observations. sum(BENCH) = 0.06 -> mean(BENCH) = 0.01, so the
# benchmark's own Treynor ratio is not trivially zero.
BENCH = np.array([0.02, -0.01, 0.03, -0.02, 0.04, 0.00])

# LINEAR = 2 * BENCH + 0.01 -> beta 2; sum(LINEAR) = 0.18 -> mean 0.03
# -> treynor 0.03 / 2 = 0.015
LINEAR = np.array([0.05, -0.01, 0.07, -0.03, 0.09, 0.01])

# INVERTED = -BENCH -> beta -1, mean -0.01 -> treynor -0.01 / -1 = 0.01
INVERTED = np.array([-0.02, 0.01, -0.03, 0.02, -0.04, 0.00])


# ---------------------------------------------------------------------------
# the ratio itself


def test_treynor_ratio_of_the_benchmark_against_itself_is_its_mean_excess_return():
    # beta of the benchmark on itself is 1, so the ratio collapses to
    # mean(BENCH) = 0.06 / 6 = 0.01
    obtained = stats.treynor_ratio(BENCH, BENCH)

    np.testing.assert_almost_equal(obtained, 0.01)


def test_treynor_ratio_of_the_benchmark_against_itself_nets_the_risk_free_rate():
    # (mean(BENCH) - rf) / 1 = (0.01 - 0.002) / 1 = 0.008
    obtained = stats.treynor_ratio(BENCH, BENCH, risk_free=0.002)

    np.testing.assert_almost_equal(obtained, 0.008)


def test_treynor_ratio_of_linear_returns_is_the_mean_return_over_beta():
    # LINEAR = 2 * BENCH + 0.01: beta 2, mean(LINEAR) 0.03
    # -> treynor 0.03 / 2 = 0.015
    obtained = stats.treynor_ratio(LINEAR, BENCH)

    np.testing.assert_almost_equal(obtained, 0.015)


def test_treynor_ratio_divides_the_mean_return_by_a_hand_computed_beta():
    # benchmark deviations [-0.01, 0, 0.01], returns deviations
    # [-0.02, -0.01, 0.03] -> cov 0.00025, var 0.0001 -> beta 2.5;
    # mean(returns) = 0.09 / 3 = 0.03 -> treynor 0.03 / 2.5 = 0.012
    returns = np.array([0.01, 0.02, 0.06])
    benchmark = np.array([0.01, 0.02, 0.03])

    obtained = stats.treynor_ratio(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 0.012)


def test_treynor_ratio_of_a_negative_beta_flips_the_sign_of_the_mean_return():
    # INVERTED = -BENCH: beta -1 and mean -0.01, so a losing series gets a
    # positive ratio, -0.01 / -1 = 0.01. The sign of Treynor follows the
    # sign of the excess return only when beta is positive.
    obtained = stats.treynor_ratio(INVERTED, BENCH)

    np.testing.assert_almost_equal(obtained, 0.01)


def test_treynor_ratio_subtracts_the_risk_free_rate_from_the_numerator():
    # (mean(LINEAR) - rf) / beta = (0.03 - 0.005) / 2 = 0.0125, i.e. the
    # ratio drops by rf / beta = 0.0025 from 0.015
    obtained = stats.treynor_ratio(LINEAR, BENCH, risk_free=0.005)

    np.testing.assert_almost_equal(obtained, 0.0125)


def test_treynor_ratio_is_multiplied_by_factor():
    # Morningstar annualizes the numerator of a monthly ratio by 12, which
    # scales the whole ratio: 0.015 * 12 = 0.18
    obtained = stats.treynor_ratio(LINEAR, BENCH, factor=12)

    np.testing.assert_almost_equal(obtained, 0.18)


def test_treynor_ratio_applies_risk_free_and_factor_together():
    # (0.03 - 0.005) / 2 * 12 = 0.0125 * 12 = 0.15
    obtained = stats.treynor_ratio(
        LINEAR, BENCH, risk_free=0.005, factor=12
    )

    np.testing.assert_almost_equal(obtained, 0.15)


# ---------------------------------------------------------------------------
# container types


def test_treynor_ratio_of_a_series_reduces_to_a_float():
    returns = pd.Series(LINEAR, name="fund")
    benchmark = pd.Series(BENCH, name="index")

    obtained = stats.treynor_ratio(returns, benchmark)

    assert isinstance(obtained, float)
    np.testing.assert_almost_equal(obtained, 0.015)


def test_treynor_ratio_of_two_numpy_columns_gives_one_value_per_column():
    # LINEAR -> 0.03 / 2 = 0.015, INVERTED -> -0.01 / -1 = 0.01
    returns = np.column_stack([LINEAR, INVERTED])

    obtained = stats.treynor_ratio(returns, BENCH)

    assert isinstance(obtained, np.ndarray)
    np.testing.assert_almost_equal(obtained, [0.015, 0.01])


def test_treynor_ratio_of_a_dataframe_is_a_series_indexed_by_columns():
    returns = pd.DataFrame({"fund": LINEAR, "short": INVERTED})
    benchmark = pd.Series(BENCH)

    obtained = stats.treynor_ratio(returns, benchmark)

    expected = pd.Series([0.015, 0.01], index=["fund", "short"])
    pd.testing.assert_series_equal(obtained, expected)


def test_treynor_ratio_factor_applies_to_every_column():
    returns = pd.DataFrame({"fund": LINEAR, "short": INVERTED})
    benchmark = pd.Series(BENCH)

    obtained = stats.treynor_ratio(returns, benchmark, factor=12)

    expected = pd.Series([0.18, 0.12], index=["fund", "short"])
    pd.testing.assert_series_equal(obtained, expected)


# ---------------------------------------------------------------------------
# missing data


def test_nan_at_the_start_of_returns_is_dropped_from_both_moments():
    # dropping row 0 leaves sum(LINEAR) = 0.13 over 5 rows -> mean 0.026,
    # beta is still exactly 2 -> treynor 0.026 / 2 = 0.013
    returns = LINEAR.copy()
    returns[0] = np.nan

    obtained = stats.treynor_ratio(returns, BENCH)

    np.testing.assert_almost_equal(obtained, 0.013)


def test_nan_in_the_middle_of_returns_is_dropped_from_both_moments():
    # dropping row 3 (-0.03) leaves sum 0.21 over 5 rows -> mean 0.042,
    # beta still 2 -> treynor 0.042 / 2 = 0.021
    returns = LINEAR.copy()
    returns[3] = np.nan

    obtained = stats.treynor_ratio(returns, BENCH)

    np.testing.assert_almost_equal(obtained, 0.021)


def test_nan_in_the_benchmark_drops_that_row():
    # the return on the NaN benchmark row is wild and must not reach the
    # mean: the 5 surviving rows sum to 0.11 -> mean 0.022, beta 2
    # -> treynor 0.022 / 2 = 0.011
    returns = LINEAR.copy()
    returns[2] = 100.0
    benchmark = BENCH.copy()
    benchmark[2] = np.nan

    obtained = stats.treynor_ratio(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 0.011)


def test_nan_in_one_column_drops_the_row_only_for_that_column():
    # column 0 loses row 2 -> mean 0.022, beta 2 -> 0.011; column 1 keeps
    # all six rows -> -0.01 / -1 = 0.01
    returns = np.column_stack([LINEAR, INVERTED])
    returns[2, 0] = np.nan

    obtained = stats.treynor_ratio(returns, BENCH)

    np.testing.assert_almost_equal(obtained, [0.011, 0.01])


def test_treynor_ratio_of_empty_numpy_returns_is_nan():
    obtained = stats.treynor_ratio(np.array([]), np.array([]))

    assert np.isnan(obtained)


def test_treynor_ratio_of_an_empty_series_is_nan():
    returns = pd.Series([], dtype=float)
    benchmark = pd.Series([], dtype=float)

    obtained = stats.treynor_ratio(returns, benchmark)

    assert np.isnan(obtained)


def test_treynor_ratio_of_all_nan_returns_is_nan():
    returns = np.array([np.nan, np.nan, np.nan, np.nan])

    obtained = stats.treynor_ratio(returns, BENCH[:4])

    assert np.isnan(obtained)


def test_treynor_ratio_of_an_all_nan_dataframe_is_nan_per_column():
    returns = pd.DataFrame(np.full((4, 2), np.nan), columns=["a", "b"])
    benchmark = pd.Series(BENCH[:4])

    obtained = stats.treynor_ratio(returns, benchmark)

    expected = pd.Series([np.nan, np.nan], index=["a", "b"])
    pd.testing.assert_series_equal(obtained, expected)


def test_treynor_ratio_of_a_single_complete_row_is_nan():
    # only the last row survives: no beta can be estimated from one point
    returns = np.array([np.nan, np.nan, 0.05])
    benchmark = np.array([0.02, -0.01, 0.03])

    obtained = stats.treynor_ratio(returns, benchmark)

    assert np.isnan(obtained)


# ---------------------------------------------------------------------------
# degenerate denominators


def test_treynor_ratio_of_a_zero_variance_benchmark_is_nan():
    # a constant benchmark has no variance, so beta is NaN and so is the
    # ratio: never a huge number coming from a rounded variance
    returns = np.array([0.01, 0.02, 0.03])
    benchmark = np.array([0.05, 0.05, 0.05])

    obtained = stats.treynor_ratio(returns, benchmark)

    assert np.isnan(obtained)


def test_treynor_ratio_of_constant_returns_is_nan_because_beta_is_zero():
    # constant returns cannot covary with anything, so beta is 0 and
    # 0.1 / 0 is not a number. The computed beta of np.full(3, 0.1) is
    # 1.2e-31 rather than 0 because the mean is rounded, so the guard must
    # not trust an exact comparison against zero.
    returns = np.full(3, 0.1)
    benchmark = np.array([0.01, 0.02, 0.03])

    obtained = stats.treynor_ratio(returns, benchmark)

    assert np.isnan(obtained)


def test_treynor_ratio_of_returns_uncorrelated_with_the_benchmark_is_nan():
    # returns are even in the benchmark (rows 0 and 2 share a value while
    # the benchmark deviations are +/- 0.25), so cov = 0 -> beta 0 -> NaN,
    # not +/- inf
    returns = np.array([0.5, 0.25, 0.5])
    benchmark = np.array([-0.25, 0.0, 0.25])

    obtained = stats.treynor_ratio(returns, benchmark)

    assert np.isnan(obtained)


def test_treynor_ratio_of_a_zero_beta_column_is_nan_next_to_a_normal_one():
    # only the constant column degenerates; the other keeps its value
    returns = pd.DataFrame({"fund": LINEAR, "cash": np.full(6, 0.001)})
    benchmark = pd.Series(BENCH)

    obtained = stats.treynor_ratio(returns, benchmark)

    expected = pd.Series([0.015, np.nan], index=["fund", "cash"])
    pd.testing.assert_series_equal(obtained, expected)


# ---------------------------------------------------------------------------
# alignment


def test_treynor_ratio_uses_the_overlap_of_misaligned_pandas_indexes():
    dates = pd.date_range("2020-01-01", periods=8)
    # benchmark is known on d0..d6 and returns on d1..d7; on the overlap
    # d1..d6 returns = 2 * benchmark + 0.01. The values outside the overlap
    # are wild so a positional match would be visibly wrong.
    benchmark = pd.Series(np.append(1.0, BENCH), index=dates[:7])
    returns = pd.Series(np.append(LINEAR, -1.0), index=dates[1:])

    obtained = stats.treynor_ratio(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 0.015)


def test_treynor_ratio_rejects_numpy_of_different_length():
    returns = np.array([0.01, 0.02, 0.03])
    benchmark = np.array([0.02, -0.01])

    with pytest.raises(ValueError, match="length"):
        stats.treynor_ratio(returns, benchmark)


def test_treynor_ratio_rejects_a_two_dimensional_benchmark():
    returns = np.array([0.01, 0.02, 0.03])
    benchmark = np.array([[0.02], [-0.01], [0.03]])

    with pytest.raises(ValueError, match="1D"):
        stats.treynor_ratio(returns, benchmark)
