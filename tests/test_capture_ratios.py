"""Tests of the capture ratios and the batting average.

Every function receives ``(returns, benchmark)`` where ``returns`` is 1D or
2D (numpy or pandas) and ``benchmark`` is 1D. Reductions are column-wise on
the rows where both the column and the benchmark are non-NaN, and the result
is in parts per unit: 1.10 means 110%, the number Morningstar quotes as a
percentage.
"""
from quantkit import stats

import pytest
import numpy as np
import pandas as pd


# Four observations: the benchmark rises twice and falls twice.
BENCH = np.array([0.10, -0.10, 0.10, -0.10])

# LEADER doubles the benchmark going up and halves it going down.
LEADER = np.array([0.20, -0.05, 0.20, -0.05])

# LAGGARD halves the benchmark going up and doubles it going down.
LAGGARD = np.array([0.05, -0.20, 0.05, -0.20])

# up periods (b >= 0): b 0.10, 0.10 -> 1.1 * 1.1 - 1 = 0.21
#   LEADER  0.20, 0.20 -> 1.2 * 1.2 - 1 = 0.44    -> 0.44 / 0.21 = 2.0952...
#   LAGGARD 0.05, 0.05 -> 1.05 ** 2 - 1 = 0.1025  -> 0.1025 / 0.21 = 0.4881...
LEADER_UP = (1.20**2 - 1) / (1.10**2 - 1)
LAGGARD_UP = (1.05**2 - 1) / (1.10**2 - 1)

# down periods (b < 0): b -0.10, -0.10 -> 0.9 * 0.9 - 1 = -0.19
#   LEADER  -0.05, -0.05 -> 0.95 ** 2 - 1 = -0.0975 -> 0.5131...
#   LAGGARD -0.20, -0.20 -> 0.80 ** 2 - 1 = -0.36   -> 1.8947...
LEADER_DOWN = (0.95**2 - 1) / (0.90**2 - 1)
LAGGARD_DOWN = (0.80**2 - 1) / (0.90**2 - 1)

CAPTURE_FUNCTIONS = [
    "up_capture",
    "down_capture",
    "overall_capture",
    "batting_average",
]

# Expected statistic of LEADER against BENCH, per function. LEADER is at or
# above the benchmark on all four rows, so its batting average is 1.
LEADER_EXPECTED = [
    ("up_capture", LEADER_UP),
    ("down_capture", LEADER_DOWN),
    ("overall_capture", LEADER_UP / LEADER_DOWN),
    ("batting_average", 1.0),
]

# Expected statistic of the two columns [LEADER, LAGGARD] against BENCH.
# LAGGARD is below the benchmark on every row, so it never bats.
TWO_COLUMN_EXPECTED = [
    ("up_capture", [LEADER_UP, LAGGARD_UP]),
    ("down_capture", [LEADER_DOWN, LAGGARD_DOWN]),
    (
        "overall_capture",
        [LEADER_UP / LEADER_DOWN, LAGGARD_UP / LAGGARD_DOWN],
    ),
    ("batting_average", [1.0, 0.0]),
]

# A fifth row where the benchmark rises again. LEADER is NaN there, LAGGARD
# is not, so LEADER reduces over four rows and LAGGARD over five.
BENCH_5 = np.append(BENCH, 0.10)
LEADER_5 = np.append(LEADER, np.nan)
LAGGARD_5 = np.append(LAGGARD, 0.05)

# LAGGARD up periods 0.05, 0.05, 0.05 -> 1.05 ** 3 - 1 = 0.157625 over
# b 0.10, 0.10, 0.10 -> 1.10 ** 3 - 1 = 0.331 -> 0.4762...
LAGGARD_UP_5 = (1.05**3 - 1) / (1.10**3 - 1)

ONE_COLUMN_NAN_EXPECTED = [
    ("up_capture", [LEADER_UP, LAGGARD_UP_5]),
    ("down_capture", [LEADER_DOWN, LAGGARD_DOWN]),
    (
        "overall_capture",
        [LEADER_UP / LEADER_DOWN, LAGGARD_UP_5 / LAGGARD_DOWN],
    ),
    ("batting_average", [1.0, 0.0]),
]


# ---------------------------------------------------------------------------
# Shared contract of every capture function


@pytest.mark.parametrize("name, expected", LEADER_EXPECTED)
def test_one_dimensional_numpy_returns_reduce_to_a_float(name, expected):
    obtained = getattr(stats, name)(LEADER, BENCH)

    assert isinstance(obtained, float)
    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", LEADER_EXPECTED)
def test_series_returns_reduce_to_a_float(name, expected):
    returns = pd.Series(LEADER, name="fund")
    benchmark = pd.Series(BENCH, name="index")

    obtained = getattr(stats, name)(returns, benchmark)

    assert isinstance(obtained, float)
    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", TWO_COLUMN_EXPECTED)
def test_two_column_numpy_returns_reduce_to_one_value_per_column(
    name, expected
):
    returns = np.column_stack([LEADER, LAGGARD])

    obtained = getattr(stats, name)(returns, BENCH)

    assert isinstance(obtained, np.ndarray)
    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", TWO_COLUMN_EXPECTED)
def test_dataframe_returns_reduce_to_a_series_indexed_by_columns(
    name, expected
):
    returns = pd.DataFrame({"fund": LEADER, "slow": LAGGARD})
    benchmark = pd.Series(BENCH)

    obtained = getattr(stats, name)(returns, benchmark)

    expected = pd.Series(expected, index=["fund", "slow"])
    pd.testing.assert_series_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", LEADER_EXPECTED)
def test_nan_in_the_returns_drops_that_row(name, expected):
    # the fifth row is NaN in the returns, so the statistic is the one of
    # the first four rows
    obtained = getattr(stats, name)(LEADER_5, BENCH_5)

    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", LEADER_EXPECTED)
def test_nan_in_the_benchmark_drops_that_row(name, expected):
    # the fifth benchmark value is NaN and the return on that row is wild:
    # it must not reach any compound product nor any hit count
    returns = np.append(LEADER, 5.0)
    benchmark = np.append(BENCH, np.nan)

    obtained = getattr(stats, name)(returns, benchmark)

    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", ONE_COLUMN_NAN_EXPECTED)
def test_nan_in_one_column_drops_the_row_only_for_that_column(name, expected):
    returns = np.column_stack([LEADER_5, LAGGARD_5])

    obtained = getattr(stats, name)(returns, BENCH_5)

    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("name, expected", LEADER_EXPECTED)
def test_misaligned_pandas_indexes_use_the_overlap(name, expected):
    dates = pd.date_range("2020-01-01", periods=6)
    # the benchmark is known on d0..d4 and the returns on d1..d5; on the
    # overlap d1..d4 they are BENCH and LEADER. The values outside the
    # overlap are wild so a positional match would be visibly wrong.
    benchmark = pd.Series(np.append(1.0, BENCH), index=dates[:5])
    returns = pd.Series(np.append(LEADER, -0.90), index=dates[1:])

    obtained = getattr(stats, name)(returns, benchmark)

    np.testing.assert_almost_equal(obtained, expected)


@pytest.mark.parametrize("name", CAPTURE_FUNCTIONS)
def test_empty_numpy_returns_give_nan(name):
    obtained = getattr(stats, name)(np.array([]), np.array([]))

    assert np.isnan(obtained)


@pytest.mark.parametrize("name", CAPTURE_FUNCTIONS)
def test_empty_series_returns_give_nan(name):
    returns = pd.Series([], dtype=float)
    benchmark = pd.Series([], dtype=float)

    obtained = getattr(stats, name)(returns, benchmark)

    assert np.isnan(obtained)


@pytest.mark.parametrize("name", CAPTURE_FUNCTIONS)
def test_all_nan_returns_give_nan(name):
    returns = np.array([np.nan, np.nan, np.nan, np.nan])

    obtained = getattr(stats, name)(returns, BENCH)

    assert np.isnan(obtained)


@pytest.mark.parametrize("name", CAPTURE_FUNCTIONS)
def test_all_nan_dataframe_gives_nan_per_column(name):
    returns = pd.DataFrame(np.full((4, 2), np.nan), columns=["fund", "slow"])
    benchmark = pd.Series(BENCH)

    obtained = getattr(stats, name)(returns, benchmark)

    expected = pd.Series([np.nan, np.nan], index=["fund", "slow"])
    pd.testing.assert_series_equal(obtained, expected)


@pytest.mark.parametrize("name", CAPTURE_FUNCTIONS)
def test_no_row_where_both_sides_are_valid_gives_nan(name):
    # the NaN alternate, so the pairwise complete sample is empty
    returns = np.array([0.10, np.nan, 0.20, np.nan])
    benchmark = np.array([np.nan, -0.10, np.nan, -0.20])

    obtained = getattr(stats, name)(returns, benchmark)

    assert np.isnan(obtained)


@pytest.mark.parametrize("name, expected", LEADER_EXPECTED)
def test_returns_equal_to_the_benchmark_give_one(name, expected):
    # an asset that tracks the benchmark exactly captures all of the upside
    # and all of the downside and never loses a period
    obtained = getattr(stats, name)(BENCH, BENCH)

    np.testing.assert_almost_equal(obtained, 1.0)


@pytest.mark.parametrize("name", CAPTURE_FUNCTIONS)
def test_numpy_of_different_length_is_rejected(name):
    returns = np.array([0.01, 0.02, 0.03])
    benchmark = np.array([0.02, -0.01])

    with pytest.raises(ValueError, match="length"):
        getattr(stats, name)(returns, benchmark)


@pytest.mark.parametrize("name", CAPTURE_FUNCTIONS)
def test_two_dimensional_numpy_benchmark_is_rejected(name):
    returns = np.array([0.01, 0.02, 0.03])
    benchmark = np.array([[0.02], [-0.01], [0.03]])

    with pytest.raises(ValueError, match="1D"):
        getattr(stats, name)(returns, benchmark)


@pytest.mark.parametrize("name", CAPTURE_FUNCTIONS)
def test_dataframe_benchmark_is_rejected(name):
    returns = pd.Series([0.01, 0.02, 0.03])
    benchmark = pd.DataFrame([[0.02], [-0.01], [0.03]])

    with pytest.raises(ValueError, match="1D"):
        getattr(stats, name)(returns, benchmark)


# ---------------------------------------------------------------------------
# up_capture


def test_up_capture_of_a_single_up_period_is_the_ratio_of_the_returns():
    # one up period: compounding is linear, so 2 * benchmark gives 2 exactly
    # r 0.10 -> 0.10; b 0.05 -> 0.05; ratio 0.10 / 0.05 = 2.0
    returns = np.array([0.10])
    benchmark = np.array([0.05])

    obtained = stats.up_capture(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 2.0)


def test_up_capture_compounds_the_up_periods_instead_of_adding_them():
    # up periods (b >= 0): r 0.10, 0.20 -> 1.1 * 1.2 - 1 = 0.32;
    # b 0.05, 0.10 -> 1.05 * 1.10 - 1 = 0.155; ratio 2.0645...
    # adding instead of compounding would give 0.30 / 0.15 = 2.0
    returns = np.array([0.10, 0.20])
    benchmark = np.array([0.05, 0.10])

    obtained = stats.up_capture(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 0.32 / 0.155)
    assert obtained != pytest.approx(2.0)


def test_up_capture_uses_only_the_rows_where_the_benchmark_did_not_fall():
    # up periods are rows 0 and 2: r 0.20, 0.20 -> 0.44; b 0.10, 0.10 -> 0.21
    # the down rows carry a wild return that must not be captured
    returns = np.array([0.20, 9.0, 0.20, 9.0])
    benchmark = np.array([0.10, -0.10, 0.10, -0.10])

    obtained = stats.up_capture(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 0.44 / 0.21)


def test_up_capture_counts_a_zero_benchmark_period_as_an_up_period():
    # up periods (b >= 0) are rows 0 and 1: b 0.10, 0.00 -> 1.1 * 1.0 - 1
    # = 0.10; r 0.20, 0.50 -> 1.2 * 1.5 - 1 = 0.80; ratio 8.0.
    # Dropping the zero row would give 0.20 / 0.10 = 2.0 instead.
    returns = np.array([0.20, 0.50, -0.05])
    benchmark = np.array([0.10, 0.00, -0.10])

    obtained = stats.up_capture(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 8.0)


def test_up_capture_below_one_means_the_asset_rose_less_than_the_benchmark():
    obtained = stats.up_capture(LAGGARD, BENCH)

    np.testing.assert_almost_equal(obtained, LAGGARD_UP)
    assert obtained < 1.0


def test_up_capture_is_nan_when_the_benchmark_never_rises():
    returns = np.array([-0.05, -0.10, -0.15])
    benchmark = np.array([-0.10, -0.20, -0.30])

    assert np.isnan(stats.up_capture(returns, benchmark))


def test_up_capture_is_nan_when_the_benchmark_is_flat_on_the_up_periods():
    # every benchmark return is 0, so the up compound return is exactly 0
    # and the ratio has no denominator
    returns = np.array([0.10, 0.20, -0.05])
    benchmark = np.array([0.0, 0.0, 0.0])

    assert np.isnan(stats.up_capture(returns, benchmark))


def test_up_capture_of_a_flat_asset_on_a_rising_benchmark_is_zero():
    # the asset stands still while the benchmark rises: it captures nothing
    returns = np.array([0.0, 0.0])
    benchmark = np.array([0.10, 0.10])

    np.testing.assert_almost_equal(stats.up_capture(returns, benchmark), 0.0)


# ---------------------------------------------------------------------------
# down_capture


def test_down_capture_above_one_means_the_asset_fell_more_than_the_benchmark():
    # down periods: r -0.20, -0.20 -> 0.8 * 0.8 - 1 = -0.36;
    # b -0.10, -0.10 -> 0.9 * 0.9 - 1 = -0.19; ratio 1.8947...
    obtained = stats.down_capture(LAGGARD, BENCH)

    np.testing.assert_almost_equal(obtained, 0.36 / 0.19)
    assert obtained > 1.0


def test_down_capture_below_one_means_the_asset_fell_less_than_the_benchmark():
    # down periods: r -0.05, -0.05 -> 0.95 ** 2 - 1 = -0.0975;
    # b -0.10, -0.10 -> -0.19; ratio 0.5131...
    obtained = stats.down_capture(LEADER, BENCH)

    np.testing.assert_almost_equal(obtained, 0.0975 / 0.19)
    assert obtained < 1.0


def test_down_capture_compounds_the_down_periods_instead_of_adding_them():
    # down periods: r -0.10, -0.20 -> 0.9 * 0.8 - 1 = -0.28;
    # b -0.05, -0.10 -> 0.95 * 0.90 - 1 = -0.145; ratio 1.9310...
    # adding instead of compounding would give 0.30 / 0.15 = 2.0
    returns = np.array([-0.10, -0.20])
    benchmark = np.array([-0.05, -0.10])

    obtained = stats.down_capture(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 0.28 / 0.145)
    assert obtained != pytest.approx(2.0)


def test_down_capture_excludes_the_rows_with_a_zero_benchmark():
    # only row 1 is a down period: r -0.05 over b -0.10 -> 0.5. The zero
    # benchmark row is an up period and its wild return must not appear.
    returns = np.array([9.0, -0.05])
    benchmark = np.array([0.00, -0.10])

    obtained = stats.down_capture(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 0.5)


def test_down_capture_is_nan_when_the_benchmark_never_falls():
    returns = np.array([0.10, -0.05, 0.20])
    benchmark = np.array([0.10, 0.00, 0.20])

    assert np.isnan(stats.down_capture(returns, benchmark))


def test_down_capture_is_negative_when_the_asset_rises_as_the_benchmark_falls():
    # r 0.10 -> 0.10 over b -0.20 -> -0.20: the asset gains on the way down
    returns = np.array([0.10])
    benchmark = np.array([-0.20])

    obtained = stats.down_capture(returns, benchmark)

    np.testing.assert_almost_equal(obtained, -0.5)


# ---------------------------------------------------------------------------
# overall_capture


def test_overall_capture_is_up_capture_divided_by_down_capture():
    # up 0.44 / 0.21 = 2.0952...; down 0.0975 / 0.19 = 0.5131...;
    # overall 2.0952 / 0.5131 = 4.0830...
    obtained = stats.overall_capture(LEADER, BENCH)

    np.testing.assert_almost_equal(obtained, (0.44 / 0.21) / (0.0975 / 0.19))


def test_overall_capture_above_one_means_more_upside_than_downside():
    assert stats.overall_capture(LEADER, BENCH) > 1.0


def test_overall_capture_below_one_means_more_downside_than_upside():
    # up 0.1025 / 0.21 = 0.4881...; down 0.36 / 0.19 = 1.8947...
    obtained = stats.overall_capture(LAGGARD, BENCH)

    np.testing.assert_almost_equal(obtained, (0.1025 / 0.21) / (0.36 / 0.19))
    assert obtained < 1.0


def test_overall_capture_is_nan_when_the_benchmark_never_falls():
    returns = np.array([0.20, 0.10, 0.30])
    benchmark = np.array([0.10, 0.00, 0.20])

    assert np.isnan(stats.down_capture(returns, benchmark))
    assert np.isnan(stats.overall_capture(returns, benchmark))


def test_overall_capture_is_nan_when_the_benchmark_never_rises():
    returns = np.array([-0.05, -0.10])
    benchmark = np.array([-0.10, -0.20])

    assert np.isnan(stats.up_capture(returns, benchmark))
    assert np.isnan(stats.overall_capture(returns, benchmark))


def test_overall_capture_is_nan_when_down_capture_is_zero():
    # the asset is flat on the only down period: down capture is 0 and the
    # overall ratio has no denominator
    returns = np.array([0.20, 0.0])
    benchmark = np.array([0.10, -0.10])

    np.testing.assert_almost_equal(
        stats.down_capture(returns, benchmark), 0.0
    )
    assert np.isnan(stats.overall_capture(returns, benchmark))


# ---------------------------------------------------------------------------
# batting_average


def test_batting_average_is_the_fraction_of_periods_at_or_above_benchmark():
    # hits: 0.10 >= 0.05 yes, -0.05 >= -0.10 yes, 0.02 >= 0.05 no,
    # -0.30 >= -0.20 no -> 2 of 4 -> 0.5 (parts per unit, not 50)
    returns = np.array([0.10, -0.05, 0.02, -0.30])
    benchmark = np.array([0.05, -0.10, 0.05, -0.20])

    obtained = stats.batting_average(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 0.5)


def test_batting_average_counts_a_tie_as_a_hit():
    # hits: 0.05 >= 0.05 yes (tie), 0.10 >= 0.20 no, -0.10 >= -0.05 no
    # -> 1 of 3; dropping the tie would give 0 of 3
    returns = np.array([0.05, 0.10, -0.10])
    benchmark = np.array([0.05, 0.20, -0.05])

    obtained = stats.batting_average(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 1 / 3)


def test_batting_average_is_one_when_the_asset_never_trails():
    returns = np.array([0.10, -0.05, 0.10])
    benchmark = np.array([0.05, -0.10, 0.10])

    np.testing.assert_almost_equal(
        stats.batting_average(returns, benchmark), 1.0
    )


def test_batting_average_is_zero_when_the_asset_never_beats_the_benchmark():
    returns = np.array([0.01, -0.20, 0.02])
    benchmark = np.array([0.05, -0.10, 0.10])

    np.testing.assert_almost_equal(
        stats.batting_average(returns, benchmark), 0.0
    )


def test_batting_average_counts_only_the_pairwise_complete_rows():
    # complete rows are 1 and 3: 0.10 >= 0.05 hit, 0.02 >= 0.05 miss
    # -> 1 of 2 -> 0.5, the other two rows count in neither total
    returns = np.array([np.nan, 0.10, -0.05, 0.02])
    benchmark = np.array([0.05, 0.05, np.nan, 0.05])

    obtained = stats.batting_average(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 0.5)


def test_batting_average_ignores_the_size_of_the_win_or_the_loss():
    # one huge win and two tiny losses -> 1 of 3, the same as one small win
    returns = np.array([5.0, 0.04, 0.04])
    benchmark = np.array([0.05, 0.05, 0.05])

    obtained = stats.batting_average(returns, benchmark)

    np.testing.assert_almost_equal(obtained, 1 / 3)
