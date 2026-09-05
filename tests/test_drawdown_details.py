"""Specification of the maximum drawdown details.

Peak, valley and recovery of the maximum relative drawdown, its durations,
the longest time under water and Morningstar's Average Drawdown.
"""
from quantkit import stats
from quantkit.conventions import BYEAR

import pytest
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# shared data, every expected value below is worked out from these comments

# pos:               0     1     2     3     4     5
PRICES = np.array([10.0, 8.0, 6.0, 9.0, 10.0, 7.0])
# running max        10    10    10    10    10    10
# drawdown            0  -0.2  -0.4  -0.1     0  -0.3
# peak at pos 0 (10), valley at pos 2 (6), recovers at pos 4 (10 >= 10)
# -> duration 2, recovery duration 2; the later drop to 7 (-0.3) is smaller.
# under water at pos 1-3 (3 periods) and at pos 5 (1 period) -> longest 3.

# pos:                   0    1    2     3     4     5
PRICES_NEVER_BACK = np.array([5.0, 6.0, 3.0, 4.0, 4.0, 4.0])
# running max            5    6    6     6     6     6
# drawdown               0    0  -0.5  -1/3  -1/3  -1/3
# peak at pos 1 (6), valley at pos 2 (3), never back to 6
# -> duration 1, recovery NaN; under water from pos 2 to the end -> longest 4.

PRICES_2D = np.column_stack([PRICES, PRICES_NEVER_BACK])

# pos 0 -> 2020-01-01, pos 2 -> 2020-01-03, pos 4 -> 2020-01-05
INDEX = pd.date_range("2020-01-01", periods=6, freq="D")
SERIES = pd.Series(PRICES, index=INDEX, name="px")
FRAME = pd.DataFrame({"a": PRICES, "b": PRICES_NEVER_BACK}, index=INDEX)

# two leading NaN shift every position of PRICES by two
# pos:                          0       1      2    3    4    5     6    7
PRICES_LEADING_NAN = np.array(
    [np.nan, np.nan, 10.0, 8.0, 6.0, 9.0, 10.0, 7.0]
)
# -> peak 2, valley 4, recovery 6; durations and longest stretch unchanged
INDEX_8 = pd.date_range("2020-01-01", periods=8, freq="D")
SERIES_LEADING_NAN = pd.Series(PRICES_LEADING_NAN, index=INDEX_8, name="px")
# labels: peak 2020-01-03, valley 2020-01-05, recovery 2020-01-07

# pos:                         0      1     2    3      4      5    6
PRICES_MIDDLE_NAN = np.array([10.0, np.nan, 6.0, 9.0, np.nan, 10.0, 7.0])
# running max                  10   (10)   10   10    (10)    10   10
# drawdown                      0    nan  -0.4 -0.1    nan     0 -0.3
# peak 0, valley 2, recovery 5 (the NaN at pos 4 is never chosen)
# -> duration 2, recovery duration 3 (positions: the NaN slot is a period)
# under water: 6, 9 are 2 valid observations in a row, then 7 -> longest 2
INDEX_7 = pd.date_range("2020-01-01", periods=7, freq="D")
SERIES_MIDDLE_NAN = pd.Series(PRICES_MIDDLE_NAN, index=INDEX_7, name="px")

PRICES_RISING = np.array([1.0, 2.0, 3.0, 4.0])  # never below the running max
PRICES_FLAT = np.array([5.0, 5.0, 5.0])  # never *strictly* below it

# pos:                   0    1    2    3
PRICES_TIED = np.array([10.0, 6.0, 8.0, 6.0])
# drawdown               0 -0.4 -0.2 -0.4  -> the first minimum, pos 1, wins
# peak 0, valley 1, never back to 10 -> duration 1, under water 3 periods

# pos:                      0    1     2    3
PRICES_RETOUCH = np.array([10.0, 8.0, 10.0, 6.0])
# drawdown                  0 -0.2     0 -0.4
# the running max 10 is touched again at pos 2 (drawdown 0), so the deepest
# drawdown starts there: peak 2, valley 3 -> duration 1

# pos:                          0    1    2    3     4    5     6
PRICES_LONGEST_SHALLOW = np.array([10.0, 9.0, 9.0, 9.0, 10.0, 2.0, 10.0])
# drawdown                      0 -0.1 -0.1 -0.1     0 -0.8     0
# deepest: peak 4, valley 5, recovery 6 -> duration 1, recovery duration 1
# longest under water: pos 1-3 -> 3 periods (the shallow one)

DETAIL_FUNCTIONS = [
    "max_drawdown_peak",
    "max_drawdown_valley",
    "max_drawdown_recovery",
    "max_drawdown_duration",
    "max_drawdown_recovery_duration",
    "longest_drawdown_duration",
    "average_drawdown",
]


# -----------------------------------------------------------------------------
# max_drawdown_peak


def test_max_drawdown_peak_is_the_position_as_a_float_for_numpy():
    obtained = stats.max_drawdown_peak(PRICES)
    assert obtained == 0
    assert isinstance(obtained, float)


def test_max_drawdown_peak_is_computed_per_column_for_2d_numpy():
    obtained = stats.max_drawdown_peak(PRICES_2D)
    np.testing.assert_equal(obtained, np.array([0.0, 1.0]))
    assert obtained.dtype == np.float64


def test_max_drawdown_peak_is_the_index_label_for_a_series():
    obtained = stats.max_drawdown_peak(SERIES)
    assert obtained == pd.Timestamp("2020-01-01")


def test_max_drawdown_peak_frame_returns_labels_indexed_by_columns():
    obtained = stats.max_drawdown_peak(FRAME)
    expected = pd.Series([INDEX[0], INDEX[1]], index=FRAME.columns)
    pd.testing.assert_series_equal(obtained, expected)


def test_max_drawdown_peak_position_is_shifted_by_leading_nan():
    obtained = stats.max_drawdown_peak(PRICES_LEADING_NAN)
    assert obtained == 2


def test_max_drawdown_peak_label_is_correct_with_leading_nan():
    obtained = stats.max_drawdown_peak(SERIES_LEADING_NAN)
    assert obtained == pd.Timestamp("2020-01-03")


def test_max_drawdown_peak_ignores_nan_in_the_middle():
    obtained = stats.max_drawdown_peak(PRICES_MIDDLE_NAN)
    assert obtained == 0


def test_max_drawdown_peak_is_nan_when_there_is_no_drawdown():
    assert np.isnan(stats.max_drawdown_peak(PRICES_RISING))
    assert np.isnan(stats.max_drawdown_peak(PRICES_FLAT))


def test_max_drawdown_peak_is_missing_for_a_series_without_drawdown():
    prices = pd.Series(PRICES_RISING, index=INDEX[:4], name="px")
    assert pd.isna(stats.max_drawdown_peak(prices))


def test_max_drawdown_peak_is_the_last_touch_of_the_running_max():
    obtained = stats.max_drawdown_peak(PRICES_RETOUCH)
    assert obtained == 2


def test_max_drawdown_peak_of_a_tied_minimum_belongs_to_the_first_one():
    obtained = stats.max_drawdown_peak(PRICES_TIED)
    assert obtained == 0


# -----------------------------------------------------------------------------
# max_drawdown_valley


def test_max_drawdown_valley_is_the_position_as_a_float_for_numpy():
    obtained = stats.max_drawdown_valley(PRICES)
    assert obtained == 2
    assert isinstance(obtained, float)


def test_max_drawdown_valley_is_computed_per_column_for_2d_numpy():
    obtained = stats.max_drawdown_valley(PRICES_2D)
    np.testing.assert_equal(obtained, np.array([2.0, 2.0]))


def test_max_drawdown_valley_is_the_index_label_for_a_series():
    obtained = stats.max_drawdown_valley(SERIES)
    assert obtained == pd.Timestamp("2020-01-03")


def test_max_drawdown_valley_frame_returns_labels_indexed_by_columns():
    obtained = stats.max_drawdown_valley(FRAME)
    expected = pd.Series([INDEX[2], INDEX[2]], index=FRAME.columns)
    pd.testing.assert_series_equal(obtained, expected)


def test_max_drawdown_valley_position_is_shifted_by_leading_nan():
    obtained = stats.max_drawdown_valley(PRICES_LEADING_NAN)
    assert obtained == 4


def test_max_drawdown_valley_label_is_correct_with_leading_nan():
    obtained = stats.max_drawdown_valley(SERIES_LEADING_NAN)
    assert obtained == pd.Timestamp("2020-01-05")


def test_max_drawdown_valley_is_never_a_nan_position():
    obtained = stats.max_drawdown_valley(PRICES_MIDDLE_NAN)
    assert obtained == 2


def test_max_drawdown_valley_is_nan_when_there_is_no_drawdown():
    assert np.isnan(stats.max_drawdown_valley(PRICES_RISING))
    assert np.isnan(stats.max_drawdown_valley(PRICES_FLAT))


def test_max_drawdown_valley_picks_the_first_of_two_equal_minima():
    obtained = stats.max_drawdown_valley(PRICES_TIED)
    assert obtained == 1


def test_max_drawdown_valley_ignores_a_smaller_later_drawdown():
    # the final drop to 7 (-0.3) after the recovery does not move the valley
    obtained = stats.max_drawdown_valley(PRICES)
    assert obtained == 2


# -----------------------------------------------------------------------------
# max_drawdown_recovery


def test_max_drawdown_recovery_is_the_position_as_a_float_for_numpy():
    obtained = stats.max_drawdown_recovery(PRICES)
    assert obtained == 4
    assert isinstance(obtained, float)


def test_max_drawdown_recovery_is_nan_when_price_never_regains_the_peak():
    assert np.isnan(stats.max_drawdown_recovery(PRICES_NEVER_BACK))


def test_max_drawdown_recovery_is_computed_per_column_for_2d_numpy():
    obtained = stats.max_drawdown_recovery(PRICES_2D)
    np.testing.assert_equal(obtained, np.array([4.0, np.nan]))


def test_max_drawdown_recovery_is_the_index_label_for_a_series():
    obtained = stats.max_drawdown_recovery(SERIES)
    assert obtained == pd.Timestamp("2020-01-05")


def test_max_drawdown_recovery_is_missing_for_a_series_never_recovered():
    prices = pd.Series(PRICES_NEVER_BACK, index=INDEX, name="px")
    assert pd.isna(stats.max_drawdown_recovery(prices))


def test_max_drawdown_recovery_frame_marks_never_recovered_columns_nat():
    obtained = stats.max_drawdown_recovery(FRAME)
    expected = pd.Series([INDEX[4], pd.NaT], index=FRAME.columns)
    pd.testing.assert_series_equal(obtained, expected)


def test_max_drawdown_recovery_position_is_shifted_by_leading_nan():
    obtained = stats.max_drawdown_recovery(PRICES_LEADING_NAN)
    assert obtained == 6


def test_max_drawdown_recovery_label_is_correct_with_leading_nan():
    obtained = stats.max_drawdown_recovery(SERIES_LEADING_NAN)
    assert obtained == pd.Timestamp("2020-01-07")


def test_max_drawdown_recovery_skips_nan_in_the_middle():
    assert stats.max_drawdown_recovery(PRICES_MIDDLE_NAN) == 5
    obtained = stats.max_drawdown_recovery(SERIES_MIDDLE_NAN)
    assert obtained == pd.Timestamp("2020-01-06")


def test_max_drawdown_recovery_happens_when_price_equals_the_peak():
    # pos 4 is exactly 10, the peak price: that already counts as recovered
    assert stats.max_drawdown_recovery(PRICES) == 4


def test_max_drawdown_recovery_requires_price_at_or_above_the_peak():
    # pos:              0     1    2     3
    prices = np.array([10.0, 6.0, 9.0, 12.0])
    # 9 is still below the peak 10, 12 is the first price >= 10 -> pos 3
    assert stats.max_drawdown_recovery(prices) == 3


def test_max_drawdown_recovery_is_nan_when_there_is_no_drawdown():
    assert np.isnan(stats.max_drawdown_recovery(PRICES_RISING))
    assert np.isnan(stats.max_drawdown_recovery(PRICES_FLAT))


# -----------------------------------------------------------------------------
# max_drawdown_duration


def test_max_drawdown_duration_is_valley_minus_peak_in_periods():
    np.testing.assert_almost_equal(stats.max_drawdown_duration(PRICES), 2)


def test_max_drawdown_duration_is_computed_per_column_for_2d_numpy():
    obtained = stats.max_drawdown_duration(PRICES_2D)
    np.testing.assert_almost_equal(obtained, np.array([2.0, 1.0]))


def test_max_drawdown_duration_of_a_series_is_a_number():
    np.testing.assert_almost_equal(stats.max_drawdown_duration(SERIES), 2)


def test_max_drawdown_duration_frame_is_a_series_indexed_by_columns():
    obtained = stats.max_drawdown_duration(FRAME)
    expected = pd.Series([2.0, 1.0], index=FRAME.columns)
    pd.testing.assert_series_equal(obtained, expected)


def test_max_drawdown_duration_is_unchanged_by_leading_nan():
    np.testing.assert_almost_equal(
        stats.max_drawdown_duration(PRICES_LEADING_NAN), 2
    )
    np.testing.assert_almost_equal(
        stats.max_drawdown_duration(SERIES_LEADING_NAN), 2
    )


def test_max_drawdown_duration_counts_positions_across_nan_in_the_middle():
    np.testing.assert_almost_equal(
        stats.max_drawdown_duration(PRICES_MIDDLE_NAN), 2
    )


def test_max_drawdown_duration_is_zero_when_there_is_no_drawdown():
    assert stats.max_drawdown_duration(PRICES_RISING) == 0
    assert stats.max_drawdown_duration(PRICES_FLAT) == 0


def test_max_drawdown_duration_of_a_tied_minimum_uses_the_first_one():
    assert stats.max_drawdown_duration(PRICES_TIED) == 1


def test_max_drawdown_duration_starts_at_the_last_touch_of_the_running_max():
    assert stats.max_drawdown_duration(PRICES_RETOUCH) == 1


def test_max_drawdown_duration_is_nan_for_an_all_nan_column():
    prices = np.column_stack([PRICES, np.full(6, np.nan)])
    obtained = stats.max_drawdown_duration(prices)
    np.testing.assert_equal(obtained, np.array([2.0, np.nan]))


# -----------------------------------------------------------------------------
# max_drawdown_recovery_duration


def test_max_drawdown_recovery_duration_is_recovery_minus_valley():
    np.testing.assert_almost_equal(
        stats.max_drawdown_recovery_duration(PRICES), 2
    )


def test_max_drawdown_recovery_duration_is_nan_when_never_recovered():
    assert np.isnan(stats.max_drawdown_recovery_duration(PRICES_NEVER_BACK))


def test_max_drawdown_recovery_duration_is_computed_per_column_for_2d():
    obtained = stats.max_drawdown_recovery_duration(PRICES_2D)
    np.testing.assert_equal(obtained, np.array([2.0, np.nan]))


def test_max_drawdown_recovery_duration_of_a_series_is_a_number():
    np.testing.assert_almost_equal(
        stats.max_drawdown_recovery_duration(SERIES), 2
    )


def test_max_drawdown_recovery_duration_frame_is_indexed_by_columns():
    obtained = stats.max_drawdown_recovery_duration(FRAME)
    expected = pd.Series([2.0, np.nan], index=FRAME.columns)
    pd.testing.assert_series_equal(obtained, expected)


def test_max_drawdown_recovery_duration_is_unchanged_by_leading_nan():
    np.testing.assert_almost_equal(
        stats.max_drawdown_recovery_duration(PRICES_LEADING_NAN), 2
    )


def test_max_drawdown_recovery_duration_counts_a_nan_slot_as_a_period():
    # valley at pos 2, recovery at pos 5 -> 3 periods, one of them NaN
    np.testing.assert_almost_equal(
        stats.max_drawdown_recovery_duration(PRICES_MIDDLE_NAN), 3
    )


def test_max_drawdown_recovery_duration_is_nan_when_there_is_no_drawdown():
    assert np.isnan(stats.max_drawdown_recovery_duration(PRICES_RISING))
    assert np.isnan(stats.max_drawdown_recovery_duration(PRICES_FLAT))


def test_max_drawdown_recovery_duration_ignores_a_smaller_later_drawdown():
    # recovered at pos 4; the later drop to 7 opens a new, smaller drawdown
    np.testing.assert_almost_equal(
        stats.max_drawdown_recovery_duration(PRICES), 2
    )


def test_max_drawdown_recovery_duration_of_the_shallow_stretch_is_not_used():
    # deepest drawdown: valley 5, recovery 6 -> 1 period
    np.testing.assert_almost_equal(
        stats.max_drawdown_recovery_duration(PRICES_LONGEST_SHALLOW), 1
    )


# -----------------------------------------------------------------------------
# longest_drawdown_duration


def test_longest_drawdown_duration_counts_consecutive_periods_under_water():
    np.testing.assert_almost_equal(stats.longest_drawdown_duration(PRICES), 3)


def test_longest_drawdown_duration_can_be_the_shallow_stretch():
    obtained = stats.longest_drawdown_duration(PRICES_LONGEST_SHALLOW)
    np.testing.assert_almost_equal(obtained, 3)
    # while the deepest drawdown only lasts one period
    assert stats.max_drawdown_duration(PRICES_LONGEST_SHALLOW) == 1


def test_longest_drawdown_duration_counts_an_open_stretch_at_the_end():
    # under water from pos 2 to pos 5 without recovering -> 4 periods
    np.testing.assert_almost_equal(
        stats.longest_drawdown_duration(PRICES_NEVER_BACK), 4
    )


def test_longest_drawdown_duration_is_computed_per_column_for_2d_numpy():
    obtained = stats.longest_drawdown_duration(PRICES_2D)
    np.testing.assert_almost_equal(obtained, np.array([3.0, 4.0]))


def test_longest_drawdown_duration_of_a_series_is_a_number():
    np.testing.assert_almost_equal(stats.longest_drawdown_duration(SERIES), 3)


def test_longest_drawdown_duration_frame_is_indexed_by_columns():
    obtained = stats.longest_drawdown_duration(FRAME)
    expected = pd.Series([3.0, 4.0], index=FRAME.columns)
    pd.testing.assert_series_equal(obtained, expected)


def test_longest_drawdown_duration_is_unchanged_by_leading_nan():
    np.testing.assert_almost_equal(
        stats.longest_drawdown_duration(PRICES_LEADING_NAN), 3
    )


def test_longest_drawdown_duration_ignores_nan_inside_a_stretch():
    # pos:              0     1      2     3     4
    prices = np.array([10.0, 8.0, np.nan, 6.0, 10.0])
    # drawdown          0  -0.2    nan  -0.4     0
    # 8 and 6 are consecutive valid observations under water -> 2, the NaN
    # neither breaks the stretch (1) nor extends it (3)
    np.testing.assert_almost_equal(stats.longest_drawdown_duration(prices), 2)
    np.testing.assert_almost_equal(
        stats.longest_drawdown_duration(PRICES_MIDDLE_NAN), 2
    )


def test_longest_drawdown_duration_is_zero_when_there_is_no_drawdown():
    assert stats.longest_drawdown_duration(PRICES_RISING) == 0
    assert stats.longest_drawdown_duration(PRICES_FLAT) == 0


def test_longest_drawdown_duration_counts_a_tied_valley_stretch_fully():
    # under water at pos 1, 2 and 3 -> 3 periods
    np.testing.assert_almost_equal(
        stats.longest_drawdown_duration(PRICES_TIED), 3
    )


def test_longest_drawdown_duration_is_nan_for_an_all_nan_column():
    prices = np.column_stack([PRICES, np.full(6, np.nan)])
    obtained = stats.longest_drawdown_duration(prices)
    np.testing.assert_equal(obtained, np.array([3.0, np.nan]))


# -----------------------------------------------------------------------------
# average_drawdown


def test_average_drawdown_averages_the_max_drawdown_of_exact_blocks():
    # periods_per_year=3 -> blocks [10, 8, 6] -> -0.4 and [9, 10, 7] -> -0.3
    # sum -0.7 over 6 / 3 = 2 years -> -0.35
    obtained = stats.average_drawdown(PRICES, periods_per_year=3)
    np.testing.assert_almost_equal(obtained, -0.35)


def test_average_drawdown_weights_a_partial_last_block_by_its_length():
    # pos:              0     1    2    3     4    5    6
    prices = np.array([10.0, 8.0, 6.0, 9.0, 10.0, 7.0, 5.0])
    # blocks of 3: [10, 8, 6] -> -0.4, [9, 10, 7] -> -0.3, [5] -> 0
    # sum -0.7 over 7 / 3 years -> -0.7 * 3 / 7 = -0.3
    obtained = stats.average_drawdown(prices, periods_per_year=3)
    np.testing.assert_almost_equal(obtained, -0.3)


def test_average_drawdown_equals_max_drawdown_for_exactly_one_block():
    # pos:              0     1    2    3
    prices = np.array([10.0, 8.0, 6.0, 9.0])
    # one block of 4 -> -0.4 over 4 / 4 = 1 year -> -0.4
    obtained = stats.average_drawdown(prices, periods_per_year=4)
    np.testing.assert_almost_equal(obtained, -0.4)
    np.testing.assert_almost_equal(obtained, stats.max_drawdown(prices))


def test_average_drawdown_annualizes_a_block_shorter_than_a_year():
    # Morningstar generalization: sum(mdd_t) / (n / periods_per_year)
    # pos:              0     1    2    3
    prices = np.array([10.0, 8.0, 6.0, 9.0])
    # one block -> -0.4 over 4 / 8 = 0.5 years -> -0.8
    obtained = stats.average_drawdown(prices, periods_per_year=8)
    np.testing.assert_almost_equal(obtained, -0.8)


def test_average_drawdown_blocks_start_their_own_running_max():
    # pos:              0     1    2    3    4    5
    prices = np.array([10.0, 5.0, 5.0, 6.0, 6.0, 6.0])
    # [10, 5, 5] -> -0.5; [6, 6, 6] -> 0 (never below 6, although below 10)
    # sum -0.5 over 6 / 3 = 2 years -> -0.25, not the global -0.5
    obtained = stats.average_drawdown(prices, periods_per_year=3)
    np.testing.assert_almost_equal(obtained, -0.25)
    np.testing.assert_almost_equal(stats.max_drawdown(prices), -0.5)


def test_average_drawdown_is_zero_with_one_period_per_year():
    # every block is a single observation, so every block drawdown is 0
    obtained = stats.average_drawdown(PRICES, periods_per_year=1)
    np.testing.assert_almost_equal(obtained, 0)


def test_average_drawdown_defaults_to_byear_periods_per_year():
    # 6 < BYEAR observations -> one block -0.4 over 6 / BYEAR years
    obtained = stats.average_drawdown(PRICES)
    expected = -0.4 / (6 / BYEAR)
    np.testing.assert_almost_equal(obtained, expected)


def test_average_drawdown_is_computed_per_column_for_2d_numpy():
    # column b: [5, 6, 3] -> -0.5, [4, 4, 4] -> 0 -> -0.5 / 2 = -0.25
    obtained = stats.average_drawdown(PRICES_2D, periods_per_year=3)
    np.testing.assert_almost_equal(obtained, np.array([-0.35, -0.25]))


def test_average_drawdown_of_a_series_is_a_number():
    obtained = stats.average_drawdown(SERIES, periods_per_year=3)
    np.testing.assert_almost_equal(obtained, -0.35)


def test_average_drawdown_frame_is_a_series_indexed_by_columns():
    obtained = stats.average_drawdown(FRAME, periods_per_year=3)
    expected = pd.Series([-0.35, -0.25], index=FRAME.columns)
    pd.testing.assert_series_equal(obtained, expected)


def test_average_drawdown_drops_leading_nan_before_forming_blocks():
    # 6 valid observations -> the same two blocks as PRICES -> -0.35
    obtained = stats.average_drawdown(PRICES_LEADING_NAN, periods_per_year=3)
    np.testing.assert_almost_equal(obtained, -0.35)


def test_average_drawdown_forms_blocks_from_valid_observations_only():
    # pos:              0     1      2     3    4     5    6
    prices = np.array([10.0, 8.0, np.nan, 6.0, 9.0, 10.0, 7.0])
    # valid: [10, 8, 6, 9, 10, 7] -> the same two blocks as PRICES -> -0.35
    obtained = stats.average_drawdown(prices, periods_per_year=3)
    np.testing.assert_almost_equal(obtained, -0.35)


def test_average_drawdown_is_zero_when_there_is_no_drawdown():
    # blocks [1, 2, 3] -> 0 and [4] -> 0
    obtained = stats.average_drawdown(PRICES_RISING, periods_per_year=3)
    np.testing.assert_almost_equal(obtained, 0)


def test_average_drawdown_is_nan_for_an_all_nan_column():
    prices = np.column_stack([PRICES, np.full(6, np.nan)])
    obtained = stats.average_drawdown(prices, periods_per_year=3)
    np.testing.assert_equal(obtained, np.array([-0.35, np.nan]))


# -----------------------------------------------------------------------------
# empty and all-NaN input


@pytest.mark.parametrize("func_name", DETAIL_FUNCTIONS)
def test_empty_numpy_input_returns_nan(func_name):
    func = getattr(stats, func_name)
    assert np.isnan(func(np.array([])))


@pytest.mark.parametrize("func_name", DETAIL_FUNCTIONS)
def test_all_nan_numpy_input_returns_nan(func_name):
    func = getattr(stats, func_name)
    assert np.isnan(func(np.array([np.nan, np.nan, np.nan])))


@pytest.mark.parametrize("func_name", DETAIL_FUNCTIONS)
def test_all_nan_2d_numpy_input_returns_nan_per_column(func_name):
    func = getattr(stats, func_name)
    obtained = func(np.full((3, 2), np.nan))
    assert obtained.shape == (2,)
    assert np.isnan(obtained).all()


@pytest.mark.parametrize("func_name", DETAIL_FUNCTIONS)
def test_empty_series_returns_missing(func_name):
    func = getattr(stats, func_name)
    assert pd.isna(func(pd.Series([], dtype=float)))


@pytest.mark.parametrize("func_name", DETAIL_FUNCTIONS)
def test_all_nan_series_returns_missing(func_name):
    func = getattr(stats, func_name)
    prices = pd.Series(np.full(3, np.nan), index=INDEX[:3], name="px")
    assert pd.isna(func(prices))


@pytest.mark.parametrize("func_name", DETAIL_FUNCTIONS)
def test_all_nan_frame_returns_missing_per_column(func_name):
    func = getattr(stats, func_name)
    prices = pd.DataFrame(np.full((3, 2), np.nan), index=INDEX[:3])
    prices.columns = ["a", "b"]
    obtained = func(prices)
    pd.testing.assert_index_equal(obtained.index, prices.columns)
    assert obtained.isna().all()
