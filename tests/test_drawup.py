"""Drawup and maximum drawup tests.

The drawup of a price series is its rise from the running minimum, the
mirror image of the drawdown. Every expected value is worked out by hand in
a comment next to its input.
"""
from quantkit import expanding, stats

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# expanding.drawup


def test_drawup_absolute_is_the_rise_from_the_running_minimum():
    prices = np.array([4, 2, 3, 6])
    # cummin: 4, 2, 2, 2 -> absolute drawup 0, 0, 1, 4
    expected = np.array([0.0, 0.0, 1.0, 4.0])

    obtained = expanding.drawup(prices, relative=False)

    np.testing.assert_almost_equal(obtained, expected)


def test_drawup_relative_divides_the_rise_by_the_running_minimum():
    prices = np.array([4, 2, 3, 6])
    # cummin: 4, 2, 2, 2 -> relative drawup 0/4, 0/2, 1/2, 4/2
    expected = np.array([0.0, 0.0, 0.5, 2.0])

    obtained = expanding.drawup(prices, relative=True)

    np.testing.assert_almost_equal(obtained, expected)


def test_drawup_is_relative_by_default():
    prices = np.array([4, 2, 3, 6])
    # cummin: 4, 2, 2, 2 -> relative drawup 0, 0, 0.5, 2.0
    expected = np.array([0.0, 0.0, 0.5, 2.0])

    obtained = expanding.drawup(prices)

    np.testing.assert_almost_equal(obtained, expected)


def test_drawup_is_zero_when_prices_only_fall():
    prices = np.array([5, 4, 3, 1])
    # cummin is the price itself at every step -> drawup 0, 0, 0, 0
    expected = np.zeros(4)

    absolute = expanding.drawup(prices, relative=False)
    relative = expanding.drawup(prices, relative=True)

    np.testing.assert_almost_equal(absolute, expected)
    np.testing.assert_almost_equal(relative, expected)


def test_drawup_grows_from_zero_when_prices_only_rise():
    prices = np.array([2, 3, 5, 8])
    # cummin: 2, 2, 2, 2 -> absolute 0, 1, 3, 6 ; relative 0, 0.5, 1.5, 3.0
    expected_absolute = np.array([0.0, 1.0, 3.0, 6.0])
    expected_relative = np.array([0.0, 0.5, 1.5, 3.0])

    absolute = expanding.drawup(prices, relative=False)
    relative = expanding.drawup(prices, relative=True)

    np.testing.assert_almost_equal(absolute, expected_absolute)
    np.testing.assert_almost_equal(relative, expected_relative)


def test_drawup_keeps_a_leading_nan():
    prices = np.array([np.nan, 4, 2, 3])
    # cummin: nan, 4, 2, 2 -> absolute nan, 0, 0, 1 ; relative nan, 0, 0, 0.5
    expected_absolute = np.array([np.nan, 0.0, 0.0, 1.0])
    expected_relative = np.array([np.nan, 0.0, 0.0, 0.5])

    absolute = expanding.drawup(prices, relative=False)
    relative = expanding.drawup(prices, relative=True)

    np.testing.assert_almost_equal(absolute, expected_absolute)
    np.testing.assert_almost_equal(relative, expected_relative)


def test_drawup_keeps_nan_in_the_middle_without_resetting_the_minimum():
    prices = np.array([2, np.nan, 3, 6])
    # cummin: 2, nan, 2, 2 (the 2 survives the gap, a reset would give 3)
    # -> absolute 0, nan, 1, 4 ; relative 0, nan, 0.5, 2.0
    expected_absolute = np.array([0.0, np.nan, 1.0, 4.0])
    expected_relative = np.array([0.0, np.nan, 0.5, 2.0])

    absolute = expanding.drawup(prices, relative=False)
    relative = expanding.drawup(prices, relative=True)

    np.testing.assert_almost_equal(absolute, expected_absolute)
    np.testing.assert_almost_equal(relative, expected_relative)


def test_drawup_is_all_nan_when_input_is_all_nan():
    prices = np.array([np.nan, np.nan, np.nan])
    expected = np.array([np.nan, np.nan, np.nan])

    absolute = expanding.drawup(prices, relative=False)
    relative = expanding.drawup(prices, relative=True)

    np.testing.assert_almost_equal(absolute, expected)
    np.testing.assert_almost_equal(relative, expected)


def test_drawup_is_empty_when_input_is_empty():
    prices = np.array([])

    obtained = expanding.drawup(prices)

    assert len(obtained) == 0


def test_drawup_relative_is_nan_where_the_running_minimum_is_zero():
    prices = np.array([2, 0, 1])
    # cummin: 2, 0, 0 -> absolute 0, 0, 1
    # relative 0/2, 0/0, 1/0 -> 0, nan, nan (never inf)
    expected_absolute = np.array([0.0, 0.0, 1.0])
    expected_relative = np.array([0.0, np.nan, np.nan])

    absolute = expanding.drawup(prices, relative=False)
    relative = expanding.drawup(prices, relative=True)

    np.testing.assert_almost_equal(absolute, expected_absolute)
    np.testing.assert_almost_equal(relative, expected_relative)


def test_drawup_2d_treats_each_column_independently():
    prices = np.array([[4, 5], [2, 4], [3, 3], [6, 1]])
    # column 0 dips then rises: cummin 4, 2, 2, 2 -> absolute 0, 0, 1, 4
    # column 1 only falls:      cummin 5, 4, 3, 1 -> absolute 0, 0, 0, 0
    expected_absolute = np.array(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [4.0, 0.0]]
    )
    expected_relative = np.array(
        [[0.0, 0.0], [0.0, 0.0], [0.5, 0.0], [2.0, 0.0]]
    )

    absolute = expanding.drawup(prices, relative=False)
    relative = expanding.drawup(prices, relative=True)

    np.testing.assert_almost_equal(absolute, expected_absolute)
    np.testing.assert_almost_equal(relative, expected_relative)


def test_drawup_series_keeps_index_and_name():
    index = pd.date_range("2020", periods=4)
    prices = pd.Series([4, 2, 3, 6], index=index, name="px")
    # cummin: 4, 2, 2, 2 -> absolute drawup 0, 0, 1, 4
    expected = pd.Series([0.0, 0.0, 1.0, 4.0], index=index, name="px")

    obtained = expanding.drawup(prices, relative=False)

    pd.testing.assert_series_equal(obtained, expected)


def test_drawup_dataframe_keeps_index_and_columns():
    index = pd.date_range("2020", periods=4)
    prices = pd.DataFrame(
        [[4, 5], [2, 4], [3, 3], [6, 1]], index=index, columns=["a", "b"]
    )
    # a: cummin 4, 2, 2, 2 -> 0, 0, 1, 4 ; b: only falls -> 0, 0, 0, 0
    expected = pd.DataFrame(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [4.0, 0.0]],
        index=index,
        columns=["a", "b"],
    )

    obtained = expanding.drawup(prices, relative=False)

    pd.testing.assert_frame_equal(obtained, expected)


def test_drawup_fills_out_in_place_for_numpy_input():
    prices = np.array([4.0, 2.0, 3.0, 6.0])
    out = np.empty(4)
    # cummin: 4, 2, 2, 2 -> absolute drawup 0, 0, 1, 4
    expected = np.array([0.0, 0.0, 1.0, 4.0])

    expanding.drawup(prices, relative=False, out=out)

    np.testing.assert_almost_equal(out, expected)


# ---------------------------------------------------------------------------
# stats.max_drawup


def test_max_drawup_absolute_is_the_largest_rise_from_the_running_minimum():
    prices = np.array([4, 2, 3, 6])
    # absolute drawup 0, 0, 1, 4 -> max 4
    expected = 4.0

    obtained = stats.max_drawup(prices, relative=False)

    np.testing.assert_almost_equal(obtained, expected)


def test_max_drawup_relative_is_the_largest_relative_rise():
    prices = np.array([4, 2, 3, 6])
    # relative drawup 0, 0, 0.5, 2.0 -> max 2.0
    expected = 2.0

    obtained = stats.max_drawup(prices, relative=True)

    np.testing.assert_almost_equal(obtained, expected)


def test_max_drawup_is_relative_by_default():
    prices = np.array([4, 2, 3, 6])
    # relative drawup 0, 0, 0.5, 2.0 -> max 2.0
    expected = 2.0

    obtained = stats.max_drawup(prices)

    np.testing.assert_almost_equal(obtained, expected)


def test_max_drawup_is_zero_when_prices_only_fall():
    prices = np.array([5, 4, 3, 1])
    # drawup 0, 0, 0, 0 -> max 0

    absolute = stats.max_drawup(prices, relative=False)
    relative = stats.max_drawup(prices, relative=True)

    np.testing.assert_almost_equal(absolute, 0.0)
    np.testing.assert_almost_equal(relative, 0.0)


def test_max_drawup_ignores_a_leading_nan():
    prices = np.array([np.nan, 4, 2, 3])
    # drawup nan, 0, 0, 1 -> absolute max 1 ; relative nan, 0, 0, 0.5 -> 0.5

    absolute = stats.max_drawup(prices, relative=False)
    relative = stats.max_drawup(prices, relative=True)

    np.testing.assert_almost_equal(absolute, 1.0)
    np.testing.assert_almost_equal(relative, 0.5)


def test_max_drawup_ignores_nan_in_the_middle():
    prices = np.array([2, np.nan, 3, 6])
    # drawup 0, nan, 1, 4 -> absolute max 4 ; relative 0, nan, 0.5, 2 -> 2

    absolute = stats.max_drawup(prices, relative=False)
    relative = stats.max_drawup(prices, relative=True)

    np.testing.assert_almost_equal(absolute, 4.0)
    np.testing.assert_almost_equal(relative, 2.0)


def test_max_drawup_is_nan_when_input_is_all_nan():
    prices_1d = np.array([np.nan, np.nan, np.nan])
    prices_2d = np.array([[np.nan, np.nan], [np.nan, np.nan]])

    obtained_1d = stats.max_drawup(prices_1d)
    obtained_2d = stats.max_drawup(prices_2d)

    assert np.isnan(obtained_1d)
    np.testing.assert_almost_equal(obtained_2d, [np.nan, np.nan])


def test_max_drawup_is_nan_when_input_is_empty():
    obtained_numpy = stats.max_drawup(np.array([]))
    obtained_pandas = stats.max_drawup(pd.Series([], dtype=float))

    assert np.isnan(obtained_numpy)
    assert np.isnan(obtained_pandas)


def test_max_drawup_relative_is_nan_when_prices_start_at_zero():
    prices = np.array([0, 1, 2])
    # cummin 0, 0, 0 -> relative drawup nan, nan, nan -> max nan (never inf)
    # absolute drawup 0, 1, 2 -> max 2

    relative = stats.max_drawup(prices, relative=True)
    absolute = stats.max_drawup(prices, relative=False)

    assert np.isnan(relative)
    np.testing.assert_almost_equal(absolute, 2.0)


def test_max_drawup_relative_skips_positions_with_zero_running_minimum():
    prices = np.array([2, 0, 1])
    # relative drawup 0, nan, nan -> only the 0 is a valid observation -> 0

    obtained = stats.max_drawup(prices, relative=True)

    np.testing.assert_almost_equal(obtained, 0.0)


def test_max_drawup_2d_reduces_each_column():
    prices = np.array([[4, 5], [2, 4], [3, 3], [6, 1]])
    # column 0: absolute drawup 0, 0, 1, 4 -> 4 ; relative -> 2.0
    # column 1 only falls -> 0 in both cases

    absolute = stats.max_drawup(prices, relative=False)
    relative = stats.max_drawup(prices, relative=True)

    assert isinstance(absolute, np.ndarray)
    np.testing.assert_almost_equal(absolute, [4.0, 0.0])
    np.testing.assert_almost_equal(relative, [2.0, 0.0])


def test_max_drawup_series_returns_a_float():
    prices = pd.Series([4, 2, 3, 6], name="px")
    # absolute drawup 0, 0, 1, 4 -> 4

    obtained = stats.max_drawup(prices, relative=False)

    assert isinstance(obtained, float)
    np.testing.assert_almost_equal(obtained, 4.0)


def test_max_drawup_dataframe_returns_a_series_indexed_by_columns():
    prices = pd.DataFrame(
        [[4, 5], [2, 4], [3, 3], [6, 1]], columns=["a", "b"]
    )
    # a: absolute drawup 0, 0, 1, 4 -> 4 ; b only falls -> 0
    expected = pd.Series([4.0, 0.0], index=["a", "b"])

    obtained = stats.max_drawup(prices, relative=False)

    pd.testing.assert_series_equal(obtained, expected)
