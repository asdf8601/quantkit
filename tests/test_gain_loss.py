"""Tests of the gain and loss statistics.

Every function reduces along axis 0 (one value per column), ignores NaN and
returns NaN, never inf, when there is nothing to average or to divide by.
"""
from quantkit import stats

import pytest
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# average_gain


def test_average_gain_defaults_to_arithmetic_mean_of_positive_returns():
    # gains 0.10, 0.20 -> arithmetic mean 0.15
    rets = np.array([0.10, -0.05, 0.20, -0.10])
    obtained = stats.average_gain(rets)
    np.testing.assert_almost_equal(obtained, 0.15)


def test_average_gain_geo_is_geometric_mean_of_positive_returns():
    # gains 0.10, 0.20 -> sqrt(1.1 * 1.2) - 1 = 0.148912529...
    rets = np.array([0.10, -0.05, 0.20, -0.10])
    obtained = stats.average_gain(rets, method="geo")
    np.testing.assert_almost_equal(obtained, np.sqrt(1.1 * 1.2) - 1)


def test_average_gain_rejects_unknown_method():
    rets = np.array([0.10, -0.05, 0.20])
    with pytest.raises(ValueError):
        stats.average_gain(rets, method="median")


def test_average_gain_ignores_zero_returns():
    # zeros are not gains: gains 0.10, 0.20 -> 0.15, not 0.30 / 4
    rets = np.array([0.0, 0.10, 0.0, 0.20])
    obtained = stats.average_gain(rets)
    np.testing.assert_almost_equal(obtained, 0.15)


def test_average_gain_ignores_nan_at_start_and_in_the_middle():
    # gains 0.10, 0.20 -> 0.15; NaN neither counts nor propagates
    rets = np.array([np.nan, 0.10, np.nan, 0.20, -0.10])
    obtained = stats.average_gain(rets)
    np.testing.assert_almost_equal(obtained, 0.15)


def test_average_gain_returns_nan_when_there_are_no_gains():
    rets = np.array([-0.10, 0.0, -0.20])
    assert np.isnan(stats.average_gain(rets, method="arith"))
    assert np.isnan(stats.average_gain(rets, method="geo"))


def test_average_gain_returns_nan_for_empty_input():
    assert np.isnan(stats.average_gain(np.array([])))


def test_average_gain_returns_nan_for_all_nan_input():
    rets = np.array([np.nan, np.nan, np.nan])
    assert np.isnan(stats.average_gain(rets))


@pytest.mark.parametrize(
    "method, expected_first_column",
    [("arith", 0.15), ("geo", np.sqrt(1.1 * 1.2) - 1)],
)
def test_average_gain_2d_reduces_each_column(method, expected_first_column):
    # column 0: gains 0.10, 0.20; column 1: no gains -> NaN
    rets = np.array([[0.10, -0.10], [0.20, 0.0], [-0.10, -0.20]])
    obtained = stats.average_gain(rets, method=method)
    expected = np.array([expected_first_column, np.nan])
    np.testing.assert_almost_equal(obtained, expected)


def test_average_gain_series_returns_float_and_keeps_input_name():
    # gains 0.10, 0.20 -> 0.15; a Series reduces to a float and the input
    # keeps its name and values
    rets = pd.Series([0.10, -0.05, 0.20], name="asset")
    obtained = stats.average_gain(rets)
    np.testing.assert_almost_equal(obtained, 0.15)
    assert isinstance(obtained, float)
    pd.testing.assert_series_equal(
        rets, pd.Series([0.10, -0.05, 0.20], name="asset")
    )


def test_average_gain_dataframe_returns_series_indexed_by_columns():
    # a: gains 0.10, 0.20 -> 0.15; b: no gains -> NaN
    rets = pd.DataFrame({"a": [0.10, 0.20, -0.10], "b": [-0.10, 0.0, -0.20]})
    obtained = stats.average_gain(rets)
    expected = pd.Series([0.15, np.nan], index=["a", "b"])
    pd.testing.assert_series_equal(obtained, expected)


# ---------------------------------------------------------------------------
# average_loss


def test_average_loss_defaults_to_arithmetic_mean_of_negative_returns():
    # losses -0.05, -0.15 -> arithmetic mean -0.10
    rets = np.array([0.10, -0.05, 0.20, -0.15])
    obtained = stats.average_loss(rets)
    np.testing.assert_almost_equal(obtained, -0.10)


def test_average_loss_geo_is_geometric_mean_of_negative_returns():
    # losses -0.10, -0.20 -> sqrt(0.9 * 0.8) - 1 = -0.151471862...
    rets = np.array([-0.10, 0.05, -0.20, 0.10])
    obtained = stats.average_loss(rets, method="geo")
    np.testing.assert_almost_equal(obtained, np.sqrt(0.9 * 0.8) - 1)


def test_average_loss_geo_of_a_total_loss_is_minus_one():
    # losses -1.0, -0.5 -> prod(1 + r) = 0 -> 0 ** (1 / 2) - 1 = -1
    rets = np.array([-1.0, 0.10, -0.50])
    obtained = stats.average_loss(rets, method="geo")
    np.testing.assert_almost_equal(obtained, -1.0)


def test_average_loss_rejects_unknown_method():
    rets = np.array([0.10, -0.05, 0.20])
    with pytest.raises(ValueError):
        stats.average_loss(rets, method="median")


def test_average_loss_ignores_zero_returns():
    # zeros are not losses: losses -0.10, -0.20 -> -0.15, not -0.30 / 4
    rets = np.array([0.0, -0.10, 0.0, -0.20])
    obtained = stats.average_loss(rets)
    np.testing.assert_almost_equal(obtained, -0.15)


def test_average_loss_ignores_nan_at_start_and_in_the_middle():
    # losses -0.10, -0.20 -> -0.15; NaN neither counts nor propagates
    rets = np.array([np.nan, -0.10, np.nan, -0.20, 0.10])
    obtained = stats.average_loss(rets)
    np.testing.assert_almost_equal(obtained, -0.15)


def test_average_loss_returns_nan_when_there_are_no_losses():
    rets = np.array([0.10, 0.0, 0.20])
    assert np.isnan(stats.average_loss(rets, method="arith"))
    assert np.isnan(stats.average_loss(rets, method="geo"))


def test_average_loss_returns_nan_for_empty_input():
    assert np.isnan(stats.average_loss(np.array([])))


def test_average_loss_returns_nan_for_all_nan_input():
    rets = np.array([np.nan, np.nan, np.nan])
    assert np.isnan(stats.average_loss(rets))


@pytest.mark.parametrize(
    "method, expected_first_column",
    [("arith", -0.15), ("geo", np.sqrt(0.9 * 0.8) - 1)],
)
def test_average_loss_2d_reduces_each_column(method, expected_first_column):
    # column 0: losses -0.10, -0.20; column 1: no losses -> NaN
    rets = np.array([[-0.10, 0.10], [-0.20, 0.0], [0.10, 0.20]])
    obtained = stats.average_loss(rets, method=method)
    expected = np.array([expected_first_column, np.nan])
    np.testing.assert_almost_equal(obtained, expected)


def test_average_loss_series_returns_float_and_keeps_input_name():
    # losses -0.10, -0.20 -> -0.15; a Series reduces to a float and the
    # input keeps its name and values
    rets = pd.Series([-0.10, 0.05, -0.20], name="asset")
    obtained = stats.average_loss(rets)
    np.testing.assert_almost_equal(obtained, -0.15)
    assert isinstance(obtained, float)
    pd.testing.assert_series_equal(
        rets, pd.Series([-0.10, 0.05, -0.20], name="asset")
    )


def test_average_loss_dataframe_returns_series_indexed_by_columns():
    # a: losses -0.10, -0.20 -> -0.15; b: no losses -> NaN
    rets = pd.DataFrame({"a": [-0.10, -0.20, 0.10], "b": [0.10, 0.0, 0.20]})
    obtained = stats.average_loss(rets)
    expected = pd.Series([-0.15, np.nan], index=["a", "b"])
    pd.testing.assert_series_equal(obtained, expected)


# ---------------------------------------------------------------------------
# gain_loss_ratio


def test_gain_loss_ratio_hand_example():
    # gains 0.1, 0.2 -> average 0.15; loss -0.1 -> average -0.1
    # abs(0.15 / -0.1) * (2 gains / 1 loss) = 1.5 * 2 = 3.0
    rets = np.array([0.1, 0.2, -0.1])
    obtained = stats.gain_loss_ratio(rets)
    np.testing.assert_almost_equal(obtained, 3.0)


def test_gain_loss_ratio_ignores_zero_returns():
    # zeros are neither gains nor losses, so the ratio is still 3.0
    rets = np.array([0.1, 0.0, 0.2, -0.1, 0.0])
    obtained = stats.gain_loss_ratio(rets)
    np.testing.assert_almost_equal(obtained, 3.0)


def test_gain_loss_ratio_ignores_nan_at_start_and_in_the_middle():
    # same gains and loss as the hand example -> 3.0
    rets = np.array([np.nan, 0.1, 0.2, np.nan, -0.1])
    obtained = stats.gain_loss_ratio(rets)
    np.testing.assert_almost_equal(obtained, 3.0)


def test_gain_loss_ratio_returns_nan_when_there_are_no_losses():
    rets = np.array([0.1, 0.0, 0.2])
    assert np.isnan(stats.gain_loss_ratio(rets))


def test_gain_loss_ratio_returns_zero_when_there_are_no_gains():
    rets = np.array([-0.1, 0.0, -0.2])
    obtained = stats.gain_loss_ratio(rets)
    np.testing.assert_almost_equal(obtained, 0.0)


def test_gain_loss_ratio_returns_nan_without_gains_nor_losses():
    # only zeros: there is no loss to divide by, so NaN rather than 0
    rets = np.array([0.0, 0.0, 0.0])
    assert np.isnan(stats.gain_loss_ratio(rets))


def test_gain_loss_ratio_returns_nan_for_empty_input():
    assert np.isnan(stats.gain_loss_ratio(np.array([])))


def test_gain_loss_ratio_returns_nan_for_all_nan_input():
    rets = np.array([np.nan, np.nan, np.nan])
    assert np.isnan(stats.gain_loss_ratio(rets))


def test_gain_loss_ratio_2d_reduces_each_column():
    # column 0: 0.1, 0.2, -0.1 -> 3.0 (hand example)
    # column 1: gain 0.3, losses -0.1, -0.2 -> abs(0.3 / -0.15) * (1 / 2) = 1.0
    rets = np.array([[0.1, -0.1], [0.2, 0.3], [-0.1, -0.2]])
    obtained = stats.gain_loss_ratio(rets)
    np.testing.assert_almost_equal(obtained, np.array([3.0, 1.0]))


def test_gain_loss_ratio_series_returns_float_and_keeps_input_name():
    rets = pd.Series([0.1, 0.2, -0.1], name="asset")
    obtained = stats.gain_loss_ratio(rets)
    np.testing.assert_almost_equal(obtained, 3.0)
    assert isinstance(obtained, float)
    pd.testing.assert_series_equal(
        rets, pd.Series([0.1, 0.2, -0.1], name="asset")
    )


def test_gain_loss_ratio_dataframe_returns_series_indexed_by_columns():
    # a: hand example -> 3.0; b: no losses -> NaN
    rets = pd.DataFrame({"a": [0.1, 0.2, -0.1], "b": [0.1, 0.0, 0.2]})
    obtained = stats.gain_loss_ratio(rets)
    expected = pd.Series([3.0, np.nan], index=["a", "b"])
    pd.testing.assert_series_equal(obtained, expected)


# ---------------------------------------------------------------------------
# up_period_percent / down_period_percent


def test_up_period_percent_counts_zero_as_an_up_period():
    # at or above 0: 0.1, 0.0, 0.2 -> 3 of 4 periods -> 0.75 (parts per unit)
    rets = np.array([0.1, 0.0, -0.1, 0.2])
    obtained = stats.up_period_percent(rets)
    np.testing.assert_almost_equal(obtained, 0.75)


def test_down_period_percent_counts_only_negative_returns():
    # below 0: -0.1 -> 1 of 4 periods -> 0.25 (parts per unit)
    rets = np.array([0.1, 0.0, -0.1, 0.2])
    obtained = stats.down_period_percent(rets)
    np.testing.assert_almost_equal(obtained, 0.25)


def test_up_period_percent_ignores_nan_at_start_and_in_the_middle():
    # valid: 0.1, -0.1, 0.0 -> at or above 0: 0.1, 0.0 -> 2 of 3
    rets = np.array([np.nan, 0.1, np.nan, -0.1, 0.0])
    obtained = stats.up_period_percent(rets)
    np.testing.assert_almost_equal(obtained, 2 / 3)


def test_down_period_percent_ignores_nan_at_start_and_in_the_middle():
    # valid: 0.1, -0.1, 0.0 -> below 0: -0.1 -> 1 of 3
    rets = np.array([np.nan, 0.1, np.nan, -0.1, 0.0])
    obtained = stats.down_period_percent(rets)
    np.testing.assert_almost_equal(obtained, 1 / 3)


def test_up_period_percent_is_one_when_no_return_is_negative():
    rets = np.array([0.0, 0.1, 0.2])
    np.testing.assert_almost_equal(stats.up_period_percent(rets), 1.0)
    np.testing.assert_almost_equal(stats.down_period_percent(rets), 0.0)


def test_down_period_percent_is_one_when_every_return_is_negative():
    rets = np.array([-0.1, -0.2, -0.3])
    np.testing.assert_almost_equal(stats.down_period_percent(rets), 1.0)
    np.testing.assert_almost_equal(stats.up_period_percent(rets), 0.0)


def test_up_and_down_period_percent_sum_to_one():
    # NaN is excluded from both, zero counted as up: 2 / 4 + 2 / 4 = 1
    rets = np.array([np.nan, 0.1, 0.0, -0.1, -0.2])
    total = stats.up_period_percent(rets) + stats.down_period_percent(rets)
    np.testing.assert_almost_equal(total, 1.0)


def test_up_and_down_period_percent_sum_to_one_per_column():
    # a: 2 / 3 + 1 / 3; b: 2 / 4 + 2 / 4
    rets = pd.DataFrame(
        {"a": [0.1, 0.0, -0.1, np.nan], "b": [-0.1, -0.2, 0.3, 0.0]}
    )
    total = stats.up_period_percent(rets) + stats.down_period_percent(rets)
    expected = pd.Series([1.0, 1.0], index=["a", "b"])
    pd.testing.assert_series_equal(total, expected)


def test_up_period_percent_returns_nan_for_empty_input():
    assert np.isnan(stats.up_period_percent(np.array([])))


def test_down_period_percent_returns_nan_for_empty_input():
    assert np.isnan(stats.down_period_percent(np.array([])))


def test_up_period_percent_returns_nan_for_all_nan_input():
    rets = np.array([np.nan, np.nan, np.nan])
    assert np.isnan(stats.up_period_percent(rets))


def test_down_period_percent_returns_nan_for_all_nan_input():
    rets = np.array([np.nan, np.nan, np.nan])
    assert np.isnan(stats.down_period_percent(rets))


def test_up_period_percent_2d_reduces_each_column():
    # column 0: 0.1, 0.0, -0.1, 0.2 -> 3 of 4 -> 0.75
    # column 1: -0.1, -0.2, 0.3, NaN -> 1 of 3 valid -> 1 / 3
    rets = np.array([[0.1, -0.1], [0.0, -0.2], [-0.1, 0.3], [0.2, np.nan]])
    obtained = stats.up_period_percent(rets)
    np.testing.assert_almost_equal(obtained, np.array([0.75, 1 / 3]))


def test_down_period_percent_2d_reduces_each_column():
    # column 0: 0.1, 0.0, -0.1, 0.2 -> 1 of 4 -> 0.25
    # column 1: -0.1, -0.2, 0.3, NaN -> 2 of 3 valid -> 2 / 3
    rets = np.array([[0.1, -0.1], [0.0, -0.2], [-0.1, 0.3], [0.2, np.nan]])
    obtained = stats.down_period_percent(rets)
    np.testing.assert_almost_equal(obtained, np.array([0.25, 2 / 3]))


def test_up_period_percent_series_returns_float_and_keeps_input_name():
    rets = pd.Series([0.1, 0.0, -0.1, 0.2], name="asset")
    obtained = stats.up_period_percent(rets)
    np.testing.assert_almost_equal(obtained, 0.75)
    assert isinstance(obtained, float)
    pd.testing.assert_series_equal(
        rets, pd.Series([0.1, 0.0, -0.1, 0.2], name="asset")
    )


def test_down_period_percent_series_returns_float_and_keeps_input_name():
    rets = pd.Series([0.1, 0.0, -0.1, 0.2], name="asset")
    obtained = stats.down_period_percent(rets)
    np.testing.assert_almost_equal(obtained, 0.25)
    assert isinstance(obtained, float)
    pd.testing.assert_series_equal(
        rets, pd.Series([0.1, 0.0, -0.1, 0.2], name="asset")
    )


def test_up_period_percent_dataframe_returns_series_indexed_by_columns():
    # a: 3 of 4 -> 0.75; b: 1 of 3 valid -> 1 / 3
    rets = pd.DataFrame(
        {"a": [0.1, 0.0, -0.1, 0.2], "b": [-0.1, -0.2, 0.3, np.nan]}
    )
    obtained = stats.up_period_percent(rets)
    expected = pd.Series([0.75, 1 / 3], index=["a", "b"])
    pd.testing.assert_series_equal(obtained, expected)


def test_down_period_percent_dataframe_returns_series_indexed_by_columns():
    # a: 1 of 4 -> 0.25; b: 2 of 3 valid -> 2 / 3
    rets = pd.DataFrame(
        {"a": [0.1, 0.0, -0.1, 0.2], "b": [-0.1, -0.2, 0.3, np.nan]}
    )
    obtained = stats.down_period_percent(rets)
    expected = pd.Series([0.25, 2 / 3], index=["a", "b"])
    pd.testing.assert_series_equal(obtained, expected)
