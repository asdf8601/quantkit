"""Shared rolling and expanding drawdown path specifications."""

import numpy as np
import pandas as pd
import pytest

from quantkit._window_drawdown import drawdown_path


def test_expanding_drawdown_and_drawup_are_extreme_counterparts():
    prices = np.array([4.0, 2.0, 3.0, 6.0])

    drawdown = drawdown_path(prices, relative=False)
    drawup = drawdown_path(prices, relative=False, upward=True)

    np.testing.assert_equal(drawdown, [0.0, -2.0, -1.0, 0.0])
    np.testing.assert_equal(drawup, [0.0, 0.0, 1.0, 4.0])


def test_integer_window_forgets_an_exited_peak():
    prices = np.array([100.0, 50.0, 60.0, 70.0])

    obtained = drawdown_path(prices, window=3, relative=True)

    np.testing.assert_allclose(obtained, [np.nan, np.nan, -0.4, 0.0])


def test_integer_window_defaults_min_periods_to_its_size():
    prices = np.array([3.0, 2.0, 1.0])

    default = drawdown_path(prices, window=2, relative=False)
    one_period = drawdown_path(prices, window=2, min_periods=1, relative=False)

    np.testing.assert_equal(default, [np.nan, -1.0, -1.0])
    np.testing.assert_equal(one_period, [0.0, -1.0, -1.0])


def test_maximum_drawdown_requires_peak_and_valley_inside_window():
    prices = np.array([100.0, 50.0, 60.0, 70.0, 80.0])

    rolling = drawdown_path(prices, window=3, maximum=True)
    expanding = drawdown_path(prices, maximum=True)

    np.testing.assert_allclose(rolling, [np.nan, np.nan, -0.5, 0.0, 0.0])
    np.testing.assert_allclose(expanding, [0.0, -0.5, -0.5, -0.5, -0.5])


def test_maximum_drawup_is_the_largest_complete_rise():
    prices = np.array([4.0, 2.0, 3.0, 6.0, 5.0])

    expanding = drawdown_path(prices, upward=True, maximum=True)
    rolling = drawdown_path(prices, window=3, upward=True, maximum=True)

    np.testing.assert_allclose(expanding, [0.0, 0.0, 0.5, 2.0, 2.0])
    np.testing.assert_allclose(rolling, [np.nan, np.nan, 0.5, 2.0, 1.0])


def test_nan_is_missing_at_the_point_but_does_not_reset_extremum():
    prices = np.array([100.0, np.nan, 50.0])

    obtained = drawdown_path(prices, min_periods=2)

    np.testing.assert_allclose(obtained, [np.nan, np.nan, -0.5])


def test_maximum_path_can_report_worst_episode_at_missing_current_price():
    prices = np.array([100.0, 50.0, np.nan, 60.0])

    point = drawdown_path(prices, window=3, min_periods=2)
    maximum = drawdown_path(prices, window=3, min_periods=2, maximum=True)

    np.testing.assert_allclose(point, [np.nan, -0.5, np.nan, 0.0])
    np.testing.assert_allclose(maximum, [np.nan, -0.5, -0.5, 0.0])


def test_columns_have_independent_extrema_and_valid_counts():
    prices = np.array(
        [
            [4.0, np.nan],
            [2.0, 5.0],
            [3.0, 4.0],
            [6.0, 3.0],
        ]
    )

    obtained = drawdown_path(prices, window=3, min_periods=2, relative=False)

    expected = np.array(
        [
            [np.nan, np.nan],
            [-2.0, np.nan],
            [-1.0, -1.0],
            [0.0, -2.0],
        ]
    )
    np.testing.assert_equal(obtained, expected)


def test_offset_window_uses_open_left_boundary_and_minimum_one():
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"])
    prices = pd.Series([100.0, 50.0, 40.0], index=index, name="price")

    obtained = drawdown_path(prices, window="2D", relative=False)

    expected = pd.Series([0.0, -50.0, 0.0], index=index, name="price")
    pd.testing.assert_series_equal(obtained, expected)


def test_relative_and_absolute_paths_use_the_same_extremum():
    prices = np.array([8.0, 4.0, 6.0])

    relative = drawdown_path(prices)
    absolute = drawdown_path(prices, relative=False)

    np.testing.assert_allclose(relative, [0.0, -0.5, -0.25])
    np.testing.assert_allclose(absolute, [0.0, -4.0, -2.0])


def test_relative_path_subtracts_before_dividing_like_scalar_statistic():
    prices = np.array([1.1, 0.65])

    obtained = drawdown_path(prices)

    assert obtained[-1] == (0.65 - 1.1) / 1.1


def test_relative_path_preserves_scalar_semantics_for_negative_prices():
    prices = np.array([-1.0, -2.0, -1.0])

    point = drawdown_path(prices)
    maximum = drawdown_path(prices, maximum=True)

    np.testing.assert_equal(point, [0.0, 1.0, 0.0])
    np.testing.assert_equal(maximum, [0.0, 0.0, 0.0])


@pytest.mark.parametrize("maximum", [False, True])
def test_relative_zero_extremum_is_nan_never_infinite(maximum):
    prices = np.array([0.0, 1.0, 2.0])

    obtained = drawdown_path(
        prices, upward=True, relative=True, maximum=maximum
    )

    assert np.isnan(obtained).all()
    assert not np.isinf(obtained).any()


def test_series_and_dataframe_metadata_are_preserved():
    index = pd.date_range("2024-01-01", periods=3)
    series = pd.Series([3, 2, 4], index=index, name="price")
    frame = pd.DataFrame(
        {"asset-a": [3, 2, 4], "asset-b": [1, 3, 2]}, index=index
    )

    series_result = drawdown_path(series, relative=False)
    frame_result = drawdown_path(frame, relative=False)

    pd.testing.assert_series_equal(
        series_result,
        pd.Series([0.0, -1.0, 0.0], index=index, name="price"),
    )
    pd.testing.assert_frame_equal(
        frame_result,
        pd.DataFrame(
            {"asset-a": [0.0, -1.0, 0.0], "asset-b": [0.0, 0.0, -1.0]},
            index=index,
        ),
    )


@pytest.mark.parametrize(
    "prices, expected_shape",
    [
        (np.array([]), (0,)),
        (np.empty((0, 2)), (0, 2)),
        (np.array([np.nan, np.nan]), (2,)),
    ],
)
def test_empty_and_all_nan_inputs_keep_shape(prices, expected_shape):
    obtained = drawdown_path(prices, maximum=True)

    assert obtained.shape == expected_shape
    assert np.isnan(obtained).all()


def test_out_is_filled_and_result_keeps_the_input_container():
    prices = pd.Series([4.0, 2.0, 3.0], name="price")
    out = np.empty(3)

    obtained = drawdown_path(prices, relative=False, out=out)

    np.testing.assert_equal(out, [0.0, -2.0, -1.0])
    pd.testing.assert_series_equal(
        obtained, pd.Series([0.0, -2.0, -1.0], name="price")
    )


def test_numpy_result_is_the_provided_out_buffer():
    prices = np.array([4.0, 2.0, 3.0])
    out = np.empty(3)

    obtained = drawdown_path(prices, relative=False, out=out)

    assert obtained is out
    np.testing.assert_equal(out, [0.0, -2.0, -1.0])


@pytest.mark.parametrize(
    "out, error",
    [
        (np.empty(2), ValueError),
        (np.empty(3, dtype=int), TypeError),
    ],
)
def test_out_rejects_wrong_shape_and_non_float_dtype(out, error):
    with pytest.raises(error):
        drawdown_path(np.array([3.0, 2.0, 1.0]), out=out)


@pytest.mark.parametrize("parameter", ["relative", "upward", "maximum"])
def test_boolean_options_reject_non_boolean_values(parameter):
    with pytest.raises(TypeError, match=f"{parameter} must be a boolean"):
        drawdown_path(np.array([3.0, 2.0]), **{parameter: 1})
