"""Tests for the shared rolling and expanding reduction engine."""

import numpy as np
import pandas as pd
import pytest

from quantkit._windows import evaluate, prepare


def test_prepare_builds_integer_and_expanding_bounds():
    values = np.arange(5.0)

    frame, starts, minimum = prepare(values, window=3)
    _, expanding_starts, expanding_minimum = prepare(values)

    assert frame.shape == (5, 1)
    np.testing.assert_array_equal(starts, [0, 0, 0, 1, 2])
    np.testing.assert_array_equal(expanding_starts, [0, 0, 0, 0, 0])
    assert minimum == 3
    assert expanding_minimum == 1


def test_integer_window_uses_valid_observations_for_min_periods():
    values = np.array([1.0, np.nan, 3.0, 5.0])

    result = evaluate(
        values,
        "volatility",
        window=3,
        min_periods=2,
        parameters={"ddof": 0},
    )

    np.testing.assert_allclose(result, [np.nan, np.nan, 1.0, 1.0], equal_nan=True)


def test_default_integer_min_periods_is_the_window_size():
    result = evaluate(
        np.array([1.0, 2.0, 3.0, 4.0]),
        "average_gain",
        window=3,
    )

    np.testing.assert_allclose(result, [np.nan, np.nan, 2.0, 3.0], equal_nan=True)


def test_explicit_zero_min_periods_still_requires_an_observation():
    values = np.array([np.nan, 2.0])

    result = evaluate(
        values, "average_gain", window=None, min_periods=0
    )

    np.testing.assert_allclose(result, [np.nan, 2.0], equal_nan=True)


def test_time_window_is_right_closed_and_never_uses_future_rows():
    index = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"]
    )
    values = pd.Series([1.0, 2.0, 100.0, 4.0], index=index, name="returns")

    result = evaluate(values, "average_gain", window="2D")

    expected = pd.Series([1.0, 1.5, 100.0, 52.0], index=index, name="returns")
    pd.testing.assert_series_equal(result, expected)


def test_benchmark_series_reindexes_without_removing_primary_rows():
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    returns = pd.Series([0.0, 1.0, 2.0, 3.0], index=index, name="fund")
    benchmark = pd.Series([0.0, 1.0, 2.0], index=index[[0, 1, 3]])

    result = evaluate(
        returns,
        "beta",
        window=None,
        min_periods=2,
        benchmark=benchmark,
    )

    expected = pd.Series([np.nan, 1.0, 1.0, 1.5], index=index, name="fund")
    pd.testing.assert_series_equal(result, expected)


def test_duplicate_primary_labels_do_not_expand_paired_windows():
    returns = pd.Series([1.0, 2.0, 3.0], index=["a", "a", "b"])
    benchmark = pd.Series([1.0, 2.0], index=["a", "b"])

    result = evaluate(
        returns, "beta", min_periods=2, benchmark=benchmark
    )

    expected = pd.Series([np.nan, np.nan, 1.5], index=returns.index)
    pd.testing.assert_series_equal(result, expected)


def test_numeric_tail_reducer_accepts_duplicate_count_window_labels():
    returns = pd.Series([0.1, -0.2, 0.05], index=["a", "a", "b"])

    result = evaluate(
        returns,
        "expected_shortfall",
        min_periods=1,
        parameters={"confidence": 0.5},
    )

    expected = evaluate(
        returns.to_numpy(),
        "expected_shortfall",
        min_periods=1,
        parameters={"confidence": 0.5},
    )
    np.testing.assert_allclose(result.to_numpy(), expected)


def test_dynamic_risk_free_counts_complete_pairs_and_keeps_gaps():
    returns = np.array([0.0, 2.0, 4.0])
    risk_free = np.array([0.0, np.nan, 0.0])

    result = evaluate(
        returns,
        "sharpe_ratio",
        min_periods=2,
        parameters={"risk_free": risk_free, "factor": 1.0},
    )

    np.testing.assert_allclose(result, [np.nan, np.nan, 1.0], equal_nan=True)


def test_label_reduction_uses_original_numpy_positions():
    prices = np.array([10.0, 11.0, 8.0, 12.0, 6.0])

    result = evaluate(
        prices,
        "max_drawdown_valley",
        window=3,
        min_periods=1,
        label=True,
    )

    np.testing.assert_allclose(
        result, [np.nan, np.nan, 2.0, 2.0, 4.0], equal_nan=True
    )


def test_label_reduction_preserves_datetime_labels_and_nat():
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.Series([10.0, 8.0, 10.0, 12.0], index=index, name="price")

    result = evaluate(
        prices,
        "max_drawdown_recovery",
        min_periods=1,
        label=True,
    )

    expected = pd.Series(
        [pd.NaT, pd.NaT, index[2], index[2]], index=index, name="price"
    )
    pd.testing.assert_series_equal(result, expected)


def test_label_reduction_preserves_timezone_and_string_label_dtype():
    zoned_index = pd.date_range("2024-01-01", periods=3, tz="UTC")
    zoned = pd.Series([3.0, 2.0, 3.0], index=zoned_index, name="price")
    strings = pd.Series([3.0, 2.0, 3.0], index=["a", "b", "c"])

    zoned_result = evaluate(zoned, "max_drawdown_recovery", label=True)
    string_result = evaluate(strings, "max_drawdown_recovery", label=True)

    assert zoned_result.dtype == zoned_index.dtype
    assert string_result.dtype == object
    assert zoned_result.iloc[-1] == zoned_index[-1]
    assert string_result.iloc[-1] == "c"


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([2**60, 2**60 + 1, 2**60 + 2]),
        pd.period_range("2024-01", periods=3, freq="M"),
        pd.CategoricalIndex(["a", "b", "c"]),
        pd.MultiIndex.from_tuples([("a", 1), ("b", 2), ("c", 3)]),
    ],
)
def test_label_reduction_preserves_non_time_index_values_exactly(index):
    prices = pd.Series([3.0, 2.0, 3.0], index=index)

    result = evaluate(prices, "max_drawdown_recovery", label=True)

    assert result.dtype == object
    assert result.iloc[-1] == index[-1]


def test_empty_datetime_input_accepts_a_time_window():
    data = pd.Series([], index=pd.DatetimeIndex([]), dtype=float, name="x")

    result = evaluate(data, "average_gain", window="1D")

    pd.testing.assert_series_equal(result, data)


def test_dataframe_result_preserves_axes():
    index = pd.Index(["a", "b", "c"], name="period")
    columns = pd.Index(["fund", "peer"], name="asset")
    data = pd.DataFrame([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0]], index, columns)

    result = evaluate(data, "average_gain")

    expected = pd.DataFrame(
        [[1.0, 4.0], [1.5, 3.5], [2.0, 3.0]], index, columns
    )
    pd.testing.assert_frame_equal(result, expected)


def test_terminal_masks_only_the_missing_current_column():
    data = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]})

    result = evaluate(data, "average_gain", terminal=True)

    expected = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 2.5]})
    pd.testing.assert_frame_equal(result, expected)


def test_empty_inputs_keep_their_container_and_shape():
    series = pd.Series([], dtype=float, name="returns")
    frame = pd.DataFrame(columns=["a", "b"], dtype=float)

    series_result = evaluate(series, "average_gain")
    frame_result = evaluate(frame, "average_gain")
    array_result = evaluate(np.empty((0, 2)), "average_gain")

    pd.testing.assert_series_equal(series_result, series)
    pd.testing.assert_frame_equal(frame_result, frame)
    assert array_result.shape == (0, 2)


@pytest.mark.parametrize("window", [0, -1, "ME", "0D"])
def test_invalid_windows_are_rejected(window):
    data = pd.Series([1.0], index=pd.date_range("2024-01-01", periods=1))

    with pytest.raises((TypeError, ValueError)):
        prepare(data, window=window)


def test_time_windows_require_a_monotonic_unique_time_index():
    duplicated = pd.Series(
        [1.0, 2.0], index=pd.to_datetime(["2024-01-01", "2024-01-01"])
    )

    with pytest.raises(ValueError, match="monotonic and unique"):
        prepare(duplicated, window="1D")


@pytest.mark.parametrize(
    "data, message",
    [
        (np.array([1.0, np.inf]), "infinity"),
        (np.array(["1", "2"]), "numeric"),
        (np.ones((1, 1, 1)), "one- or two-dimensional"),
    ],
)
def test_invalid_data_is_rejected(data, message):
    with pytest.raises((TypeError, ValueError), match=message):
        prepare(data)


def test_parameters_are_validated_before_short_windows_are_skipped():
    with pytest.raises(ValueError, match="confidence"):
        evaluate(
            np.array([np.nan]),
            "value_at_risk",
            window=10,
            parameters={"confidence": 1.0},
        )


def test_imported_non_reducer_is_not_a_window_statistic():
    with pytest.raises(ValueError, match="unknown window statistic"):
        evaluate(np.array([1.0]), "cum_returns")


def test_cdar_validates_a_window_baseline_before_min_periods_gate():
    wealth = np.array([np.nan, 0.0, np.nan])

    with pytest.raises(ValueError, match="initial wealth must be positive"):
        evaluate(
            wealth,
            "conditional_drawdown_at_risk",
            window=2,
            min_periods=2,
        )


def test_benchmark_numpy_input_must_match_primary_length():
    with pytest.raises(ValueError, match="same length"):
        evaluate(
            np.array([1.0, 2.0]),
            "beta",
            benchmark=np.array([1.0]),
        )


def test_complex_companion_series_is_rejected_before_conversion():
    returns = pd.Series([1.0, 2.0])
    benchmark = pd.Series([1.0 + 2.0j, 2.0 + 3.0j])

    with pytest.raises(TypeError, match="real numeric"):
        evaluate(returns, "beta", benchmark=benchmark)
