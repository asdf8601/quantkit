"""Tests for portfolio calculation helpers."""

import numpy as np
import pandas as pd
import pytest

from quantkit.portfolio._utils import (
    asset_array,
    pair_assets,
    portfolio_amount,
    positive_divide,
    wrap_assets,
    wrap_reduction,
)


def test_asset_array_rejects_invalid_asset_data():
    """Asset data must be nonempty, finite, and real numeric."""
    for values in (
        np.array([], dtype=float),
        np.array([True, False]),
        np.array(["a", "b"]),
        np.array([1 + 2j]),
        np.array([np.inf]),
    ):
        with pytest.raises((TypeError, ValueError)):
            asset_array(values)


def test_asset_array_rejects_duplicate_pandas_labels():
    """Pandas asset labels must be unique."""
    values = pd.DataFrame([[1.0, 2.0]], columns=["a", "a"])

    with pytest.raises(ValueError, match="duplicate"):
        asset_array(values)


def test_asset_array_converts_integer_inputs_to_float():
    """Asset arithmetic receives floating values to avoid integer overflow."""
    result = asset_array(np.array([1, 2], dtype=np.int64))

    assert result.dtype == float


def test_pair_assets_aligns_pandas_labels_and_broadcasts_assets():
    """Paired pandas inputs align labels before vector broadcasting."""
    reference = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["t1", "t2"],
        columns=["a", "b"],
    )
    other = pd.Series([20.0, 10.0], index=["b", "a"])

    arr_reference, arr_other = pair_assets(reference, other)

    np.testing.assert_array_equal(arr_reference, reference.to_numpy())
    np.testing.assert_array_equal(arr_other, [[10.0, 20.0], [10.0, 20.0]])


def test_pair_assets_aligns_dataframe_time_and_asset_labels():
    """DataFrames align both axes using exact label sets."""
    reference = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["t1", "t2"],
        columns=["a", "b"],
    )
    other = pd.DataFrame(
        [[40.0, 30.0], [20.0, 10.0]],
        index=["t2", "t1"],
        columns=["b", "a"],
    )

    _, aligned = pair_assets(reference, other)

    np.testing.assert_array_equal(aligned, [[10.0, 20.0], [30.0, 40.0]])


def test_pair_assets_rejects_incompatible_shapes_and_labels():
    """Asset pairs cannot broadcast scalars or silently join labels."""
    reference = pd.DataFrame([[1.0, 2.0]], columns=["a", "b"])

    with pytest.raises(ValueError, match="labels"):
        pair_assets(reference, pd.Series([1.0, 2.0], index=["a", "c"]))
    with pytest.raises(ValueError, match="asset count"):
        pair_assets(np.ones((2, 2)), np.ones(3))


def test_wrappers_preserve_pandas_metadata_and_reduction_shape():
    """Asset and reduction wrappers retain their defined pandas metadata."""
    frame = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["t1", "t2"],
        columns=["a", "b"],
    )

    wrapped = wrap_assets(frame, [[10.0, 20.0], [30.0, 40.0]])
    reduced = wrap_reduction(frame, [30.0, 70.0])

    pd.testing.assert_frame_equal(
        wrapped,
        pd.DataFrame(
            [[10.0, 20.0], [30.0, 40.0]],
            index=frame.index,
            columns=frame.columns,
        ),
    )
    pd.testing.assert_series_equal(
        reduced, pd.Series([30.0, 70.0], index=frame.index)
    )
    assert wrap_reduction(np.array([1.0, 2.0]), np.array(3.0)) == 3.0


def test_portfolio_amount_aligns_time_series_and_validates_shape():
    """Amounts align to DataFrame time labels and remain positional otherwise."""
    frame = pd.DataFrame(
        [[1.0], [2.0]], index=["t1", "t2"], columns=["asset"]
    )
    amount = pd.Series([200.0, 100.0], index=["t2", "t1"])

    aligned = portfolio_amount(frame, amount, "nav")

    np.testing.assert_array_equal(aligned, [100.0, 200.0])
    positional = portfolio_amount(
        np.ones((2, 1)), pd.Series([100.0, 200.0], index=["x", "y"]), "nav"
    )
    np.testing.assert_array_equal(positional, [100.0, 200.0])
    with pytest.raises(ValueError, match="time axis"):
        portfolio_amount(np.ones((2, 1)), np.ones(3), "nav")
    with pytest.raises(ValueError, match="scalar"):
        portfolio_amount(np.ones(2), np.ones(2), "nav")


def test_positive_divide_marks_nonpositive_and_missing_denominators_nan():
    """Positive division suppresses zero, negative, and missing divisors."""
    result = positive_divide([4.0, 4.0, 4.0, 4.0], [2.0, 0.0, -1.0, np.nan])

    np.testing.assert_allclose(result, [2.0, np.nan, np.nan, np.nan], equal_nan=True)
