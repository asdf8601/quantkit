"""Tests for weighted portfolio and expected returns."""

import numpy as np
import pandas as pd
import pytest

from quantkit.portfolio import returns


def test_portfolio_return_uses_beginning_weights():
    """The contract's two-asset example aggregates to four percent."""
    result = returns.portfolio_returns(
        np.array([0.1, -0.05]), np.array([0.6, 0.4])
    )

    assert result == pytest.approx(0.04)


def test_portfolio_returns_use_each_row_of_time_varying_beginning_weights():
    """Time-varying weights are used directly without a look-ahead shift."""
    asset_returns = np.array([[0.1, 0.0], [0.0, 0.1]])
    beginning_weights = np.array([[0.6, 0.4], [0.4, 0.6]])
    original_returns = asset_returns.copy()
    original_weights = beginning_weights.copy()

    result = returns.portfolio_returns(asset_returns, beginning_weights)

    np.testing.assert_allclose(result, [0.06, 0.06])
    np.testing.assert_array_equal(asset_returns, original_returns)
    np.testing.assert_array_equal(beginning_weights, original_weights)


def test_returns_align_pandas_assets_and_preserve_first_argument_metadata():
    """Pandas weight labels reorder while returns determine the result axes."""
    asset_returns = pd.DataFrame(
        [[0.1, -0.05], [0.2, 0.0]],
        index=["t1", "t2"],
        columns=["stock", "hedge"],
    )
    beginning_weights = pd.DataFrame(
        [[0.4, 0.6], [0.25, 0.75]],
        index=["t2", "t1"],
        columns=["hedge", "stock"],
    )

    result = returns.portfolio_returns(asset_returns, beginning_weights)

    pd.testing.assert_series_equal(
        result, pd.Series([0.0625, 0.12], index=asset_returns.index)
    )


def test_expected_return_aligns_series_labels():
    """Expected return treats Series indices as asset labels."""
    expected = pd.Series([0.1, -0.05], index=["stock", "hedge"])
    weights = pd.Series([0.4, 0.6], index=["hedge", "stock"])

    assert returns.expected_return(expected, weights) == pytest.approx(0.04)


def test_mixed_inputs_match_assets_positionally():
    """A NumPy weight vector follows DataFrame column position."""
    asset_returns = pd.DataFrame(
        [[0.1, -0.05], [0.2, 0.0]], columns=["stock", "hedge"]
    )

    result = returns.portfolio_returns(asset_returns, np.array([0.6, 0.4]))

    pd.testing.assert_series_equal(result, pd.Series([0.04, 0.12]))


def test_returns_use_float_arithmetic_and_propagate_missing_values():
    """Integer input cannot overflow and missing asset data stays missing."""
    integer_returns = np.array([1, 2], dtype=np.int64)
    integer_weights = np.array([2, 3], dtype=np.int64)
    assert returns.portfolio_returns(integer_returns, integer_weights) == 8.0

    asset_returns = np.array([[1.0, 2.0], [np.nan, 1.0]])
    original = asset_returns.copy()

    result = returns.portfolio_returns(asset_returns, np.array([2, 3]))

    np.testing.assert_allclose(result, [8.0, np.nan], equal_nan=True)
    np.testing.assert_allclose(asset_returns, original, equal_nan=True)


def test_returns_allow_nonstandard_weights_and_negative_returns():
    """The caller controls the weight sum and financial return convention."""
    result = returns.portfolio_returns(
        np.array([-1.5, 0.1]), np.array([0.2, 0.1])
    )

    assert result == pytest.approx(-0.29)


@pytest.mark.parametrize(
    ("asset_returns", "weights"),
    [
        (np.array(0.1), np.array([1.0])),
        (np.ones((1, 1, 1)), np.ones((1, 1, 1))),
        (np.ones((2, 2)), np.ones((2, 1))),
    ],
)
def test_returns_reject_invalid_asset_dimensions(asset_returns, weights):
    """Asset inputs only accept compatible nonempty vectors or matrices."""
    with pytest.raises(ValueError):
        returns.portfolio_returns(asset_returns, weights)
