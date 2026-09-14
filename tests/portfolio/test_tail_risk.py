"""Tests for historical portfolio tail-risk measures."""

import numpy as np
import pandas as pd
import pytest

from quantkit.portfolio.tail_risk import (
    conditional_drawdown_at_risk,
    expected_shortfall,
)


def test_expected_shortfall_integrates_fractional_boundary():
    """The exact tail mass includes half of the second-worst observation."""
    returns = np.array([-0.4, -0.2, 0.1, 0.2])

    result = expected_shortfall(returns, confidence=0.625)

    assert isinstance(result, float)
    assert result == pytest.approx((0.4 + 0.5 * 0.2) / 1.5)


def test_expected_shortfall_handles_ties_high_confidence_and_all_gains():
    """Tail integration is stable at ties and at a sub-observation mass."""
    assert expected_shortfall(
        np.array([-0.3, -0.3, 0.2]), 0.5
    ) == pytest.approx(0.3)
    assert expected_shortfall(
        np.array([-0.4, -0.2, 0.1]), 0.99
    ) == pytest.approx(0.4)
    assert expected_shortfall(np.array([0.1, 0.2, 0.3]), 0.5) == pytest.approx(
        -(0.1 + 0.5 * 0.2) / 1.5
    )


def test_expected_shortfall_reduces_time_axis_and_preserves_column_labels():
    """A non-square frame produces one unnamed value per portfolio column."""
    returns = pd.DataFrame(
        {
            "core": [-0.4, -0.2, 0.1, 0.2],
            "hedge": [-0.1, 0.0, 0.2, 0.3],
            "gain": [0.1, 0.2, 0.3, 0.4],
        },
        index=["t0", "t1", "t2", "t3"],
    )

    result = expected_shortfall(returns, confidence=0.625)

    pd.testing.assert_series_equal(
        result,
        pd.Series(
            [1 / 3, (0.1 + 0.5 * 0.0) / 1.5, -(0.1 + 0.5 * 0.2) / 1.5],
            index=returns.columns,
            name=None,
        ),
    )


def test_expected_shortfall_ndarray_matrix_has_one_value_per_column():
    """The sample axis is axis zero even for a non-square ndarray."""
    returns = np.array(
        [[-0.4, 0.1], [-0.2, 0.2], [0.1, 0.3], [0.2, 0.4], [0.3, 0.5]]
    )

    result = expected_shortfall(returns, confidence=0.7)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    np.testing.assert_allclose(
        result, [(0.4 + 0.5 * 0.2) / 1.5, (-0.1 + 0.5 * -0.2) / 1.5]
    )


def test_tail_risk_propagates_nan_per_column_and_accepts_nullable_numeric():
    """One missing observation invalidates only its own portfolio history."""
    returns = pd.DataFrame(
        {
            "complete": pd.Series([-0.2, 0.1, 0.2], dtype="Float64"),
            "missing": pd.Series([-0.4, pd.NA, 0.3], dtype="Float64"),
        }
    )
    wealth = pd.DataFrame(
        {
            "complete": pd.Series([100, 120, 90], dtype="Int64"),
            "missing": pd.Series([100, pd.NA, 90], dtype="Int64"),
        }
    )

    es = expected_shortfall(returns, confidence=0.5)
    cdar = conditional_drawdown_at_risk(wealth, confidence=0.5)

    assert es["complete"] == pytest.approx((0.2 + 0.5 * -0.1) / 1.5)
    assert cdar["complete"] == pytest.approx((0.25 + 0.5 * 0.0) / 1.5)
    assert np.isnan(es["missing"])
    assert np.isnan(cdar["missing"])


def test_conditional_drawdown_at_risk_uses_observed_drawdown_tail():
    """The known wealth path has drawdowns zero, zero, 25%, and 10%."""
    wealth = np.array([100.0, 120.0, 90.0, 108.0])

    result = conditional_drawdown_at_risk(wealth, confidence=0.5)

    assert isinstance(result, float)
    assert result == pytest.approx((0.25 + 0.10) / 2)


def test_conditional_drawdown_at_risk_includes_initial_observation():
    """The initial zero drawdown carries equal empirical probability."""
    result = conditional_drawdown_at_risk(
        np.array([100.0, 80.0]), confidence=0.25
    )

    assert result == pytest.approx((0.2 + 0.5 * 0.0) / 1.5)


def test_conditional_drawdown_at_risk_is_scale_invariant():
    """Changing wealth units does not change relative drawdown risk."""
    wealth = np.array([[100.0, 80.0], [120.0, 100.0], [90.0, 50.0]])

    np.testing.assert_allclose(
        conditional_drawdown_at_risk(wealth * 37, confidence=0.6),
        conditional_drawdown_at_risk(wealth, confidence=0.6),
    )


@pytest.mark.parametrize("initial", [0.0, -1.0])
def test_conditional_drawdown_at_risk_requires_positive_initial_wealth(
    initial,
):
    """An observed initial baseline cannot be zero or negative."""
    with pytest.raises(ValueError, match="initial wealth"):
        conditional_drawdown_at_risk(np.array([initial, 1.0]))


def test_conditional_drawdown_at_risk_rejects_negative_later_wealth():
    """A compounded wealth history cannot become negative."""
    with pytest.raises(ValueError, match="nonnegative"):
        conditional_drawdown_at_risk(np.array([1.0, -0.1]))


def test_missing_initial_wealth_propagates():
    """An unknown initial baseline produces an unknown result."""
    assert np.isnan(conditional_drawdown_at_risk(np.array([np.nan, 1.0])))


@pytest.mark.parametrize(
    "confidence",
    [True, False, "0.95", None, np.array(0.95), np.inf, np.nan, 0, 1],
)
def test_tail_risk_rejects_invalid_confidence(confidence):
    """Confidence must be a finite, non-boolean real scalar inside (0, 1)."""
    for measure, values in (
        (expected_shortfall, np.array([0.1, -0.1])),
        (conditional_drawdown_at_risk, np.array([100.0, 90.0])),
    ):
        with pytest.raises((TypeError, ValueError)):
            measure(values, confidence)


@pytest.mark.parametrize(
    "values",
    [
        np.array([]),
        np.array([True, False]),
        np.array(["1", "2"]),
        np.array([1 + 0j]),
        np.array([1.0, np.inf]),
    ],
)
def test_tail_risk_rejects_invalid_observation_data(values):
    """Histories must be nonempty, finite, real numeric arrays."""
    with pytest.raises((TypeError, ValueError)):
        expected_shortfall(values)


def test_tail_risk_rejects_duplicate_pandas_labels():
    """Duplicate time or portfolio labels are ambiguous."""
    duplicate_time = pd.Series([0.1, -0.1], index=["t", "t"])
    duplicate_columns = pd.DataFrame([[0.1, 0.2]], columns=["p", "p"])

    with pytest.raises(ValueError, match="duplicate"):
        expected_shortfall(duplicate_time)
    with pytest.raises(ValueError, match="duplicate"):
        conditional_drawdown_at_risk(duplicate_columns)
