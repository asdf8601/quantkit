"""Tests for portfolio flows and time-weighted returns."""

import numpy as np
import pandas as pd
import pytest

from quantkit.portfolio.flows import (
    period_return,
    profit_loss,
    time_weighted_return,
)


def test_contribution_is_removed_from_profit_and_loss():
    assert profit_loss(100.0, 200.0, 100.0) == 0.0


def test_flow_timing_changes_the_invested_capital():
    assert period_return(100.0, 220.0, 100.0, flow_timing="beginning") == 0.1
    assert period_return(100.0, 220.0, 100.0, flow_timing="end") == 0.2


def test_withdrawal_has_negative_flow_sign_and_timing_effect():
    assert profit_loss(100.0, 45.0, -50.0) == -5.0
    assert period_return(100.0, 45.0, -50.0) == -0.05
    assert period_return(100.0, 45.0, -50.0, flow_timing="beginning") == -0.1


def test_interval_history_and_linked_return():
    beginning = np.array([100.0, 210.0])
    ending = np.array([210.0, 189.0])
    flow = np.array([100.0, 0.0])

    np.testing.assert_allclose(
        profit_loss(beginning, ending, flow), [10.0, -21.0]
    )
    np.testing.assert_allclose(
        period_return(beginning, ending, flow), [0.1, -0.1]
    )
    assert time_weighted_return(beginning, ending, flow) == pytest.approx(
        -0.01
    )


def test_series_labels_align_and_first_metadata_is_preserved():
    beginning = pd.Series([100.0, 210.0], index=["a", "b"], name="NAV")
    ending = pd.Series([189.0, 210.0], index=["b", "a"], name="other")
    flow = pd.Series([0.0, 100.0], index=["b", "a"])

    expected_profit = pd.Series(
        [10.0, -21.0], index=beginning.index, name="NAV"
    )
    expected_return = pd.Series([0.1, -0.1], index=beginning.index, name="NAV")
    pd.testing.assert_series_equal(
        profit_loss(beginning, ending, flow), expected_profit
    )
    pd.testing.assert_series_equal(
        period_return(beginning, ending, flow), expected_return
    )
    assert time_weighted_return(beginning, ending, flow) == pytest.approx(
        -0.01
    )


def test_nullable_values_and_missing_values_propagate_without_mutation():
    beginning = pd.Series([100, 200], dtype="Int64", name="NAV")
    ending = pd.Series([110, None], dtype="Int64")
    flow = pd.Series([0, 0], dtype="Int64")
    original = ending.copy()

    expected = pd.Series([0.1, np.nan], name="NAV")
    pd.testing.assert_series_equal(
        period_return(beginning, ending, flow), expected
    )
    assert np.isnan(time_weighted_return(beginning, ending, flow))
    pd.testing.assert_series_equal(ending, original)


@pytest.mark.parametrize(
    ("beginning", "ending", "flow_timing"),
    [
        (0.0, 1.0, "end"),
        (-1.0, 1.0, "end"),
        (np.nan, 1.0, "end"),
        (100.0, 0.0, "beginning"),
    ],
)
def test_nonpositive_or_missing_denominator_is_undefined(
    beginning, ending, flow_timing
):
    flow = -100.0 if flow_timing == "beginning" else 0.0
    assert np.isnan(
        period_return(beginning, ending, flow, flow_timing=flow_timing)
    )


def test_compounding_rejects_returns_below_negative_one_but_accepts_loss_of_one():
    assert time_weighted_return(100.0, 0.0) == -1.0
    with pytest.raises(ValueError, match="below -1"):
        time_weighted_return(100.0, -1.0)


@pytest.mark.parametrize(
    "values",
    [
        pd.DataFrame({"x": [1.0]}),
        np.array([[1.0]]),
        np.array([]),
        np.array([True]),
        np.array(["1"]),
        np.array([np.inf]),
        np.array([1j]),
        np.array([np.datetime64("2026-01-01")]),
        np.array([np.timedelta64(1, "D")]),
    ],
)
def test_rejects_invalid_beginning_values(values):
    with pytest.raises((TypeError, ValueError)):
        profit_loss(values, values)


def test_rejects_shape_dimension_label_and_timing_mismatches():
    with pytest.raises(ValueError, match="shape"):
        profit_loss(np.array([1.0]), 1.0)
    with pytest.raises(ValueError, match="shape"):
        profit_loss(1.0, np.array([1.0]))
    with pytest.raises(ValueError, match="external_flow"):
        profit_loss(np.ones(2), np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="labels"):
        period_return(
            pd.Series([1.0], index=["a"]),
            pd.Series([1.0], index=["b"]),
        )
    with pytest.raises(ValueError, match="duplicate"):
        period_return(
            pd.Series([1.0, 1.0], index=["a", "a"]),
            pd.Series([1.0, 1.0], index=["a", "b"]),
        )
    with pytest.raises(ValueError, match="flow_timing"):
        period_return(1.0, 1.0, flow_timing="middle")
