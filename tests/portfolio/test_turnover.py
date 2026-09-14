"""Turnover uses gross executed volume, not changes in net holdings."""

import numpy as np
import pandas as pd
import pytest

from quantkit.portfolio.turnover import turnover


def test_round_trip_is_not_netted():
    assert turnover(np.array([20., -20.]), 100.) == 0.2
    assert turnover(np.array([0., 0.]), 100.) == 0.
    assert turnover(np.array([20., 0.]), 100.) == 0.1


def test_history_and_time_labels():
    trades = pd.DataFrame([[20., -20.], [40., -40.]], index=["a", "b"])
    nav = pd.Series([200., 100.], index=["b", "a"])
    pd.testing.assert_series_equal(turnover(trades, nav), pd.Series([.2, .2], index=trades.index))
    np.testing.assert_allclose(turnover(trades.to_numpy(), np.array([100., 200.])), [.2, .2])


@pytest.mark.parametrize("nav", [0., -1., np.nan])
def test_nonpositive_or_missing_capital_is_undefined(nav):
    assert np.isnan(turnover(np.array([20., -20.]), nav))


def test_missing_trade_propagates_and_inputs_unchanged():
    trades = np.array([[np.nan, 0.], [20., -20.]])
    original = trades.copy()
    np.testing.assert_allclose(turnover(trades, 100.), [np.nan, .2], equal_nan=True)
    np.testing.assert_array_equal(trades, original)


def test_scale_invariance():
    trades = np.array([10., -5., 15.])
    assert turnover(trades * 3, 300.) == turnover(trades, 100.)


def test_reject_mismatched_capital_labels():
    with pytest.raises(ValueError):
        turnover(pd.DataFrame([[20.]], index=["a"]), pd.Series([100.], index=["b"]))
