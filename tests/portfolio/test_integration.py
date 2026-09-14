"""Independent economic examples across portfolio modules and existing stats."""

import numpy as np
import pandas as pd
import pytest

from quantkit import core, stats
from quantkit.portfolio import exposure, returns, valuation, weights


def test_long_short_balance_sheet_and_scaling():
    positions = pd.Series([100.0, -40.0], index=["long", "short"])
    nav = valuation.net_asset_value(positions, cash=40)
    assert nav == 100
    assert valuation.gross_asset_value(positions, cash=40) == 140
    assert exposure.gross_exposure(positions) == 140
    assert exposure.net_exposure(positions) == 60
    assert exposure.gross_leverage(positions, nav) == 1.4
    pd.testing.assert_series_equal(
        weights.nav_weights(positions, nav),
        pd.Series([1.0, -0.4], index=positions.index),
    )
    scaled = positions * 7
    assert valuation.net_asset_value(scaled, cash=280) == 7 * nav
    pd.testing.assert_series_equal(
        weights.gross_weights(scaled), weights.gross_weights(positions)
    )


def test_buy_and_hold_differs_from_constant_weight_rebalancing():
    prices = pd.DataFrame(
        {"a": [100.0, 200.0, 100.0], "b": [100.0, 100.0, 100.0]},
        index=pd.date_range("2026-01-01", periods=3),
    )
    quantities = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    values = valuation.position_values(quantities, prices)
    nav = valuation.net_asset_value(values)
    pd.testing.assert_series_equal(
        nav, pd.Series([200.0, 300.0, 200.0], index=prices.index)
    )
    actual_returns = core.returns(nav)
    np.testing.assert_allclose(actual_returns, [np.nan, 0.5, -1 / 3])
    asset_returns = core.returns(prices).iloc[1:]
    rebalanced = returns.portfolio_returns(asset_returns, np.array([0.5, 0.5]))
    np.testing.assert_allclose(rebalanced, [0.5, -0.25])
    assert stats.total_returns(nav) == pytest.approx(0.0)
    assert core.cum_returns(rebalanced).iloc[-1] == pytest.approx(0.125)
    # Actual beginning weights reconstruct the held portfolio's returns.
    beginning = weights.nav_weights(values, nav).iloc[:-1].copy()
    beginning.index = asset_returns.index
    reconstructed = returns.portfolio_returns(asset_returns, beginning)
    pd.testing.assert_series_equal(reconstructed, actual_returns.iloc[1:])
    assert stats.max_drawdown(nav) == pytest.approx(-1 / 3)


def test_aggregated_returns_work_with_existing_ratios():
    asset_returns = pd.DataFrame({"a": [0.1, -0.1, 0.1], "cash": [0., 0., 0.]})
    portfolio = returns.portfolio_returns(asset_returns, np.array([0.5, 0.5]))
    expected = pd.Series([0.05, -0.05, 0.05])
    pd.testing.assert_series_equal(portfolio, expected)
    # Existing Sharpe uses population deviation; information ratio uses sample.
    assert stats.sharpe_ratio(portfolio, risk_free=0.0, factor=1) == pytest.approx(1 / np.sqrt(8))
    assert stats.sortino_ratio(portfolio, factor=1) == pytest.approx(1 / np.sqrt(3))
    benchmark = pd.Series([0., 0., 0.])
    assert stats.information_ratio(portfolio, benchmark, factor=1) == pytest.approx(
        1 / np.sqrt(12)
    )
