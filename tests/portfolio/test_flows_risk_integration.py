"""Independent cross-module checks for portfolio performance and risk."""

import numpy as np
import pandas as pd
import pytest

from quantkit.portfolio import flows, returns, risk, tail_risk, valuation


def test_deposit_does_not_create_performance():
    times = pd.Index(["deposit", "market_gain"], name="period")
    beginning_positions = pd.DataFrame({"stock": [100., 200.]}, index=times)
    ending_positions = pd.DataFrame({"stock": [200., 220.]}, index=times)
    beginning = valuation.net_asset_value(beginning_positions)
    ending = valuation.net_asset_value(ending_positions)
    external = pd.Series([100., 0.], index=times)
    pd.testing.assert_series_equal(
        flows.profit_loss(beginning, ending, external),
        pd.Series([0., 20.], index=times),
    )
    pd.testing.assert_series_equal(
        flows.period_return(beginning, ending, external),
        pd.Series([0., .1], index=times),
    )
    assert flows.time_weighted_return(beginning, ending, external) == pytest.approx(.1)


def test_covariance_risk_matches_actual_weighted_returns():
    history = pd.DataFrame({"a": [.1, -.1, .05, .02], "b": [-.02, .03, .04, -.01]})
    allocation = pd.Series([.6, .4], index=history.columns)
    realized = returns.portfolio_returns(history, allocation)
    covariance = history.cov()
    assert risk.variance(allocation, covariance) == pytest.approx(realized.var(ddof=1))
    assert risk.volatility(allocation, covariance) == pytest.approx(realized.std(ddof=1))
    components = risk.risk_contribution(allocation, covariance)
    assert components.sum() == pytest.approx(realized.std(ddof=1))
    pd.testing.assert_index_equal(components.index, allocation.index)


def test_tail_drawdowns_use_flow_adjusted_wealth():
    beginning = np.array([100., 220.])
    ending = np.array([220., 198.])
    period_returns = flows.period_return(beginning, ending, np.array([100., 0.]))
    # Returns are +20%, then -10%; the deposit is not investment gain.
    np.testing.assert_allclose(period_returns, [.2, -.1])
    wealth = np.r_[100., 100. * np.cumprod(1. + period_returns)]
    np.testing.assert_allclose(wealth, [100., 120., 108.])
    assert tail_risk.expected_shortfall(period_returns, confidence=.75) == pytest.approx(.1)
    assert tail_risk.conditional_drawdown_at_risk(wealth, confidence=.75) == pytest.approx(.1)
