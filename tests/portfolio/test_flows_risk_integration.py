"""Independent cross-module checks for portfolio performance and risk."""

import numpy as np
import pandas as pd
import pytest

from quantkit.portfolio import flows, returns, risk, tail_risk, valuation


@pytest.mark.parametrize("reverse_cash_order", [False, True])
def test_timestamped_cash_balances_deposits_purchase_and_withdrawal(
    reverse_cash_order,
):
    times = pd.DatetimeIndex(
        [
            "2025-01-02 10:00",
            "2025-01-02 12:00",
            "2025-01-02 14:00",
            "2025-01-03 10:00",
            "2025-01-03 12:00",
            "2025-01-03 14:00",
        ],
        name="timestamp",
    )
    # Start with 70 shares at $100 and $3,000 in cash (NAV $10,000).
    # Deposit $5,000; buy 20 shares for $2,000; price rises to $110;
    # deposit another $1,000; withdraw $2,000. No fees or other movements.
    quantities = pd.DataFrame({"stock": [70., 70., 90., 90., 90., 90.]}, index=times)
    prices = pd.DataFrame({"stock": [100., 100., 100., 110., 110., 110.]}, index=times)
    cash = pd.Series([3000., 8000., 6000., 6000., 7000., 5000.], index=times)
    external = pd.Series([0., 5000., 0., 0., 1000., -2000.], index=times)
    if reverse_cash_order:
        cash = cash.iloc[::-1]  # Balances align by timestamp, not position.

    position_values = valuation.position_values(quantities, prices)
    nav = valuation.net_asset_value(position_values, cash=cash)
    pd.testing.assert_series_equal(
        nav,
        pd.Series([10000., 15000., 15000., 15900., 16900., 14900.], index=times),
    )
    # The purchase moves cash into securities without changing NAV.
    assert position_values.iloc[2, 0] - position_values.iloc[1, 0] == 2000.
    assert nav.iloc[2] == nav.iloc[1]

    # Snapshots include the flow at each interval's end. Purchases are
    # internal transfers, so they are not included in external flows.
    beginning = nav.shift(1).iloc[1:]
    ending = nav.iloc[1:]
    external = external.iloc[1:]
    pd.testing.assert_series_equal(
        flows.profit_loss(beginning, ending, external),
        pd.Series([0., 0., 900., 0., 0.], index=times[1:]),
    )
    pd.testing.assert_series_equal(
        flows.period_return(beginning, ending, external, flow_timing="end"),
        pd.Series([0., 0., .06, 0., 0.], index=times[1:]),
    )
    assert flows.time_weighted_return(
        beginning, ending, external, flow_timing="end"
    ) == pytest.approx(.06)


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
