"""Interactive portfolio tour using local historical prices."""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=False)
def _():
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    from _data import load_prices

    import quantkit as qnt

    mo.show_code(position="above")
    return (
        load_prices,
        mo,
        pd,
        px,
        qnt,
    )


@app.cell(hide_code=False)
def _(mo):
    mo.md(r"""
    # Portfolio: from positions to performance and risk

    This example uses **real** historical quotes stored in the repository.
    The positions, trades, and cash flows are entirely **fictional** and do
    not constitute investment advice.
    """)
    return


@app.cell(hide_code=False)
def _(load_prices, mo):
    market_data = load_prices()
    mo.show_code(position="above")
    return (market_data,)


@app.cell(hide_code=False)
def _(market_data, mo, pd):
    _data, _metadata = market_data
    _dates = pd.DatetimeIndex(_data["date"].drop_duplicates().sort_values())
    _years = _dates.year.unique()
    controls = mo.ui.dictionary(
        {
            "Period": mo.ui.range_slider(
                int(_years.min()),
                int(_years.max()),
                value=[2020, int(_years.max())],
                step=1,
                show_value=True,
                label="Years for the history",
            ),
            "Date": mo.ui.dropdown(
                {str(_date.date()): _date for _date in _dates},
                value=str(_dates[-1].date()),
                searchable=True,
                label="Valuation date",
            ),
            "SPY": mo.ui.slider(0, 25, value=10, step=1, show_value=True),
            "QQQ": mo.ui.slider(-3, 12, value=-2, step=1, show_value=True),
            "TLT": mo.ui.slider(0, 40, value=20, step=1, show_value=True),
            "GLD": mo.ui.slider(0, 20, value=8, step=1, show_value=True),
        },
        label="Fictional assumptions",
    )
    mo.show_code(position="above")
    return (controls,)


@app.cell(hide_code=False)
def _(controls, market_data, mo):
    _metadata = market_data[1]
    _output = mo.vstack(
        [
            controls,
            mo.md(
                f"Snapshot source: **{_metadata['source']}**, "
                f"{_metadata['first_date']} to {_metadata['last_date']}. "
                "Fixed cash: **USD 10,000**. QQQ allows a small short; its "
                "limits keep NAV positive throughout the snapshot."
            ),
        ]
    )
    mo.show_code(_output, position="above")
    return


@app.cell(hide_code=False)
def _(controls, mo, pd):
    selected_date = pd.Timestamp(controls.value["Date"])
    mo.show_code(position="above")
    return (selected_date,)


@app.cell(hide_code=False)
def _(controls, mo, pd):
    quantities = pd.Series(
        {
            ticker: float(controls.value[ticker])
            for ticker in ["SPY", "QQQ", "TLT", "GLD"]
        },
        name="Units",
    )
    mo.show_code(position="above")
    return (quantities,)


@app.cell(hide_code=False)
def _(market_data, mo, selected_date):
    _close = market_data[0].pivot(
        index="date", columns="ticker", values="close"
    )
    selected_prices = _close.loc[selected_date].rename("Close")
    mo.show_code(position="above")
    return (selected_prices,)


@app.cell(hide_code=False)
def _(mo, qnt, quantities, selected_prices):
    position_values = qnt.portfolio.valuation.position_values(
        quantities, selected_prices
    )
    mo.show_code(position="above")
    return (position_values,)


@app.cell(hide_code=False)
def _(mo, position_values, qnt):
    nav = qnt.portfolio.valuation.net_asset_value(
        position_values, cash=10_000.0
    )
    mo.show_code(position="above")
    return (nav,)


@app.cell(hide_code=False)
def _(mo, nav, position_values, qnt):
    portfolio_weights = qnt.portfolio.weights.nav_weights(position_values, nav)
    mo.show_code(position="above")
    return (portfolio_weights,)


@app.cell(hide_code=False)
def _(
    mo,
    pd,
    portfolio_weights,
    position_values,
    qnt,
    quantities,
    selected_prices,
):
    _table = pd.concat(
        [
            quantities,
            selected_prices.rename("Close (USD)"),
            position_values.rename("Signed value (USD)"),
            (100.0 * portfolio_weights).rename("NAV weight (%)"),
            (
                100.0 * qnt.portfolio.weights.gross_weights(position_values)
            ).rename("Gross weight (%)"),
        ],
        axis=1,
    )
    snapshot_table = _table
    mo.show_code(position="above")
    return (snapshot_table,)


@app.cell(hide_code=False)
def _(mo, nav, pd, position_values, qnt):
    snapshot_metrics = pd.DataFrame(
        {
            "Metric": [
                "NAV",
                "GAV",
                "Long value",
                "Short value",
                "Gross exposure",
                "Net exposure",
                "Gross leverage",
                "H concentration",
                "Effective number of assets",
            ],
            "Value": [
                nav,
                qnt.portfolio.valuation.gross_asset_value(
                    position_values, cash=10_000.0
                ),
                qnt.portfolio.exposure.long_value(position_values),
                qnt.portfolio.exposure.short_value(position_values),
                qnt.portfolio.exposure.gross_exposure(position_values),
                qnt.portfolio.exposure.net_exposure(position_values),
                qnt.portfolio.exposure.gross_leverage(position_values, nav),
                qnt.portfolio.weights.concentration(position_values),
                qnt.portfolio.weights.effective_number_of_assets(
                    position_values
                ),
            ],
            "Unit": ["USD"] * 6 + ["ratio"] * 3,
        }
    )
    mo.show_code(position="above")
    return (snapshot_metrics,)


@app.cell(hide_code=False)
def _(mo, selected_date, snapshot_metrics, snapshot_table):
    mo.vstack(
        [
            mo.md(
                f"## 1. Balance sheet on {selected_date.date()}\n\n"
                "The unadjusted close values the units. A short is already a "
                "negative signed value, so it is not recorded again as a "
                "liability. GAV adds long positions and positive cash; gross "
                "exposure uses magnitudes and excludes cash. With no "
                "positions, "
                "concentration and the effective asset count are undefined."
            ),
            mo.ui.table(
                snapshot_table.reset_index(names="Ticker").round(4),
                pagination=False,
            ),
            mo.ui.table(snapshot_metrics.round(4), pagination=False),
        ]
    )
    return


@app.cell(hide_code=False)
def _(controls, market_data, mo):
    _start, _end = controls.value["Period"]
    _adjusted = market_data[0].pivot(
        index="date", columns="ticker", values="adjusted_close"
    )
    history_prices = _adjusted.loc[str(int(_start)) : str(int(_end))]
    mo.show_code(position="above")
    return (history_prices,)


@app.cell(hide_code=False)
def _(history_prices, market_data, mo, pd, qnt, quantities):
    _raw = market_data[0].pivot(index="date", columns="ticker", values="close")
    _start_values = qnt.portfolio.valuation.position_values(
        quantities, _raw.loc[history_prices.index[0]]
    )
    _start_nav = qnt.portfolio.valuation.net_asset_value(
        _start_values, cash=10_000.0
    )
    initial_weights = pd.concat(
        [
            qnt.portfolio.weights.nav_weights(_start_values, _start_nav),
            pd.Series({"CASH": 10_000.0 / _start_nav}),
        ]
    ).rename("Initial weight")
    mo.show_code(position="above")
    return (initial_weights,)


@app.cell(hide_code=False)
def _(history_prices, mo, qnt):
    _returns = qnt.core.returns(history_prices).iloc[1:].copy()
    _returns["CASH"] = 0.0
    asset_returns = _returns
    mo.show_code(position="above")
    return (asset_returns,)


@app.cell(hide_code=False)
def _(history_prices, initial_weights, mo, pd):
    _relative = history_prices.div(history_prices.iloc[0])
    _sleeves = _relative.mul(initial_weights.drop("CASH") * 100.0, axis=1)
    _sleeves["CASH"] = initial_weights["CASH"] * 100.0
    buy_hold_sleeves = _sleeves
    mo.show_code(position="above")
    return (buy_hold_sleeves,)


@app.cell(hide_code=False)
def _(asset_returns, buy_hold_sleeves, mo):
    _weights = buy_hold_sleeves.div(buy_hold_sleeves.sum(axis=1), axis=0).iloc[
        :-1
    ]
    _weights.index = asset_returns.index
    buy_hold_beginning_weights = _weights
    mo.show_code(position="above")
    return (buy_hold_beginning_weights,)


@app.cell(hide_code=False)
def _(asset_returns, buy_hold_beginning_weights, mo, qnt):
    buy_hold_returns = qnt.portfolio.returns.portfolio_returns(
        asset_returns, buy_hold_beginning_weights
    )
    mo.show_code(position="above")
    return (buy_hold_returns,)


@app.cell(hide_code=False)
def _(asset_returns, initial_weights, mo, qnt):
    rebalanced_returns = qnt.portfolio.returns.portfolio_returns(
        asset_returns, initial_weights
    )
    mo.show_code(position="above")
    return (rebalanced_returns,)


@app.cell(hide_code=False)
def _(buy_hold_returns, history_prices, mo, pd):
    buy_hold_wealth = pd.concat(
        [
            pd.Series([100.0], index=history_prices.index[:1]),
            100.0 * (1.0 + buy_hold_returns).cumprod(),
        ]
    ).rename("Buy and hold")
    mo.show_code(position="above")
    return (buy_hold_wealth,)


@app.cell(hide_code=False)
def _(history_prices, mo, pd, rebalanced_returns):
    rebalanced_wealth = pd.concat(
        [
            pd.Series([100.0], index=history_prices.index[:1]),
            100.0 * (1.0 + rebalanced_returns).cumprod(),
        ]
    ).rename("Daily rebalancing")
    mo.show_code(position="above")
    return (rebalanced_wealth,)


@app.cell(hide_code=False)
def _(buy_hold_wealth, mo, pd, rebalanced_wealth):
    performance = pd.concat([buy_hold_wealth, rebalanced_wealth], axis=1)
    mo.show_code(position="above")
    return (performance,)


@app.cell(hide_code=False)
def _(initial_weights, mo, performance, px):
    _long = (
        performance.rename_axis("Date")
        .reset_index()
        .melt("Date", var_name="Strategy", value_name="Wealth")
    )
    _figure = px.line(
        _long,
        x="Date",
        y="Wealth",
        color="Strategy",
        title="Hypothetical growth of USD 100",
    )
    _figure.update_layout(hovermode="x unified", yaxis_title="Wealth index")
    _output = mo.vstack(
        [
            mo.md(
                "## 2. Two comparable investment rules\n\n"
                "Each asset is a **hypothetical total-return allocation** "
                "based on `adjusted_close`; the initial amounts are "
                "allocations, not "
                "shares from the balance sheet above. The same fictional "
                "quantities are valued at the first close in the period to "
                "obtain the weights, including zero-return cash. Buy and hold "
                "lets weights drift; daily rebalancing restores them each "
                "day. There are no costs. Every return uses the weight at "
                "the start "
                "of the interval, before the next close is known."
            ),
            mo.ui.table(
                (100.0 * initial_weights)
                .rename("Initial weight (%)")
                .rename_axis("Asset")
                .reset_index()
                .round(4),
                pagination=False,
            ),
            _figure,
        ]
    )
    mo.show_code(_output, position="above")
    return


@app.cell(hide_code=False)
def _(asset_returns, mo):
    covariance = asset_returns.drop(columns="CASH").cov(ddof=1) * 252
    mo.show_code(position="above")
    return (covariance,)


@app.cell(hide_code=False)
def _(covariance, mo, portfolio_weights, qnt):
    portfolio_volatility = qnt.portfolio.risk.volatility(
        portfolio_weights, covariance
    )
    mo.show_code(position="above")
    return (portfolio_volatility,)


@app.cell(hide_code=False)
def _(covariance, mo, portfolio_weights, qnt):
    risk_contributions = qnt.portfolio.risk.risk_contribution(
        portfolio_weights, covariance
    )
    mo.show_code(position="above")
    return (risk_contributions,)


@app.cell(hide_code=False)
def _(
    covariance,
    mo,
    pd,
    portfolio_volatility,
    portfolio_weights,
    qnt,
    risk_contributions,
):
    risk_table = pd.DataFrame(
        {
            "Signed weight (%)": 100.0 * portfolio_weights,
            "Volatility contribution (pp)": 100.0 * risk_contributions,
            "Share of risk (%)": 100.0
            * risk_contributions
            / portfolio_volatility,
        }
    )
    _variance = qnt.portfolio.risk.variance(portfolio_weights, covariance)
    risk_table.attrs["variance"] = _variance
    mo.show_code(position="above")
    return (risk_table,)


@app.cell(hide_code=False)
def _(mo, portfolio_volatility, risk_contributions, risk_table):
    _undefined = risk_contributions.isna().all()
    _sum = "Undefined" if _undefined else f"{risk_contributions.sum():.2%}"
    _note = (
        " With no asset exposure, volatility and contributions do not provide "
        "a useful decomposition."
        if _undefined
        else ""
    )
    _output = mo.vstack(
        [
            mo.md(
                "## 3. Risk of the mix\n\n"
                "The period's sample covariance uses daily returns, `ddof=1`, "
                "and an annualization factor of 252. **Retrospective "
                "estimate:** "
                "it applies the valuation-date weights to the entire selected "
                "period, which may include later dates. It is not a risk "
                "estimate made with information available at that time. "
                "Weights "
                f"are signed; an allocation can hedge the others.{_note}"
            ),
            mo.hstack(
                [
                    mo.stat(
                        f"{risk_table.attrs['variance']:.6f}",
                        label="Annual variance (return²)",
                    ),
                    mo.stat(
                        f"{portfolio_volatility:.2%}",
                        label="Annual volatility",
                    ),
                    mo.stat(
                        _sum,
                        label="Sum of contributions",
                    ),
                ],
                justify="start",
            ),
            mo.ui.table(
                risk_table.reset_index(names="Asset").round(5),
                pagination=False,
            ),
        ]
    )
    mo.show_code(_output, position="above")
    return


@app.cell(hide_code=False)
def _(buy_hold_returns, mo, pd, qnt):
    _daily = buy_hold_returns.iloc[:2]
    _beginning = pd.Series(
        [10_000.0, 0.0], index=_daily.index, name="Beginning NAV"
    )
    _external = pd.Series(
        [2_000.0, -1_000.0], index=_daily.index, name="End flow"
    )
    _ending = _beginning.copy().rename("Ending NAV")
    _ending.iloc[0] = (
        _beginning.iloc[0] * (1.0 + _daily.iloc[0]) + _external.iloc[0]
    )
    _beginning.iloc[1] = _ending.iloc[0]
    _ending.iloc[1] = (
        _beginning.iloc[1] * (1.0 + _daily.iloc[1]) + _external.iloc[1]
    )
    _pnl = qnt.portfolio.flows.profit_loss(_beginning, _ending, _external)
    _period_returns = qnt.portfolio.flows.period_return(
        _beginning, _ending, _external, flow_timing="end"
    )
    _table = pd.DataFrame(
        {
            "Interval end": _daily.index.date,
            "Beginning NAV (USD)": _beginning,
            "Market return (%)": 100.0 * _daily,
            "End flow (USD)": _external,
            "Ending NAV (USD)": _ending,
            "P&L (USD)": _pnl,
            "Period return (%)": 100.0 * _period_returns,
        }
    ).reset_index(drop=True)
    flow_example = {
        "table": _table,
        "twr": qnt.portfolio.flows.time_weighted_return(
            _beginning, _ending, _external, flow_timing="end"
        ),
        "funded_wealth": float(_ending.iloc[-1]),
        "no_flow_wealth": float(10_000.0 * (1.0 + _daily).prod()),
    }
    mo.show_code(position="above")
    return (flow_example,)


@app.cell(hide_code=False)
def _(flow_example, mo):
    mo.vstack(
        [
            mo.md(
                "## 4. Cash flows: funded money is not performance\n\n"
                "Two real daily intervals receive a hypothetical contribution "
                "and withdrawal at each close. The second beginning NAV is "
                "the "
                "first ending NAV, so funded capital evolves between periods. "
                "Each flow is assumed to scale every allocation, including "
                "cash, "
                "in the same proportion."
            ),
            mo.ui.table(flow_example["table"].round(4), pagination=False),
            mo.hstack(
                [
                    mo.stat(f"{flow_example['twr']:.3%}", label="Linked TWR"),
                    mo.stat(
                        f"${flow_example['funded_wealth']:,.2f}",
                        label="Wealth with flows",
                    ),
                    mo.stat(
                        f"${flow_example['no_flow_wealth']:,.2f}",
                        label="Wealth without flows",
                    ),
                ],
                justify="start",
            ),
        ]
    )
    return


@app.cell(hide_code=False)
def _(mo):
    mo.md("""
    ### Cash balances at different timestamps

    `cash` is the **uninvested cash balance at each timestamp**, not total
    contributions or total portfolio value. Pass a Series indexed like the
    position-value DataFrame. Quantkit aligns the balances by timestamp;
    it does not reconstruct them from transactions.

    This small, entirely **synthetic example** makes the accounting explicit.
    Start with 70 shares at USD 100 and USD 3,000 cash. Deposit USD 5,000,
    buy another 20 shares for USD 2,000, then let the price rise to USD 110.
    Finally, deposit USD 1,000 and withdraw USD 2,000. Assume no fees.

    Each snapshot records holdings and cash **after** the event. Deposits
    and withdrawals occur at interval ends. The purchase transfers cash
    into securities and is not an external flow. This example is independent
    of the historical-data controls above.
    """)
    return


@app.cell(hide_code=False)
def _(mo, pd, qnt):
    cash_example_times = pd.DatetimeIndex(
        [
            "2025-01-02 10:00",
            "2025-01-02 12:00",
            "2025-01-02 14:00",
            "2025-01-03 10:00",
            "2025-01-03 12:00",
            "2025-01-03 14:00",
        ],
        name="Timestamp",
    )
    cash_example_quantities = pd.DataFrame(
        {"stock": [70.0, 70.0, 90.0, 90.0, 90.0, 90.0]},
        index=cash_example_times,
    )
    cash_example_prices = pd.DataFrame(
        {"stock": [100.0, 100.0, 100.0, 110.0, 110.0, 110.0]},
        index=cash_example_times,
    )
    cash_balances = pd.Series(
        [3000.0, 8000.0, 6000.0, 6000.0, 7000.0, 5000.0],
        index=cash_example_times,
    )
    cash_example_flows = pd.Series(
        [0.0, 5000.0, 0.0, 0.0, 1000.0, -2000.0], index=cash_example_times
    )
    cash_example_positions = qnt.portfolio.valuation.position_values(
        cash_example_quantities, cash_example_prices
    )
    cash_example_nav = qnt.portfolio.valuation.net_asset_value(
        cash_example_positions, cash=cash_balances
    )
    mo.show_code(position="above")
    return (
        cash_balances,
        cash_example_flows,
        cash_example_nav,
        cash_example_positions,
        cash_example_prices,
        cash_example_quantities,
        cash_example_times,
    )


@app.cell(hide_code=False)
def _(cash_example_flows, cash_example_nav, mo, qnt):
    _beginning = cash_example_nav.shift(1).iloc[1:]
    _ending = cash_example_nav.iloc[1:]
    _external = cash_example_flows.iloc[1:]
    cash_example_pnl = qnt.portfolio.flows.profit_loss(
        _beginning, _ending, _external
    )
    cash_example_returns = qnt.portfolio.flows.period_return(
        _beginning, _ending, _external, flow_timing="end"
    )
    cash_example_twr = qnt.portfolio.flows.time_weighted_return(
        _beginning, _ending, _external, flow_timing="end"
    )
    mo.show_code(position="above")
    return cash_example_pnl, cash_example_returns, cash_example_twr


@app.cell(hide_code=False)
def _(
    cash_balances,
    cash_example_flows,
    cash_example_nav,
    cash_example_pnl,
    cash_example_positions,
    cash_example_returns,
    cash_example_twr,
    mo,
    pd,
):
    _balances = pd.DataFrame(
        {
            "Positions (USD)": cash_example_positions["stock"],
            "Cash balance (USD)": cash_balances,
            "NAV (USD)": cash_example_nav,
        }
    )
    _performance = pd.DataFrame(
        {
            "External flow (USD)": cash_example_flows.iloc[1:],
            "P&L (USD)": cash_example_pnl,
            "Period return (%)": 100.0 * cash_example_returns,
        }
    )
    mo.vstack(
        [
            mo.ui.table(_balances.reset_index(), pagination=False),
            mo.ui.table(_performance.reset_index(), pagination=False),
            mo.stat(f"{cash_example_twr:.2%}", label="Cash example TWR"),
            mo.md("""
            The first deposit raises NAV from **USD 10,000 to USD 15,000**.
            The purchase moves USD 2,000 from cash to positions, leaving
            NAV at **USD 15,000**. Only the price increase earns a profit:
            **90 shares × USD 10 = USD 900**, or **6%** of the preceding NAV.

            The later deposit and withdrawal also have zero adjusted return.
            Final NAV is **USD 14,900** and the linked TWR is **6%**.
            For flows inside an interval, split it at the flow timestamps
            and supply valuations there to calculate an exact TWR.
            """),
        ]
    )
    return


@app.cell(hide_code=False)
def _(mo, nav, pd, qnt):
    _trades = pd.Series(
        {"SPY": 1_200.0, "QQQ": 0.0, "TLT": -700.0, "GLD": -500.0},
        name="Executed trade",
    )
    turnover_example = {
        "trades": _trades,
        "turnover": qnt.portfolio.turnover.turnover(_trades, nav),
    }
    mo.show_code(position="above")
    return (turnover_example,)


@app.cell(hide_code=False)
def _(mo, turnover_example):
    mo.vstack(
        [
            mo.md(
                "### Turnover from an executed order\n\n"
                "The numerator is half the absolute volume bought and sold. "
                "These are explicit trades; a weight change caused only by "
                "the "
                "market is not turnover."
            ),
            mo.ui.table(
                turnover_example["trades"]
                .rename("Trade (USD)")
                .rename_axis("Asset")
                .reset_index(),
                pagination=False,
            ),
            mo.stat(
                f"{turnover_example['turnover']:.3%}", label="Turnover / NAV"
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
