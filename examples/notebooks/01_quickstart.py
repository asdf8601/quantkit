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
    mo.md("""
    # A quick tour of quantkit

    In 5–10 minutes, you will calculate returns, investment growth, drawdown,
    and risk metrics using local data. Choose an asset and date range, then
    see how each calculation responds.
    """)
    return


@app.cell(hide_code=False)
def _(load_prices, mo):
    market_data, metadata = load_prices()
    mo.show_code(position="above")
    return market_data, metadata


@app.cell(hide_code=False)
def _(market_data, mo):
    _ticker_options = sorted(market_data["ticker"].unique())
    ticker = mo.ui.dropdown(
        options=_ticker_options,
        value="SPY",
        label="Asset",
    )
    years = mo.ui.range_slider(
        start=2018,
        stop=2025,
        step=1,
        value=[2018, 2025],
        show_value=True,
        label="Years",
    )
    return ticker, years


@app.cell(hide_code=False)
def _(mo, ticker, years):
    mo.hstack([ticker, years], justify="start", widths="equal")
    return


@app.cell(hide_code=False)
def _(market_data, metadata, mo):
    _first_date = market_data["date"].min().date()
    _last_date = market_data["date"].max().date()
    _source = metadata["source"]
    mo.md(
        f"""
        ## The data

        The local sample contains daily closing prices from {_first_date} to
        {_last_date}, downloaded from {_source}. We use `adjusted_close`, a
        series adjusted for dividends and stock splits. It therefore reflects
        the usual assumption that dividends are reinvested, rather than the
        closing price shown on screen.
        """
    )
    return


@app.cell(hide_code=False)
def _(market_data, mo, pd, qnt, ticker, years):
    _start_year, _end_year = years.value
    _selected_rows = market_data.loc[
        (market_data["ticker"] == ticker.value)
        & market_data["date"].dt.year.between(_start_year, _end_year)
    ].sort_values("date")
    prices = _selected_rows.set_index("date")["adjusted_close"].rename(
        ticker.value
    )
    asset_returns = qnt.core.returns(prices).iloc[1:]
    _initial_return = pd.Series(
        [0.0], index=prices.index[:1], name=prices.name
    )
    _wealth_returns = pd.concat([_initial_return, asset_returns])
    wealth = qnt.core.cum_returns(_wealth_returns, first_price=100)
    mo.show_code(position="above")
    return asset_returns, prices, wealth


@app.cell(hide_code=False)
def _(mo):
    mo.md("""
    ## Rebase prices to a common starting value

    `qnt.core.rebase(prices, base=100)` expresses the price history starting at
    100, making assets with different price levels easier to compare.
    A value of 120 means a 20% increase from the starting point.

    For this complete adjusted-price series, rebasing gives the same path
    as compounding returns from 100. `rebase` currently returns a NumPy
    array, so we wrap it in a Series to retain the dates and asset name.
    """)
    return


@app.cell(hide_code=False)
def _(mo, pd, prices, qnt, wealth):
    rebased_prices = pd.Series(
        qnt.core.rebase(prices, base=100), index=prices.index, name=prices.name
    )
    _comparison = pd.DataFrame(
        {"Rebased prices": rebased_prices, "Compounded returns": wealth}
    ).iloc[[0, -1]]
    mo.show_code(_comparison, position="above")
    return (rebased_prices,)


@app.cell(hide_code=False)
def _(mo, qnt, wealth):
    drawdown = qnt.expanding.drawdown(wealth, relative=True)
    mo.show_code(position="above")
    return (drawdown,)


@app.cell(hide_code=False)
def _(asset_returns, mo, pd, prices, qnt, wealth):
    _total_return = qnt.stats.total_returns(prices, relative=True)
    _annual_return = qnt.stats.annualized_return(
        asset_returns, periods_per_year=252
    )
    _annual_volatility = qnt.stats.volatility(
        asset_returns, factor=252**0.5, ddof=0
    )
    _sharpe = qnt.stats.sharpe_ratio(
        asset_returns, risk_free=0.0, factor=252**0.5
    )
    _maximum_drawdown = qnt.stats.max_drawdown(wealth, relative=True)
    metrics = pd.DataFrame(
        {
            "Metric": [
                "Total return",
                "Annualized return",
                "Annualized volatility",
                "Annualized Sharpe",
                "Maximum drawdown",
            ],
            "Value": [
                _total_return,
                _annual_return,
                _annual_volatility,
                _sharpe,
                _maximum_drawdown,
            ],
        }
    )
    mo.show_code(position="above")
    return (metrics,)


@app.cell(hide_code=False)
def _(metrics, mo):
    _metrics_display = metrics.astype({"Value": "object"})
    _metrics_display.loc[:2, "Value"] = _metrics_display.loc[:2, "Value"].map(
        "{:.2%}".format
    )
    _metrics_display.loc[3, "Value"] = "{:.2f}".format(
        _metrics_display.loc[3, "Value"]
    )
    _metrics_display.loc[4, "Value"] = "{:.2%}".format(
        _metrics_display.loc[4, "Value"]
    )
    mo.show_code(
        mo.ui.table(_metrics_display, label="Summary"), position="above"
    )
    return


@app.cell(hide_code=False)
def _(mo, px, rebased_prices):
    _wealth_frame = (
        rebased_prices.rename("Value").rename_axis("date").reset_index()
    )
    _wealth_chart = px.line(
        _wealth_frame,
        x="date",
        y="Value",
        title="Growth of 100 USD",
        labels={"date": "Date", "Value": "Normalized value"},
    )
    _wealth_chart.update_yaxes(tickprefix="$")
    mo.show_code(mo.ui.plotly(_wealth_chart), position="above")
    return


@app.cell(hide_code=False)
def _(drawdown, mo, px):
    _drawdown_frame = (
        drawdown.rename("Drawdown").rename_axis("date").reset_index()
    )
    _drawdown_chart = px.area(
        _drawdown_frame,
        x="date",
        y="Drawdown",
        title="Drawdown from the running peak",
        labels={"date": "Date", "Drawdown": "Drawdown"},
    )
    _drawdown_chart.update_yaxes(tickformat=".0%")
    mo.show_code(mo.ui.plotly(_drawdown_chart), position="above")
    return


@app.cell(hide_code=False)
def _(mo):
    mo.md("""
    ## Use it in your analysis

    Copy these calls and apply them to your own adjusted-price `Series`:

    ```python
    import quantkit as qnt

    rebased_prices = pd.Series(
        qnt.core.rebase(prices, base=100), index=prices.index, name=prices.name
    )
    asset_returns = qnt.core.returns(prices).iloc[1:]
    initial_return = pd.Series([0.0], index=prices.index[:1])
    wealth_returns = pd.concat([initial_return, asset_returns])
    wealth = qnt.core.cum_returns(wealth_returns, first_price=100)
    drawdown = qnt.expanding.drawdown(wealth, relative=True)

    annual_return = qnt.stats.annualized_return(
        asset_returns, periods_per_year=252
    )
    annual_volatility = qnt.stats.volatility(
        asset_returns, factor=252**0.5, ddof=0
    )
    sharpe = qnt.stats.sharpe_ratio(
        asset_returns, risk_free=0.0, factor=252**0.5
    )
    ```

    Here the risk-free rate is 0 per period. `sharpe_ratio` uses population
    standard deviation (`ddof=0`), and its factor directly multiplies the
    ratio, so we use `252**0.5` to express it annually. These metrics describe
    the sample; they do not predict returns.
    """)
    return


if __name__ == "__main__":
    app.run()
