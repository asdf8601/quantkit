"""Interactive tour of risk, drawdown, and benchmark comparison."""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=False)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    from _data import load_prices

    import quantkit as qnt
    import quantkit.rolling  # Load the submodule for qnt.rolling.

    mo.show_code(position="above")
    return (
        load_prices,
        mo,
        np,
        pd,
        px,
        qnt,
        quantkit,
    )


@app.cell(hide_code=False)
def _(mo):
    mo.md(r"""
    # 02 · Risk and benchmark comparison

    In about 15 minutes, this notebook moves from adjusted prices to daily
    returns, measures historical risk, and compares an asset with a benchmark.
    The data is a local, immutable snapshot: running the notebook downloads
    nothing.

    We use **252 sessions per year**, a **0%** risk-free return, and a daily
    minimum acceptable return (MAR) of **0%**. Annualized figures change the
    scale; they are not predictions.
    """)
    return


@app.cell(hide_code=False)
def _(load_prices, mo):
    market_data, metadata = load_prices()
    tickers = sorted(market_data["ticker"].unique())
    years = sorted(market_data["date"].dt.year.unique())
    price_panel = market_data.pivot(
        index="date", columns="ticker", values="adjusted_close"
    ).sort_index()
    mo.show_code(position="above")
    return metadata, price_panel, tickers, years


@app.cell(hide_code=False)
def _(mo, tickers, years):
    asset_control = mo.ui.dropdown(
        tickers, value="QQQ", label="Asset to analyze"
    )
    benchmark_control = mo.ui.dropdown(tickers, value="SPY", label="Benchmark")
    years_control = mo.ui.range_slider(
        start=min(years),
        stop=max(years),
        step=1,
        value=[min(years), max(years)],
        show_value=True,
        label="Years included",
    )
    window_control = mo.ui.slider(
        start=21,
        stop=252,
        step=21,
        value=63,
        show_value=True,
        label="Volatility window (sessions)",
    )
    confidence_control = mo.ui.slider(
        start=0.90,
        stop=0.99,
        step=0.01,
        value=0.95,
        show_value=True,
        label="Historical confidence level",
    )
    mo.show_code(
        mo.vstack(
            [
                mo.hstack([asset_control, benchmark_control]),
                years_control,
                window_control,
                confidence_control,
            ]
        ),
        position="above",
    )
    return (
        asset_control,
        benchmark_control,
        confidence_control,
        window_control,
        years_control,
    )


@app.cell(hide_code=False)
def _(
    asset_control, benchmark_control, mo, pd, price_panel, qnt, years_control
):
    asset = asset_control.value
    benchmark = benchmark_control.value
    first_year, last_year = years_control.value
    _selected_tickers = [asset] if asset == benchmark else [asset, benchmark]
    selected_prices = price_panel.loc[
        f"{first_year}-01-01" : f"{last_year}-12-31", _selected_tickers
    ]

    # Compute separately so the asset and benchmark may be the same ticker.
    _asset_return_series = qnt.core.returns(selected_prices[asset])
    _benchmark_return_series = qnt.core.returns(selected_prices[benchmark])
    _paired = pd.DataFrame(
        {"asset": _asset_return_series, "benchmark": _benchmark_return_series}
    ).dropna()
    asset_returns = _paired[["asset"]].rename(columns={"asset": asset})
    benchmark_returns = _paired["benchmark"].rename(benchmark)
    mo.show_code(position="above")
    return asset, asset_returns, benchmark_returns, selected_prices


@app.cell(hide_code=False)
def _(metadata, mo):
    mo.md(
        f"""
        Data: `{metadata["source"]}`, daily frequency, closing prices adjusted
        for dividends and splits. Local coverage:
        {metadata["first_date"]}–{metadata["last_date"]}.
        """
    )
    return


@app.cell(hide_code=False)
def _(asset, asset_returns, mo, pd, qnt, selected_prices):
    if len(asset_returns) == 0:
        mo.callout(
            "There are not two valid prices in the selected interval; "
            "expand the year range.",
            kind="warn",
        )
        wealth = asset_returns[asset].copy()
    else:
        # Wealth starts before the first return: 1.0 belongs to the history
        # so the first loss is included in drawdown.
        _baseline = asset_returns[asset].iloc[:0].copy()
        _baseline.loc[selected_prices.index[0]] = 1.0
        wealth = pd.concat(
            [
                _baseline,
                qnt.core.cum_returns(asset_returns[asset], first_price=1.0),
            ]
        )
    mo.show_code(position="above")
    return (wealth,)


@app.cell(hide_code=False)
def _(
    asset,
    asset_returns,
    benchmark_returns,
    confidence_control,
    mo,
    np,
    pd,
    qnt,
    wealth,
    window_control,
):
    annualization = np.sqrt(252)
    confidence = confidence_control.value
    window = int(window_control.value)
    _enough_observations = len(asset_returns) >= 2

    if _enough_observations:
        _asset_series = asset_returns[asset]
        annual_volatility = qnt.stats.volatility(
            _asset_series, factor=annualization
        )
        beta = qnt.stats.beta(_asset_series, benchmark_returns)
        correlation = qnt.stats.correlation(_asset_series, benchmark_returns)
        tracking_error = qnt.stats.tracking_error(
            _asset_series, benchmark_returns, factor=annualization
        )
        information_ratio = qnt.stats.information_ratio(
            _asset_series, benchmark_returns, factor=annualization
        )
        sortino = qnt.stats.sortino_ratio(
            _asset_series, mar=0.0, factor=annualization
        )
        value_at_risk = qnt.stats.value_at_risk(
            _asset_series, confidence=confidence
        )
        expected_shortfall = qnt.portfolio.tail_risk.expected_shortfall(
            _asset_series, confidence=confidence
        )
        drawdown = qnt.expanding.drawdown(wealth)
        max_drawdown = qnt.stats.max_drawdown(wealth)
        recovery = qnt.stats.max_drawdown_recovery(wealth)
        cdar = qnt.portfolio.tail_risk.conditional_drawdown_at_risk(
            wealth, confidence=confidence
        )
    else:
        annual_volatility = beta = correlation = tracking_error = np.nan
        information_ratio = sortino = value_at_risk = expected_shortfall = (
            np.nan
        )
        drawdown = wealth.copy()
        max_drawdown = cdar = np.nan
        recovery = pd.NaT

    if len(asset_returns) >= window:
        rolling_volatility = qnt.rolling.volatility(
            asset_returns[asset],
            window=window,
            min_periods=window,
            factor=annualization,
        )
    else:
        rolling_volatility = asset_returns[asset].copy() * np.nan

    results = {
        "observations": len(asset_returns),
        "annual_volatility": annual_volatility,
        "beta": beta,
        "correlation": correlation,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "sortino": sortino,
        "value_at_risk_daily_return": value_at_risk,
        "expected_shortfall_daily_loss": expected_shortfall,
        "max_drawdown": max_drawdown,
        "cdar": cdar,
        "recovery": recovery,
    }
    mo.show_code(position="above")
    return drawdown, results, rolling_volatility


@app.cell(hide_code=False)
def _(asset_returns, mo, pd, results, window_control):
    def _percent(value):
        return "—" if pd.isna(value) else f"{value:.2%}"

    def _ratio(value):
        return "—" if pd.isna(value) else f"{value:.2f}"

    _recovery = results["recovery"]
    _recovery_text = (
        "Not recovered" if pd.isna(_recovery) else str(_recovery.date())
    )
    metrics_table = pd.DataFrame(
        [
            ("Observations", f"{results['observations']}", "days"),
            (
                "Annual volatility",
                _percent(results["annual_volatility"]),
                "% annual",
            ),
            ("Beta", _ratio(results["beta"]), "ratio"),
            (
                "Correlation",
                _ratio(results["correlation"]),
                "ratio [-1, 1]",
            ),
            (
                "Tracking error",
                _percent(results["tracking_error"]),
                "% annual",
            ),
            (
                "Information ratio",
                _ratio(results["information_ratio"]),
                "annual ratio",
            ),
            ("Sortino (MAR 0%)", _ratio(results["sortino"]), "annual ratio"),
            (
                "Daily VaR (signed return)",
                _percent(results["value_at_risk_daily_return"]),
                "% daily",
            ),
            (
                "Daily ES loss",
                _percent(results["expected_shortfall_daily_loss"]),
                "% daily",
            ),
            ("Maximum drawdown", _percent(results["max_drawdown"]), "%"),
            ("CDaR", _percent(results["cdar"]), "% drawdown"),
            (
                "Maximum drawdown recovery",
                _recovery_text,
                "date",
            ),
        ],
        columns=["Metric", "Value", "Unit"],
    )
    if len(asset_returns) < int(window_control.value):
        mo.callout(
            "The series is shorter than the selected window: rolling "
            "volatility is empty.",
            kind="warn",
        )
    mo.show_code(
        mo.ui.table(metrics_table, pagination=False, selection=None),
        position="above",
    )
    return (metrics_table,)


@app.cell(hide_code=False)
def _(asset, drawdown, mo, px, rolling_volatility, wealth):
    _wealth_frame = wealth.rename("Cumulative wealth").reset_index()
    _drawdown_frame = (drawdown * 100).rename("Drawdown (%)").reset_index()
    _rolling_frame = (
        (rolling_volatility * 100)
        .rename("Annual volatility (%)")
        .reset_index()
    )
    wealth_figure = px.line(
        _wealth_frame,
        x="date",
        y="Cumulative wealth",
        title=f"{asset}: cumulative wealth (initial base = 1)",
        labels={"date": "Date"},
    )
    drawdown_figure = px.area(
        _drawdown_frame,
        x="date",
        y="Drawdown (%)",
        title="Drawdown from the running peak",
        labels={"date": "Date"},
    )
    rolling_figure = px.line(
        _rolling_frame,
        x="date",
        y="Annual volatility (%)",
        title="Annualized rolling volatility",
        labels={"date": "Date"},
    )
    mo.show_code(
        mo.vstack(
            [
                mo.ui.plotly(wealth_figure),
                mo.ui.plotly(drawdown_figure),
                mo.ui.plotly(rolling_figure),
            ]
        ),
        position="above",
    )
    return


@app.cell(hide_code=False)
def _(mo):
    mo.md(r"""
    ## How to read it

    - **Beta**, correlation, tracking error, and information ratio compare
      coincident daily returns with the benchmark. They are descriptive sample
      estimates, not forecasts of covariances or future returns.
    - VaR and ES summarize the tail of **daily returns**. The **VaR** from
      `qnt.stats.value_at_risk` is a signed return: a negative VaR means
      a historical loss of that size. ES uses the loss convention (positive
      usually means a loss, though an all-gains tail can be negative).
    - Drawdown is calculated from cumulative wealth with an initial base of
      1.0. CDaR averages the deepest drawdown tail at the selected confidence
      level over the entire selected path; “Not recovered” means the series
      did not return to its previous peak before the final observation.
    """)
    return


if __name__ == "__main__":
    app.run()
