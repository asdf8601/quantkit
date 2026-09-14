<a id="quick-start"></a>

# Quick Start

This walkthrough turns a small price history into returns, summary statistics and a two-asset portfolio. The data is synthetic and deliberately small so you can follow every calculation. Run the Python examples in order in one session.

<a id="install-quantkit"></a>

## Install quantkit

From an existing uv project:

```bash
uv add git+https://github.com/asdf8601/quantkit
```

For other installation options, see [Install](install.md).

<a id="create-a-price-history"></a>

## Create a price history

Use a pandas DataFrame with dates in rows and assets in columns. This example assumes one valuation per business day, in a common currency, with no external cash flows or distributions.

```pycon
>>> import numpy as np
>>> import pandas as pd
>>> import quantkit as qnt
>>> prices = pd.DataFrame(
...     {"stocks": [100.0, 110.0, 99.0, 108.9],
...      "bonds": [100.0, 102.0, 101.0, 103.0]},
...     index=pd.date_range("2026-01-05", periods=4, freq="B"),
... )
```

<a id="calculate-period-returns"></a>

## Calculate period returns

`core.returns` calculates the change relative to the previous price. Returns are fractions: `0.10` means a gain of 10%, and `-0.10` means a loss of 10%. A DataFrame input retains its index and columns.

The first row contains NaN because there is no preceding price. Remove only that initial row for the subsequent calculations:

```pycon
>>> all_returns = qnt.core.returns(prices)
>>> asset_returns = all_returns.iloc[1:]
>>> print(asset_returns.round(4).to_string())
            stocks   bonds
2026-01-06     0.1  0.0200
2026-01-07    -0.1 -0.0098
2026-01-08     0.1  0.0198
```

With real data, investigate any other missing observations rather than silently dropping dates. The examples below use the full-precision values; rounding is only for display.

<a id="summarize-performance-and-risk"></a>

## Summarize performance and risk

Pass **prices** to `total_returns` and `max_drawdown`, and **period returns** to `volatility`. For a DataFrame, these functions calculate one result per asset and return a Series indexed by the asset names.

```pycon
>>> summary = pd.DataFrame({
...     "total_return": qnt.stats.total_returns(prices),
...     "volatility": qnt.stats.volatility(asset_returns, factor=1),
...     "max_drawdown": qnt.stats.max_drawdown(prices),
... })
>>> print(summary.round(4).to_string())
        total_return  volatility  max_drawdown
stocks         0.089      0.1155       -0.1000
bonds          0.030      0.0172       -0.0098
```

For stocks, the total return is 8.9%. Volatility here is the sample standard deviation of the period returns (`ddof=1`), without annualization. Maximum drawdown is reported as a negative return: `-0.10` is a 10% decline from a previous peak.

The Sharpe ratio takes returns and a risk-free return for the **same interval**. Set `factor=1` to keep this example on its original time scale:

```pycon
>>> qnt.stats.sharpe_ratio(
...     asset_returns, risk_free=0.0, factor=1,
... ).round(4).to_dict()
{'stocks': 0.3536, 'bonds': 0.7141}
```

Quantkit does not infer annualization from the date index. For daily returns, you can pass `factor=np.sqrt(periods_per_year)` to volatility and Sharpe, using your chosen annualization convention. Sharpe’s default factor is `sqrt(261)`, while volatility’s default is unscaled; specifying the factor makes the convention explicit.

<a id="track-cumulative-growth"></a>

## Track cumulative growth

`core.cum_returns` compounds returns. With `first_price=100`, it gives a wealth path starting from 100 units of capital. Include an initial zero return to show the starting value before the first observed return:

```pycon
>>> stock_returns = asset_returns["stocks"]
>>> initial_return = pd.Series([0.0], index=prices.index[:1])
>>> stock_wealth = qnt.core.cum_returns(
...     pd.concat([initial_return, stock_returns]), first_price=100.0,
... )
>>> stock_wealth.round(2).to_list()
[100.0, 110.0, 99.0, 108.9]
```

Without `first_price`, the function returns cumulative gains as fractions instead of capital values. Missing values are skipped during compounding, so validate your return history before using it to report performance.

<a id="build-a-two-asset-portfolio"></a>

## Build a two-asset portfolio

Use `portfolio_returns` to combine the assets with weights of 60% stocks and 40% bonds. A single Series of weights is applied to every return interval; its labels identify the DataFrame columns.

```pycon
>>> from quantkit.portfolio.returns import portfolio_returns
>>> weights = pd.Series({"stocks": 0.6, "bonds": 0.4})
>>> portfolio = portfolio_returns(asset_returns, weights)
>>> portfolio.round(4).to_list()
[0.068, -0.0639, 0.0679]
```

These are **beginning-of-interval weights**. Constant weights describe a portfolio rebalanced to that allocation each period, with transaction costs omitted here. The function does not shift or normalize weights, or add a cash position automatically.

Compound the portfolio returns, including its initial capital value:

```pycon
>>> portfolio_wealth = qnt.core.cum_returns(
...     pd.concat([initial_return, portfolio]), first_price=100.0,
... )
>>> portfolio_wealth.round(2).to_list()
[100.0, 106.8, 99.97, 106.76]
>>> portfolio_total = qnt.stats.total_returns(portfolio_wealth)
>>> print(f"Portfolio total return: {portfolio_total:.2%}")
Portfolio total return: 6.76%
```

For contributions and withdrawals, use the flow-adjusted return functions described in [Portfolio statistics](portfolios.md) rather than price changes in raw portfolio NAV.

<a id="use-numpy-arrays"></a>

## Use NumPy arrays

The same return calculation also accepts NumPy arrays. With a one-dimensional price array, the result is a one-dimensional return array:

```pycon
>>> price_array = prices["stocks"].to_numpy()
>>> qnt.core.returns(price_array).round(4).tolist()
[nan, 0.1, -0.1, 0.1]
```

For two-dimensional arrays, rows represent time and columns represent assets. NumPy inputs have no labels, so asset order must match when combining arrays.

<a id="next-steps"></a>

## Next steps

* [Portfolio statistics](portfolios.md): valuation, exposure, weights, cash flows and portfolio risk.
* [quantkit.core](autoapi/quantkit/core/index.md): return and cumulative-growth functions.
* [quantkit.stats](autoapi/quantkit/stats/index.md): performance and risk statistics.
* [API Reference](autoapi/index.md): the complete API reference.
