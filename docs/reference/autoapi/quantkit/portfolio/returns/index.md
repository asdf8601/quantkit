<a id="module-quantkit.portfolio.returns"></a>

<a id="quantkit-portfolio-returns"></a>

# quantkit.portfolio.returns

Weighted portfolio return calculations.

<!-- !! processed by numpydoc !! -->

<a id="functions"></a>

## Functions

| [`portfolio_returns`](#quantkit.portfolio.returns.portfolio_returns)(asset_returns, beginning_weights)   | Aggregate asset returns using weights at each interval's beginning.   |
|----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| [`expected_return`](#quantkit.portfolio.returns.expected_return)(expected_returns, weights)              | Aggregate per-asset expected returns using asset weights.             |

<a id="module-contents"></a>

## Module Contents

<a id="quantkit.portfolio.returns.portfolio_returns"></a>

### quantkit.portfolio.returns.portfolio_returns(asset_returns, beginning_weights)

Aggregate asset returns using weights at each interval’s beginning.

The caller supplies weights valid at the start of every return interval. This function does not shift weights, normalize them, or infer a cash position; include cash as an explicit asset when required.

* **Parameters:**
  **asset_returns** — numpy.ndarray, pandas.Series, or pandas.DataFrame
  : Per-asset returns. Matrix rows are return intervals and columns are assets.

  **beginning_weights** — numpy.ndarray, pandas.Series, or pandas.DataFrame
  : Asset weights at the beginning of each corresponding interval. An asset vector broadcasts over matrix `asset_returns`.
* **Returns:**
  float, numpy.ndarray, or pandas.Series
  : The weighted return for each interval, retaining the shape and time labels of `asset_returns`.

### References

<a id="r457aa24d440e-1"></a>

[1]

[https://www.cvxportfolio.com/en/1.5.0/_static/cvx_portfolio.pdf](https://www.cvxportfolio.com/en/1.5.0/_static/cvx_portfolio.pdf)

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.returns.expected_return"></a>

### quantkit.portfolio.returns.expected_return(expected_returns, weights)

Aggregate per-asset expected returns using asset weights.

Expected returns and weights must use the same return frequency. This function does not annualize inputs, normalize weights, or infer cash.

* **Parameters:**
  **expected_returns** — numpy.ndarray, pandas.Series, or pandas.DataFrame
  : Per-asset expected returns. Matrix rows may represent separate intervals or scenarios.

  **weights** — numpy.ndarray, pandas.Series, or pandas.DataFrame
  : Asset weights for the corresponding expected returns. An asset vector broadcasts over matrix `expected_returns`.
* **Returns:**
  float, numpy.ndarray, or pandas.Series
  : Weighted expected return values, retaining the shape and time labels of `expected_returns`.

### References

<a id="r71a8b7913ae8-1"></a>

[1]

[https://www.cvxportfolio.com/en/1.5.0/_static/cvx_portfolio.pdf](https://www.cvxportfolio.com/en/1.5.0/_static/cvx_portfolio.pdf)

<!-- !! processed by numpydoc !! -->
