<a id="module-quantkit.portfolio.valuation"></a>

<a id="quantkit-portfolio-valuation"></a>

# quantkit.portfolio.valuation

Portfolio position and net asset value calculations.

The functions in this module value signed positions and reduce them to portfolio-level amounts while preserving the input container’s metadata.

<!-- !! processed by numpydoc !! -->

<a id="functions"></a>

## Functions

| [`position_values`](#quantkit.portfolio.valuation.position_values)(quantities, prices)                     | Value signed positions at nonnegative prices.                     |
|------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| [`gross_asset_value`](#quantkit.portfolio.valuation.gross_asset_value)(position_values\[, cash\])          | Calculate gross asset value from positions and cash.              |
| [`net_asset_value`](#quantkit.portfolio.valuation.net_asset_value)(position_values\[, cash, liabilities\]) | Calculate net asset value from positions, cash, and liabilities.  |
| [`_scalar_or_vector`](#quantkit.portfolio.valuation._scalar_or_vector)(values, name)                       | Validate a scalar or nonempty one-dimensional real numeric value. |
| [`nav_per_share`](#quantkit.portfolio.valuation.nav_per_share)(nav, shares)                                | Calculate NAV per positive outstanding share.                     |

<a id="module-contents"></a>

## Module Contents

<a id="quantkit.portfolio.valuation.position_values"></a>

### quantkit.portfolio.valuation.position_values(quantities, prices)

Value signed positions at nonnegative prices.

* **Parameters:**
  **quantities** — numpy.ndarray, pandas.Series, or pandas.DataFrame
  : Signed units held in each asset. A matrix has time on rows and assets on columns.

  **prices** — numpy.ndarray, pandas.Series, or pandas.DataFrame
  : Nonnegative monetary price for each asset. A vector may be broadcast across the time axis of matrix `quantities`.
* **Returns:**
  numpy.ndarray, pandas.Series, or pandas.DataFrame
  : Position values with the type and labels of `quantities`.
* **Raises:**
  ValueError
  : If prices are negative or the inputs have incompatible asset labels or shapes.

  TypeError
  : If an input is not real numeric asset data.

### References

<a id="r2b835767f8fe-1"></a>

[1]

Boyd et al., [Multi-Period Trading via Convex Optimization](https://www.cvxportfolio.com/en/1.5.0/_static/cvx_portfolio.pdf).

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.valuation.gross_asset_value"></a>

### quantkit.portfolio.valuation.gross_asset_value(position_values, cash=0.0)

Calculate gross asset value from positions and cash.

Gross asset value is the sum of positive position values plus positive cash. Negative cash and short positions do not reduce gross asset value.

* **Parameters:**
  **position_values** — numpy.ndarray, pandas.Series, or pandas.DataFrame
  : Signed monetary value of each position.

  **cash** — float, numpy.ndarray, or pandas.Series, default 0.0
  : Cash balance. For matrix positions, it may be a scalar or one value per time row.
* **Returns:**
  float, numpy.ndarray, or pandas.Series
  : Gross asset value. DataFrame inputs return a time-indexed Series.

### References

<a id="r94b8592fbdfe-1"></a>

[1]

[Investor.gov: Net asset value](https://www.investor.gov/introduction-investing/investing-basics/glossary/net-asset-value).

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.valuation.net_asset_value"></a>

### quantkit.portfolio.valuation.net_asset_value(position_values, cash=0.0, liabilities=0.0)

Calculate net asset value from positions, cash, and liabilities.

Additional liabilities are deducted after including signed position values and cash. Short positions are already represented in `position_values`.

* **Parameters:**
  **position_values** — numpy.ndarray, pandas.Series, or pandas.DataFrame
  : Signed monetary value of each position.

  **cash** — float, numpy.ndarray, or pandas.Series, default 0.0
  : Cash balance. For matrix positions, it may be a scalar or one value per time row.

  **liabilities** — float, numpy.ndarray, or pandas.Series, default 0.0
  : Nonnegative additional liabilities. For matrix positions, it may be a scalar or one value per time row.
* **Returns:**
  float, numpy.ndarray, or pandas.Series
  : Net asset value. DataFrame inputs return a time-indexed Series.
* **Raises:**
  ValueError
  : If an additional liability is negative.

### References

<a id="r3bbd8742c762-1"></a>

[1]

[Investor.gov: Net asset value](https://www.investor.gov/introduction-investing/investing-basics/glossary/net-asset-value).

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.valuation._scalar_or_vector"></a>

### quantkit.portfolio.valuation.\_scalar_or_vector(values, name)

Validate a scalar or nonempty one-dimensional real numeric value.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.valuation.nav_per_share"></a>

### quantkit.portfolio.valuation.nav_per_share(nav, shares)

Calculate NAV per positive outstanding share.

* **Parameters:**
  **nav** — float, numpy.ndarray, or pandas.Series
  : Scalar or one-dimensional total net asset value. Negative values are valid and missing values propagate.

  **shares** — float, numpy.ndarray, or pandas.Series
  : Positive, finite outstanding shares. A Series aligns to a Series `nav` by exact index label sets; a scalar broadcasts over NAV.
* **Returns:**
  float, numpy.ndarray, or pandas.Series
  : NAV divided by shares, preserving a Series index and name.
* **Raises:**
  ValueError
  : If inputs are not scalar or one-dimensional, labels or shapes do not match, or a finite share count is not positive.

  TypeError
  : If an input is not real numeric data.

### References

<a id="r9d1188dabeab-1"></a>

[1]

[Investor.gov: Net asset value](https://www.investor.gov/introduction-investing/investing-basics/glossary/net-asset-value).

<!-- !! processed by numpydoc !! -->
