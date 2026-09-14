<a id="module-quantkit.portfolio.weights"></a>

<a id="quantkit-portfolio-weights"></a>

# quantkit.portfolio.weights

Portfolio weights and concentration measures.

<!-- !! processed by numpydoc !! -->

<a id="functions"></a>

## Functions

| [`_row_denominator`](#quantkit.portfolio.weights._row_denominator)(values, denominator)                 | Shape a per-row denominator to broadcast across assets.        |
|---------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| [`nav_weights`](#quantkit.portfolio.weights.nav_weights)(position_values, nav)                          | Return signed position values normalized by positive NAV.      |
| [`gross_weights`](#quantkit.portfolio.weights.gross_weights)(position_values)                           | Return absolute position values normalized by gross exposure.  |
| [`concentration`](#quantkit.portfolio.weights.concentration)(position_values)                           | Return the Herfindahl concentration of gross position weights. |
| [`effective_number_of_assets`](#quantkit.portfolio.weights.effective_number_of_assets)(position_values) | Return the inverse Herfindahl effective number of assets.      |

<a id="module-contents"></a>

## Module Contents

<a id="quantkit.portfolio.weights._row_denominator"></a>

### quantkit.portfolio.weights.\_row_denominator(values, denominator)

Shape a per-row denominator to broadcast across assets.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.weights.nav_weights"></a>

### quantkit.portfolio.weights.nav_weights(position_values, nav)

Return signed position values normalized by positive NAV.

* **Parameters:**
  **position_values** — numpy.ndarray or pandas.Series or pandas.DataFrame
  : Signed monetary position values, shaped as assets or time by assets.

  **nav** — scalar, array-like, or pandas.Series
  : Portfolio net asset value. Matrix inputs accept a scalar or one NAV value per time row; a pandas Series aligns to a DataFrame’s time index.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Signed dimensionless weights with the input’s container metadata. Rows with missing or nonpositive NAV contain NaN.

### Notes

NAV weights may include short positions, which have negative weights. Cash is excluded unless it is included in `position_values` as an asset.

### References

<a id="r1e912c9a75c2-1"></a>

[1]

U.S. Securities and Exchange Commission, “Net asset value”, [https://www.investor.gov/introduction-investing/investing-basics/glossary/net-asset-value](https://www.investor.gov/introduction-investing/investing-basics/glossary/net-asset-value)

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.weights.gross_weights"></a>

### quantkit.portfolio.weights.gross_weights(position_values)

Return absolute position values normalized by gross exposure.

* **Parameters:**
  **position_values** — numpy.ndarray or pandas.Series or pandas.DataFrame
  : Signed monetary position values, shaped as assets or time by assets.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Nonnegative dimensionless weights with the input’s container metadata. An all-zero or missing gross exposure produces NaN.

### Notes

Gross exposure is the sum of absolute position values and excludes cash. The returned weights therefore sum to one when gross exposure is positive.

### References

<a id="r2bcc92312edf-1"></a>

[1]

Cvxportfolio, “Leverage limit”, [https://www.cvxportfolio.com/en/1.5.0/constraints.html](https://www.cvxportfolio.com/en/1.5.0/constraints.html)

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.weights.concentration"></a>

### quantkit.portfolio.weights.concentration(position_values)

Return the Herfindahl concentration of gross position weights.

* **Parameters:**
  **position_values** — numpy.ndarray or pandas.Series or pandas.DataFrame
  : Signed monetary position values, shaped as assets or time by assets.
* **Returns:**
  float or numpy.ndarray or pandas.Series
  : The dimensionless sum of squared gross weights. Matrix inputs retain their time axis. Undefined gross exposures produce NaN.

### Notes

This measure uses absolute position magnitudes, so long and short positions contribute equally for equal absolute values.

### References

<a id="r5d51dab4a8ee-1"></a>

[1]

Cvxportfolio, “Leverage limit”, [https://www.cvxportfolio.com/en/1.5.0/constraints.html](https://www.cvxportfolio.com/en/1.5.0/constraints.html)

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.weights.effective_number_of_assets"></a>

### quantkit.portfolio.weights.effective_number_of_assets(position_values)

Return the inverse Herfindahl effective number of assets.

* **Parameters:**
  **position_values** — numpy.ndarray or pandas.Series or pandas.DataFrame
  : Signed monetary position values, shaped as assets or time by assets.
* **Returns:**
  float or numpy.ndarray or pandas.Series
  : `1 / concentration(position_values)`. Matrix inputs retain their time axis. Undefined gross exposures produce NaN.

### Notes

Equal gross exposure across `n` assets gives an effective number of assets equal to `n`.

### References

<a id="r9e757bc20095-1"></a>

[1]

Cvxportfolio, “Leverage limit”, [https://www.cvxportfolio.com/en/1.5.0/constraints.html](https://www.cvxportfolio.com/en/1.5.0/constraints.html)

<!-- !! processed by numpydoc !! -->
