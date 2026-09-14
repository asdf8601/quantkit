<a id="module-quantkit.portfolio.exposure"></a>

<a id="quantkit-portfolio-exposure"></a>

# quantkit.portfolio.exposure

Exposure measures for signed portfolio position values.

<!-- !! processed by numpydoc !! -->

<a id="functions"></a>

## Functions

| [`long_value`](#quantkit.portfolio.exposure.long_value)(position_values)              | Return the value of long positions.                           |
|---------------------------------------------------------------------------------------|---------------------------------------------------------------|
| [`short_value`](#quantkit.portfolio.exposure.short_value)(position_values)            | Return the positive magnitude of short positions.             |
| [`gross_exposure`](#quantkit.portfolio.exposure.gross_exposure)(position_values)      | Return gross exposure as the sum of absolute position values. |
| [`net_exposure`](#quantkit.portfolio.exposure.net_exposure)(position_values)          | Return net exposure as the signed sum of position values.     |
| [`gross_leverage`](#quantkit.portfolio.exposure.gross_leverage)(position_values, nav) | Return gross exposure divided by positive net asset value.    |

<a id="module-contents"></a>

## Module Contents

<a id="quantkit.portfolio.exposure.long_value"></a>

### quantkit.portfolio.exposure.long_value(position_values)

Return the value of long positions.

Long value is the sum of positive signed position values over the asset axis.  Cash is not included unless it is supplied as a position.

* **Parameters:**
  **position_values** — numpy.ndarray or pandas.Series or pandas.DataFrame
  : Signed monetary values, shaped as assets or time by assets.
* **Returns:**
  float or numpy.ndarray or pandas.Series
  : Long value in monetary units.  Matrix inputs retain their time axis.

### Notes

Missing values propagate through each asset-axis reduction.

### References

<a id="r04abb9500dac-1"></a>

[1]

Cvxportfolio, “Long and short positions”, [https://www.cvxportfolio.com/en/1.5.0/result.html](https://www.cvxportfolio.com/en/1.5.0/result.html)

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.exposure.short_value"></a>

### quantkit.portfolio.exposure.short_value(position_values)

Return the positive magnitude of short positions.

Short value is the sum of the magnitudes of negative signed position values over the asset axis.  It is reported as a positive monetary amount.

* **Parameters:**
  **position_values** — numpy.ndarray or pandas.Series or pandas.DataFrame
  : Signed monetary values, shaped as assets or time by assets.
* **Returns:**
  float or numpy.ndarray or pandas.Series
  : Positive short value in monetary units.  Matrix inputs retain their time axis.

### Notes

Missing values propagate through each asset-axis reduction.

### References

<a id="r95a7e430f6c5-1"></a>

[1]

Cvxportfolio, “Long and short positions”, [https://www.cvxportfolio.com/en/1.5.0/result.html](https://www.cvxportfolio.com/en/1.5.0/result.html)

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.exposure.gross_exposure"></a>

### quantkit.portfolio.exposure.gross_exposure(position_values)

Return gross exposure as the sum of absolute position values.

* **Parameters:**
  **position_values** — numpy.ndarray or pandas.Series or pandas.DataFrame
  : Signed monetary values, shaped as assets or time by assets.
* **Returns:**
  float or numpy.ndarray or pandas.Series
  : Gross exposure in monetary units, excluding separately held cash.

### Notes

Missing values propagate through each asset-axis reduction.

### References

<a id="rb8815ec53ccb-1"></a>

[1]

Cvxportfolio, “Leverage limit”, [https://www.cvxportfolio.com/en/1.5.0/constraints.html](https://www.cvxportfolio.com/en/1.5.0/constraints.html)

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.exposure.net_exposure"></a>

### quantkit.portfolio.exposure.net_exposure(position_values)

Return net exposure as the signed sum of position values.

* **Parameters:**
  **position_values** — numpy.ndarray or pandas.Series or pandas.DataFrame
  : Signed monetary values, shaped as assets or time by assets.
* **Returns:**
  float or numpy.ndarray or pandas.Series
  : Net exposure in monetary units, excluding separately held cash.

### Notes

Missing values propagate through each asset-axis reduction.

### References

<a id="r0b549925b15f-1"></a>

[1]

Boyd et al., “Multi-Period Trading via Convex Optimization”, [https://www.cvxportfolio.com/en/1.5.0/_static/cvx_portfolio.pdf](https://www.cvxportfolio.com/en/1.5.0/_static/cvx_portfolio.pdf)

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.exposure.gross_leverage"></a>

### quantkit.portfolio.exposure.gross_leverage(position_values, nav)

Return gross exposure divided by positive net asset value.

* **Parameters:**
  **position_values** — numpy.ndarray or pandas.Series or pandas.DataFrame
  : Signed monetary values, shaped as assets or time by assets.

  **nav** — scalar, array-like, or pandas.Series
  : Portfolio net asset value.  Matrix inputs accept a scalar or one NAV value per time row; a pandas Series aligns to a DataFrame’s index.
* **Returns:**
  float or numpy.ndarray or pandas.Series
  : Dimensionless gross leverage.  NaN is returned where NAV is missing or nonpositive.

### References

<a id="r33f77df7a2c1-1"></a>

[1]

Cvxportfolio, “Leverage limit”, [https://www.cvxportfolio.com/en/1.5.0/constraints.html](https://www.cvxportfolio.com/en/1.5.0/constraints.html)

<!-- !! processed by numpydoc !! -->
