<a id="module-quantkit.rolling"></a>

<a id="quantkit-rolling"></a>

# quantkit.rolling

<a id="functions"></a>

## Functions

| [`volatility`](#quantkit.rolling.volatility)(returns\[, window, min_periods, ddof, factor\])   | Volatility calculation.   |
|------------------------------------------------------------------------------------------------|---------------------------|

<a id="module-contents"></a>

## Module Contents

<a id="quantkit.rolling.volatility"></a>

### quantkit.rolling.volatility(returns, window=BYEAR, min_periods=2, ddof=1, factor=None)

Volatility calculation.

* **Parameters:**
  **returns** — array-like
  : Returns series.

  **window** — int or pandas.offset
  : Number of elements to take in account on each iteration.

  **ddof** — int, optional
  : Degree of freedom used in the std calculation.

  **factor** — float, optional
  : Annualization factor.
* **Returns:**
  **out** — array-like
  : Rolling volatility series.

### References

Bacon, C.Practical Portfolio Performance Measurement and Attribution. Wiley. 2004. p. 27

### Examples

```pycon
>>> ret = np.array([0.1, 0.1, 0.1])
>>> volatility(ret, 2)
array([np.nan, 0., 0.])
```

<!-- !! processed by numpydoc !! -->
