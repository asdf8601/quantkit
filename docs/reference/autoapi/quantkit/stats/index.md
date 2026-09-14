<a id="module-quantkit.stats"></a>

<a id="quantkit-stats"></a>

# quantkit.stats

Stats module.

All functions here receive a time series (1-dimension or 2-dimension) and reduce it to one value per column:

f(x_t) -> y

WHERE:

> f() : function x_t : time series y : one value per column

A 1-dimension input reduces to a scalar and a 2-dimension input to one value per column: a Series indexed by the columns for a DataFrame and a 1-dimension array for a numpy array, see [`quantkit.decorators.reduce_array_wrap()`](../decorators/index.md#quantkit.decorators.reduce_array_wrap). The value is a float, except for the functions that locate a point in time ([`max_drawdown_peak()`](#quantkit.stats.max_drawdown_peak), [`max_drawdown_valley()`](#quantkit.stats.max_drawdown_valley) and [`max_drawdown_recovery()`](#quantkit.stats.max_drawdown_recovery)), which give a position for numpy input and an index label for pandas input.

<!-- !! processed by numpydoc !! -->

<a id="functions"></a>

## Functions

| [`_empty_reduction`](#quantkit.stats._empty_reduction)(obj)                                 | Reduce an object with no observation to NaN, one value per column.   |
|---------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| [`_positions`](#quantkit.stats._positions)(index)                                           | Turn the output of the `*_valid_index` utils into integer positions. |
| [`total_returns`](#quantkit.stats.total_returns)(prices\[, factor, relative\])              | Calculate arithmetic total return.                                   |
| [`volatility`](#quantkit.stats.volatility)(returns\[, factor, ddof\])                       | Calculate the volatility of the arithmetic returns.                  |
| [`drawdown`](#quantkit.stats.drawdown)(prices\[, relative\])                                | Compute the drawdown statistic.                                      |
| [`max_drawdown`](#quantkit.stats.max_drawdown)(prices\[, relative\])                        | Calculate the Maximum Drawdown.                                      |
| [`sharpe_ratio`](#quantkit.stats.sharpe_ratio)(returns, risk_free\[, factor\])              | Calculate shape ratio.                                               |
| [`value_at_risk`](#quantkit.stats.value_at_risk)(returns\[, confidence\])                   | Calculate the historical value at risk (VaR).                        |
| [`max_drawup`](#quantkit.stats.max_drawup)(prices\[, relative\])                            | Calculate the Maximum Drawup.                                        |
| [`annualized_return`](#quantkit.stats.annualized_return)(returns\[, periods_per_year\])     | Calculate the geometric annualized return.                           |
| [`calmar_ratio`](#quantkit.stats.calmar_ratio)(returns\[, periods_per_year\])               | Calculate the Calmar ratio.                                          |
| [`_nan_divide`](#quantkit.stats._nan_divide)(numerator, denominator)                        | Divide returning NaN, never inf, where the denominator is zero.      |
| [`_masked_mean`](#quantkit.stats._masked_mean)(arr, mask, method)                           | Column-wise mean of `arr` over the periods selected by `mask`.       |
| [`average_gain`](#quantkit.stats.average_gain)(returns\[, method\])                         | Calculate the average gain.                                          |
| [`average_loss`](#quantkit.stats.average_loss)(returns\[, method\])                         | Calculate the average loss.                                          |
| [`gain_loss_ratio`](#quantkit.stats.gain_loss_ratio)(returns)                               | Calculate Morningstar's Gain/Loss Ratio.                             |
| [`up_period_percent`](#quantkit.stats.up_period_percent)(returns)                           | Calculate the fraction of periods with a return at or above zero.    |
| [`down_period_percent`](#quantkit.stats.down_period_percent)(returns)                       | Calculate the fraction of periods with a return below zero.          |
| [`_nanmean`](#quantkit.stats._nanmean)(arr)                                                 | Column-wise mean ignoring NaN.                                       |
| [`downside_deviation`](#quantkit.stats.downside_deviation)(returns\[, mar, factor\])        | Calculate the downside deviation of the returns.                     |
| [`upside_deviation`](#quantkit.stats.upside_deviation)(returns\[, mar, factor\])            | Calculate the upside deviation of the returns.                       |
| [`kappa`](#quantkit.stats.kappa)(returns\[, mar, order, factor\])                           | Calculate the Kappa ratio of Kaplan and Knowles.                     |
| [`omega_ratio`](#quantkit.stats.omega_ratio)(returns\[, mar\])                              | Calculate the Omega ratio of Shadwick and Keating.                   |
| [`sortino_ratio`](#quantkit.stats.sortino_ratio)(returns\[, mar, factor\])                  | Calculate the Sortino ratio.                                         |
| [`_columns`](#quantkit.stats._columns)(arr)                                                 | Return the columns of a 1D or 2D array as a list of 1D arrays.       |
| [`_reduce_columns`](#quantkit.stats._reduce_columns)(prices, values)                        | Wrap one value per column of `prices` like `reduce_array_wrap`.      |
| [`_apply_columns`](#quantkit.stats._apply_columns)(prices, func, \*args)                    | Apply `func(column, *args)` to every column of `prices`.             |
| [`_max_drawdown_span`](#quantkit.stats._max_drawdown_span)(arr)                             | Locate the maximum relative drawdown of a 1D array.                  |
| [`_max_drawdown_label`](#quantkit.stats._max_drawdown_label)(prices, item)                  | Report one item of `_max_drawdown_span` per column.                  |
| [`_max_drawdown_duration`](#quantkit.stats._max_drawdown_duration)(arr)                     | Periods from peak to valley; 0 without drawdown, NaN without data.   |
| [`_max_drawdown_recovery_duration`](#quantkit.stats._max_drawdown_recovery_duration)(arr)   | Periods from valley to recovery; NaN when there is none.             |
| [`_longest_drawdown_duration`](#quantkit.stats._longest_drawdown_duration)(arr)             | Longest run of valid observations strictly below the running max.    |
| [`_average_drawdown`](#quantkit.stats._average_drawdown)(arr, periods_per_year)             | Morningstar's average drawdown of a 1D array.                        |
| [`max_drawdown_peak`](#quantkit.stats.max_drawdown_peak)(prices)                            | Locate the peak the maximum drawdown fell from.                      |
| [`max_drawdown_valley`](#quantkit.stats.max_drawdown_valley)(prices)                        | Locate the valley (trough) of the maximum drawdown.                  |
| [`max_drawdown_recovery`](#quantkit.stats.max_drawdown_recovery)(prices)                    | Locate the recovery from the maximum drawdown.                       |
| [`max_drawdown_duration`](#quantkit.stats.max_drawdown_duration)(prices)                    | Calculate the duration of the maximum drawdown.                      |
| [`max_drawdown_recovery_duration`](#quantkit.stats.max_drawdown_recovery_duration)(prices)  | Calculate the recovery time of the maximum drawdown.                 |
| [`longest_drawdown_duration`](#quantkit.stats.longest_drawdown_duration)(prices)            | Calculate the longest time under water.                              |
| [`average_drawdown`](#quantkit.stats.average_drawdown)(prices\[, periods_per_year\])        | Calculate Morningstar's Average Drawdown.                            |
| [`_pairwise_complete`](#quantkit.stats._pairwise_complete)(col, bench)                      | Keep the rows where both `col` and `bench` are not NaN.              |
| [`_is_degenerate`](#quantkit.stats._is_degenerate)(arr)                                     | Tell whether a variance cannot be estimated from `arr`.              |
| [`_beta`](#quantkit.stats._beta)(col, bench)                                                | Slope of `col` on `bench`: cov(col, bench) / var(bench).             |
| [`_alpha`](#quantkit.stats._alpha)(col, bench, risk_free, factor)                           | Jensen's alpha: mean(col - rf) - beta \* mean(bench - rf).           |
| [`_correlation`](#quantkit.stats._correlation)(col, bench)                                  | Pearson correlation, NaN when either side has no dispersion.         |
| [`_r_squared`](#quantkit.stats._r_squared)(col, bench)                                      | Square of the Pearson correlation.                                   |
| [`_bull_beta`](#quantkit.stats._bull_beta)(col, bench)                                      | Beta on the rows where the benchmark went up.                        |
| [`_bear_beta`](#quantkit.stats._bear_beta)(col, bench)                                      | Beta on the rows where the benchmark went down.                      |
| [`_reduce_pairwise`](#quantkit.stats._reduce_pairwise)(returns, benchmark, func)            | Apply `func(col, bench)` to every column on its complete rows.       |
| [`beta`](#quantkit.stats.beta)(returns, benchmark)                                          | Compute the beta of `returns` against `benchmark`.                   |
| [`alpha`](#quantkit.stats.alpha)(returns, benchmark\[, risk_free, factor\])                 | Compute Jensen's alpha of `returns` against `benchmark`.             |
| [`correlation`](#quantkit.stats.correlation)(returns, benchmark)                            | Compute the Pearson correlation of `returns` with `benchmark`.       |
| [`r_squared`](#quantkit.stats.r_squared)(returns, benchmark)                                | Compute the coefficient of determination against `benchmark`.        |
| [`bull_beta`](#quantkit.stats.bull_beta)(returns, benchmark)                                | Compute the beta over the periods where the benchmark went up.       |
| [`bear_beta`](#quantkit.stats.bear_beta)(returns, benchmark)                                | Compute the beta over the periods where the benchmark went down.     |
| [`_treynor`](#quantkit.stats._treynor)(col, bench, risk_free, factor)                       | Treynor ratio of one column: mean(col - rf) / beta(col, bench).      |
| [`treynor_ratio`](#quantkit.stats.treynor_ratio)(returns, benchmark\[, risk_free, factor\]) | Compute the Treynor ratio of `returns` against `benchmark`.          |
| [`_compound`](#quantkit.stats._compound)(arr)                                               | Compound (geometrically linked) return of `arr`: prod(1 + r) - 1.    |
| [`_capture`](#quantkit.stats._capture)(col, bench, mask)                                    | Compound return of `col` over that of `bench` on `mask` rows.        |
| [`_up_capture`](#quantkit.stats._up_capture)(col, bench)                                    | Capture ratio over the rows where the benchmark did not fall.        |
| [`_down_capture`](#quantkit.stats._down_capture)(col, bench)                                | Capture ratio over the rows where the benchmark fell.                |
| [`_overall_capture`](#quantkit.stats._overall_capture)(col, bench)                          | Up capture over down capture, NaN when the latter is zero.           |
| [`_batting_average`](#quantkit.stats._batting_average)(col, bench)                          | Fraction of the rows where `col` is at or above `bench`.             |
| [`up_capture`](#quantkit.stats.up_capture)(returns, benchmark)                              | Compute the up capture ratio against `benchmark`.                    |
| [`down_capture`](#quantkit.stats.down_capture)(returns, benchmark)                          | Compute the down capture ratio against `benchmark`.                  |
| [`overall_capture`](#quantkit.stats.overall_capture)(returns, benchmark)                    | Compute the overall capture ratio against `benchmark`.               |
| [`batting_average`](#quantkit.stats.batting_average)(returns, benchmark)                    | Compute the fraction of periods that beat or match `benchmark`.      |
| [`sterling_ratio`](#quantkit.stats.sterling_ratio)(returns\[, periods_per_year, excess\])   | Calculate Morningstar's Sterling ratio.                              |
| [`_active_return`](#quantkit.stats._active_return)(col, bench)                              | Active return `col - bench` and its standard deviation, `ddof=1`.    |
| [`_tracking_error`](#quantkit.stats._tracking_error)(col, bench, factor)                    | Sample standard deviation of the active return, times `factor`.      |
| [`_information_ratio`](#quantkit.stats._information_ratio)(col, bench, factor)              | Mean active return over its standard deviation, times `factor`.      |
| [`tracking_error`](#quantkit.stats.tracking_error)(returns, benchmark\[, factor\])          | Calculate the tracking error against `benchmark`.                    |
| [`information_ratio`](#quantkit.stats.information_ratio)(returns, benchmark\[, factor\])    | Calculate the arithmetic information ratio against `benchmark`.      |

<a id="module-contents"></a>

## Module Contents

<a id="quantkit.stats._empty_reduction"></a>

### quantkit.stats.\_empty_reduction(obj)

Reduce an object with no observation to NaN, one value per column.

A reducer with nothing to reduce has an undefined result, which is NaN by convention, never an exception and never inf.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._positions"></a>

### quantkit.stats.\_positions(index)

Turn the output of the `*_valid_index` utils into integer positions.

A column without a valid observation comes out of those utils as None (1-dimension) or NaN (2-dimension), which cannot index an array. Such a column gets position 0; its value there is NaN, so the reduction of the column comes out as NaN by itself.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.total_returns"></a>

### quantkit.stats.total_returns(prices, factor=None, relative=True)

Calculate arithmetic total return.

Given a prices series this function returns the \[total return\]\[1\] (a.k.a RoR) as ouput.

* **Parameters:**
  **prices** — array-like
  : Prices data.

  **factor** — float, optional
  : The annualization factor is the (1/t) term from the annualized total return as show the following equation: r = (1+R)^(1/t)-1 = sqrt\[t\](1+R)-1

  **relative** — bool, optional
  : If True returns the output in part per units.
* **Returns:**
  **total_return** — float or array-like
  : Float for 1d input, one value per column for 2d input (a Series indexed by the columns for a DataFrame). NaN when a column has fewer than two valid observations.
* **Raises:**
  NotImplementedError
  : When prices has more than 2 dimensions.

### References

<a id="ree3c07e95b99-1"></a>

[1]

[https://en.wikipedia.org/wiki/Rate_of_return](https://en.wikipedia.org/wiki/Rate_of_return)

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.volatility"></a>

### quantkit.stats.volatility(returns, factor=None, ddof=1)

Calculate the volatility of the arithmetic returns.

Calculates the volatility $\sigma$ over a returns series $r_t$ and change the basis of the math:sigma using the factor $F$.

<!-- math:  \sigma = \sqrt{\frac{\sum{(r_t - \bar{r})^2}}{n - \text{ddof}}} \cdot F -->

The factor can be help to change the basis of the $\sigma$, for example: converting a sigma_{text{daily}} in sigma_{text{yearly}} multiplying it by a factor $F$:

<!-- math:  \sigma _{\text{yearly}} = \sigma_{\text{daily}}{ \sqrt{\frac{252}{12}}} -->

The factor above is equal to $F = \sqrt{\frac{252}{12}}$. For more information check out the \[ref\]\[1\].

* **Parameters:**
  **returns** — array-like
  : Data over Volatility will be calculated.

  **factor** — float, optional
  : Normalization factor of the volatility result. Use this to change the basis from daily to montly or yearly.

  **ddof** — int, optional
  : Degree of freedom. This help to use the unbiased estimation of the standard diviation, \[see\]\[2\].
* **Returns:**
  **volatility** — array-like
  : NaN when a column has no valid observation.

### References

<a id="r4d33e08c37d5-1"></a>

[1]

[https://en.wikipedia.org/wiki/Volatility_(finance](https://en.wikipedia.org/wiki/Volatility_(finance)) #Mathematical_definition

<a id="r4d33e08c37d5-2"></a>

[2]

[https://en.wikipedia.org/wiki/](https://en.wikipedia.org/wiki/) Unbiased_estimation_of_standard_deviation

### Examples

```pycon
>>> returns = np.array([1, 2, 3]),
>>> volatility(returns, factor=None, ddof=1):
1
```

```pycon
>>> returns = pd.Series([1, 2, 3]),
>>> volatility(returns, factor=None, ddof=1):
1
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.drawdown"></a>

### quantkit.stats.drawdown(prices, relative=True)

Compute the drawdown statistic.

The drawdown of a price process $S$ at time $t$ is defined as the drop of the asset prices from its running maximum up to time $t$

$$
D_{t} = \max_{u \in [0, t]}(S_{u}) - S_{t}
$$

For the whole period, the running maximum is the maximum of the period. Columns are reduced independently, each one against its own maximum.

* **Parameters:**
  **prices** — array-like
  : Series on which the drawdown is to be calculated.

  **relative** — bool, optional
  : Passing True makes the drawdown series relative (in parts per unit).
* **Returns:**
  **out** — float or array-like
  : Float for 1d input, one value per column for 2d input. NaN when a column has no valid observation.

### References

<a id="r8dd2430cef50-1"></a>

[1]

Jan Vecer - Maximum Drawdown and Directional Trading, Risk 19(12), 88-92, 2006.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.max_drawdown"></a>

### quantkit.stats.max_drawdown(prices, relative=True)

Calculate the Maximum Drawdown.

Max drawdown is the maximum potential loss of the trading system. Drawdown is defined as the distance between a given point and the highest point before it on the equity curve:

$$
D_{t} = \max_{u \in [0, t]}(S_{u}) - S_{t}
$$

“Max drawdown” is the largest drawdown value observed in the given time series for the whole period.

$$
MD_{T} = \max_{t \in [0, T] ( \max_{u \in [0, k]}(S_{u}) - S_{t} )
$$

* **Parameters:**
  **prices** — array-like
  : Series on which the drawdown is to be calculated.

  **relative** — bool, optional
  : Passing True makes the drawdown series relative (in parts per unit).
* **Returns:**
  **out** — float or array-like
  : Maximum Drawdown value. NaN when a column has no valid observation.

### Examples

```pycon
>>> prices = np.array([1, 2, 3])]
>>> max_drawdown(prices)
0
```

```pycon
>>> prices = np.array([1, 0, 3])]
>>> max_drawdown(prices)
-1
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.sharpe_ratio"></a>

### quantkit.stats.sharpe_ratio(returns: quantkit.conventions.ArrayLike, risk_free: float | quantkit.conventions.ArrayLike, factor: float = np.sqrt(BYEAR))

Calculate shape ratio.

It measures the performance of an investment such as a security or portfolio compared to a risk-free asset, after adjusting for its risk.

It is defined as the difference beteween the returns of the investement and the risk-free return, divided by the standard deviation of the investment returns.

It represents the additional amount of return that an investor recieves per unit of increase in risk\[Rd6c83f64a34c-1\]_.

* **Parameters:**
  **returns** — array-like
  : Asset return series. Reductions are column-wise.

  **risk_free** — number or array-like
  : Risk free return. A number is subtracted from every period; a 1d series is a per-period rate and is subtracted row by row, aligned by index when both it and `returns` are pandas objects.

  **factor** — float
  : Annualization factor which multiplies the raw sharpe ratio.
* **Returns:**
  **out** — float or array-like
  : Float for 1d input, one value per column for 2d input. NaN when the excess return has no deviation (zero denominator) or when a column has no valid observation.

### References

<a id="rd6c83f64a34c-1"></a>

[1]

[https://en.wikipedia.org/wiki/Sharpe_ratio](https://en.wikipedia.org/wiki/Sharpe_ratio)

### Examples

```pycon
>>> sharpe_ratio(np.array([0, 2]), np.array([0, 0]), factor=1)
1.0
```

Two columns reduce to one value each:

```pycon
>>> returns = np.array([[0.0, 0.0], [0.0, 0.2], [0.3, 0.4]])
>>> sharpe_ratio(returns, 0.0, factor=1)
array([0.70710678, 1.22474487])
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.value_at_risk"></a>

### quantkit.stats.value_at_risk(returns, confidence=0.95)

Calculate the historical value at risk (VaR).

The historical VaR at confidence level $c$ is the empirical $(1 - c)$ quantile of the returns, i.e. the return that is only undercut with probability $1 - c$ in the sample:

$$
\text{VaR}_{c} = Q_{1 - c}(r_t)
$$

The quantile is computed with numpy’s default linear interpolation between order statistics, ignoring NaN.

The sign of the return is kept, so a loss is a NEGATIVE number, the same convention as [`drawdown()`](#quantkit.stats.drawdown). Morningstar (and most risk reports) quote VaR as a positive loss; negate the output if that convention is needed.

* **Parameters:**
  **returns** — array-like
  : Asset return series.

  **confidence** — float, optional
  : Confidence level, strictly between 0 and 1. The default 0.95 gives the 5th percentile of the returns.
* **Returns:**
  **out** — float or array-like
  : Value at risk of each column. NaN for a column without valid observations.
* **Raises:**
  ValueError
  : If `confidence` is not in the open interval (0, 1).

### References

<a id="r58b4b31be44b-1"></a>

[1]

[https://en.wikipedia.org/wiki/Value_at_risk](https://en.wikipedia.org/wiki/Value_at_risk)

<a id="r58b4b31be44b-2"></a>

[2]

Jorion, P. (2006). Value at Risk: The New Benchmark for Managing Financial Risk. McGraw-Hill.

### Examples

```pycon
>>> returns = np.array([-0.05, -0.02, 0.0, 0.01, 0.03])
>>> value_at_risk(returns, confidence=0.75)
-0.02
```

```pycon
>>> returns = pd.DataFrame({"a": returns, "b": returns + 0.1})
>>> value_at_risk(returns, confidence=0.75)
a   -0.02
b    0.08
dtype: float64
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.max_drawup"></a>

### quantkit.stats.max_drawup(prices, relative=True)

Calculate the Maximum Drawup.

The drawup is the rise of the prices from their running minimum, the mirror image of the drawdown:

$$
U_{t} = S_{t} - \min_{u \in [0, t]}(S_{u})
$$

“Max drawup” is the largest drawup value observed in the given time series for the whole period.

$$
MU_{T} = \max_{t \in [0, T]} ( S_{t} - \min_{u \in [0, t]}(S_{u}) )
$$

* **Parameters:**
  **prices** — array-like
  : Series on which the drawup is to be calculated.

  **relative** — bool, optional
  : Passing True makes the drawup relative (in parts per unit) to the running minimum.
* **Returns:**
  **out** — float or array-like
  : Maximum Drawup value. NaN when there is no valid observation.

### References

<a id="rfce341fbf366-1"></a>

[1]

Jan Vecer - Maximum Drawdown and Directional Trading, Risk 19(12), 2006. Maximum drawdown and maximum drawup are studied as a pair. [http://www.stat.columbia.edu/~vecer/maxdrawdown3.pdf](http://www.stat.columbia.edu/~vecer/maxdrawdown3.pdf)

### Examples

```pycon
>>> prices = np.array([4, 2, 3, 6])
>>> max_drawup(prices, relative=False)
4.0
```

```pycon
>>> max_drawup(prices)
2.0
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.annualized_return"></a>

### quantkit.stats.annualized_return(returns, periods_per_year=BYEAR)

Calculate the geometric annualized return.

Compounds the returns of the whole period and rescales the result to a one year basis, so that series of different lengths are comparable:

$$
R_{\text{annual}} = \left( \prod_{t=1}^{n} (1 + r_t) \right)
    ^{\frac{P}{n}} - 1
$$

where $P$ is `periods_per_year` and $n$ is the number of non-NaN returns of each column. NaN values are dropped from both the product and the count.

* **Parameters:**
  **returns** — array-like
  : Arithmetic returns series, 1-d or 2-d (columns are reduced independently).

  **periods_per_year** — float, optional
  : Number of returns that make up a year, `BYEAR` (business days) by default. Use 12 for monthly and 52 for weekly returns.
* **Returns:**
  **out** — float or 1d-reduced-array
  : Annualized return of each column. NaN when a column has no valid return.

### References

<a id="r4cd02ae67ab9-1"></a>

[1]

[https://en.wikipedia.org/wiki/Rate_of_return#Annualization](https://en.wikipedia.org/wiki/Rate_of_return#Annualization)

### Examples

Doubling twice over two years annualizes to +100%.

```pycon
>>> annualized_return(np.array([1.0, 1.0]), periods_per_year=1)
1.0
```

Doubling in half a year annualizes to +300%.

```pycon
>>> annualized_return(np.array([1.0]), periods_per_year=2)
3.0
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.calmar_ratio"></a>

### quantkit.stats.calmar_ratio(returns, periods_per_year=BYEAR)

Calculate the Calmar ratio.

Return earned per unit of maximum drawdown suffered: the annualized return divided by the absolute maximum drawdown of the price path implied by `returns`, starting from a capital of 1 [\[1\]](#r2702a60e7162-1).

$$
\text{Calmar} = \frac{R_{\text{annual}}}{|MD|}
$$

The price path is `cum_returns(returns, first_price=1)` preceded by the starting capital itself, so a loss on the very first return counts as a drawdown. Morningstar computes the ratio over the trailing 36 months; here the caller chooses the window by slicing `returns` before calling.

* **Parameters:**
  **returns** — array-like
  : Arithmetic returns series, 1-d or 2-d (columns are reduced independently).

  **periods_per_year** — float, optional
  : Number of returns that make up a year, passed to [`annualized_return()`](#quantkit.stats.annualized_return). `BYEAR` (business days) by default.
* **Returns:**
  **out** — float or 1d-reduced-array
  : Calmar ratio of each column. NaN when there is no drawdown to divide by or no valid return.

### References

<a id="r2702a60e7162-1"></a>

[1]

[https://en.wikipedia.org/wiki/Calmar_ratio](https://en.wikipedia.org/wiki/Calmar_ratio)

<a id="r2702a60e7162-2"></a>

[2]

Young, T. W. (1991). Calmar Ratio: A Smoother Tool. Futures, 20(1), 40.

### Examples

Halve the capital, then quadruple it: the year ends +100% after a 50% drawdown.

```pycon
>>> calmar_ratio(np.array([-0.5, 3.0]), periods_per_year=2)
2.0
```

Without a drawdown the ratio is undefined.

```pycon
>>> calmar_ratio(np.array([0.1, 0.2]), periods_per_year=2)
nan
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._nan_divide"></a>

### quantkit.stats.\_nan_divide(numerator, denominator)

Divide returning NaN, never inf, where the denominator is zero.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._masked_mean"></a>

### quantkit.stats.\_masked_mean(arr, mask, method)

Column-wise mean of `arr` over the periods selected by `mask`.

Periods outside the mask, NaN included, do not contribute. A column with no selected period gives NaN.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.average_gain"></a>

### quantkit.stats.average_gain(returns, method='arith')

Calculate the average gain.

The mean of the returns over the periods with a gain, $r_t > 0$. Periods with a zero or negative return and NaN are left out.

$$
\text{arith} = \frac{1}{n_g} \sum_{r_t > 0} r_t

\text{geo} = \left( \prod_{r_t > 0} (1 + r_t) \right)^{1 / n_g} - 1
$$

The geometric mean is Morningstar’s definition of Average Gain [\[1\]](#r1edb1509ab53-1).

* **Parameters:**
  **returns** — array-like
  : Returns series, 1 or 2 dimensions. Reduces along axis 0.

  **method** — {“arith”, “geo”}, optional
  : Arithmetic (default) or geometric mean of the gains.
* **Returns:**
  **out** — float or array-like
  : Average gain per column. NaN when there is no gain.
* **Raises:**
  ValueError
  : If `method` is neither “arith” nor “geo”.

### References

<a id="r1edb1509ab53-1"></a>

[1]

Morningstar, Custom Calculation Data Points, October 2016. [https://morningstardirect.morningstar.com/clientcomm/customcalculations.pdf](https://morningstardirect.morningstar.com/clientcomm/customcalculations.pdf)

### Examples

```pycon
>>> returns = np.array([0.10, -0.05, 0.20])
>>> average_gain(returns)
0.15
```

```pycon
>>> average_gain(returns, method="geo")
0.1489125293076057
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.average_loss"></a>

### quantkit.stats.average_loss(returns, method='arith')

Calculate the average loss.

The mean of the returns over the periods with a loss, $r_t < 0$, so the result is negative. Periods with a zero or positive return and NaN are left out.

$$
\text{arith} = \frac{1}{n_l} \sum_{r_t < 0} r_t

\text{geo} = \left( \prod_{r_t < 0} (1 + r_t) \right)^{1 / n_l} - 1
$$

The geometric mean is Morningstar’s definition of Average Loss [\[1\]](#r68cdce24c3e5-1).

* **Parameters:**
  **returns** — array-like
  : Returns series, 1 or 2 dimensions. Reduces along axis 0.

  **method** — {“arith”, “geo”}, optional
  : Arithmetic (default) or geometric mean of the losses.
* **Returns:**
  **out** — float or array-like
  : Average loss per column, negative. NaN when there is no loss.
* **Raises:**
  ValueError
  : If `method` is neither “arith” nor “geo”.

### References

<a id="r68cdce24c3e5-1"></a>

[1]

Morningstar, Custom Calculation Data Points, October 2016. [https://morningstardirect.morningstar.com/clientcomm/customcalculations.pdf](https://morningstardirect.morningstar.com/clientcomm/customcalculations.pdf)

### Examples

```pycon
>>> returns = np.array([-0.10, 0.05, -0.20])
>>> average_loss(returns)
-0.15
```

```pycon
>>> average_loss(returns, method="geo")
-0.1514718625761430
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.gain_loss_ratio"></a>

### quantkit.stats.gain_loss_ratio(returns)

Calculate Morningstar’s Gain/Loss Ratio.

The ratio between the arithmetic average gain and the arithmetic average loss, in absolute value, multiplied by the ratio between the number of periods with a gain and the number of periods with a loss [\[1\]](#rb1f614ab8d61-1). Zero returns and NaN are neither gains nor losses.

$$
GL = \left| \frac{\bar{r}_g}{\bar{r}_l} \right| \frac{n_g}{n_l}
   = \frac{\sum_{r_t > 0} r_t}{\left| \sum_{r_t < 0} r_t \right|}
$$

Both forms are the same number: the counts cancel out and the ratio is the sum of the gains over the absolute sum of the losses.

* **Parameters:**
  **returns** — array-like
  : Returns series, 1 or 2 dimensions. Reduces along axis 0.
* **Returns:**
  **out** — float or array-like
  : Gain/Loss Ratio per column. NaN when there is no loss, 0 when there are losses but no gain.

### References

<a id="rb1f614ab8d61-1"></a>

[1]

Morningstar, Custom Calculation Data Points, October 2016. [https://morningstardirect.morningstar.com/clientcomm/customcalculations.pdf](https://morningstardirect.morningstar.com/clientcomm/customcalculations.pdf)

### Examples

```pycon
>>> returns = np.array([0.1, 0.2, -0.1])
>>> gain_loss_ratio(returns)
3.0
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.up_period_percent"></a>

### quantkit.stats.up_period_percent(returns)

Calculate the fraction of periods with a return at or above zero.

Number of periods whose return is greater than or equal to 0 over the number of valid periods [\[1\]](#r20dd35aa7d0b-1). A zero return counts as an up period and NaN is left out of both counts.

$$
UP = \frac{\#\{t : r_t \geq 0\}}{\#\{t : r_t \text{ is not NaN}\}}
$$

Despite the name, the output is in parts per unit (0.75, not 75), like the rest of the library with `relative=True`.

* **Parameters:**
  **returns** — array-like
  : Returns series, 1 or 2 dimensions. Reduces along axis 0.
* **Returns:**
  **out** — float or array-like
  : Fraction in \[0, 1\] per column. NaN when there is no valid period. It adds up to 1 with [`down_period_percent()`](#quantkit.stats.down_period_percent).

### References

<a id="r20dd35aa7d0b-1"></a>

[1]

Morningstar, Custom Calculation Data Points, October 2016. [https://morningstardirect.morningstar.com/clientcomm/customcalculations.pdf](https://morningstardirect.morningstar.com/clientcomm/customcalculations.pdf)

### Examples

```pycon
>>> returns = np.array([0.1, 0.0, -0.1, 0.2])
>>> up_period_percent(returns)
0.75
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.down_period_percent"></a>

### quantkit.stats.down_period_percent(returns)

Calculate the fraction of periods with a return below zero.

Number of periods whose return is less than 0 over the number of valid periods [\[1\]](#ra31197a7ed97-1). NaN is left out of both counts.

$$
DOWN = \frac{\#\{t : r_t < 0\}}{\#\{t : r_t \text{ is not NaN}\}}
$$

Despite the name, the output is in parts per unit (0.25, not 25), like the rest of the library with `relative=True`.

* **Parameters:**
  **returns** — array-like
  : Returns series, 1 or 2 dimensions. Reduces along axis 0.
* **Returns:**
  **out** — float or array-like
  : Fraction in \[0, 1\] per column. NaN when there is no valid period. It adds up to 1 with [`up_period_percent()`](#quantkit.stats.up_period_percent).

### References

<a id="ra31197a7ed97-1"></a>

[1]

Morningstar, Custom Calculation Data Points, October 2016. [https://morningstardirect.morningstar.com/clientcomm/customcalculations.pdf](https://morningstardirect.morningstar.com/clientcomm/customcalculations.pdf)

### Examples

```pycon
>>> returns = np.array([0.1, 0.0, -0.1, 0.2])
>>> down_period_percent(returns)
0.25
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._nanmean"></a>

### quantkit.stats.\_nanmean(arr)

Column-wise mean ignoring NaN.

Same as `np.nanmean(arr, axis=0)` but returns NaN silently, instead of warning, when a column has no valid observation.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.downside_deviation"></a>

### quantkit.stats.downside_deviation(returns, mar=0.0, factor=None)

Calculate the downside deviation of the returns.

Root mean square of the shortfalls below the minimum acceptable return $\text{MAR}$:

$$
\sigma_d = \sqrt{\frac{1}{N}
           \sum_{t=1}^{N} \min(r_t - \text{MAR}, 0)^2}
$$

The mean is taken over all $N$ valid observations, not only over those below the MAR. This is the Sortino / Morningstar convention [\[1\]](#ra63ae8d05531-1) [\[2\]](#ra63ae8d05531-2) and makes the statistic the denominator of [`sortino_ratio()`](#quantkit.stats.sortino_ratio).

* **Parameters:**
  **returns** — array-like
  : Returns series. Reductions are column-wise.

  **mar** — float, optional
  : Minimum acceptable return, in the same units as `returns`.

  **factor** — float, optional
  : Multiplies the result. To annualize a deviation of returns of a shorter period pass the square root of the number of periods in a year, e.g. `np.sqrt(BYEAR)` for daily returns.
* **Returns:**
  **out** — float or array-like
  : Float for 1d input, one value per column for 2d input. NaN when there is no valid observation.

### References

<a id="ra63ae8d05531-1"></a>

[1]

Sortino, F. A., & Price, L. N. (1994). Performance measurement in a downside risk framework. The Journal of Investing, 3(3), 59-64.

<a id="ra63ae8d05531-2"></a>

[2]

[https://en.wikipedia.org/wiki/Downside_risk](https://en.wikipedia.org/wiki/Downside_risk)

### Examples

```pycon
>>> downside_deviation(np.array([0.1, -0.1, 0.3, -0.2]))
0.1118033988749895
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.upside_deviation"></a>

### quantkit.stats.upside_deviation(returns, mar=0.0, factor=None)

Calculate the upside deviation of the returns.

Root mean square of the excess returns above the minimum acceptable return $\text{MAR}$, the mirror image of [`downside_deviation()`](#quantkit.stats.downside_deviation):

$$
\sigma_u = \sqrt{\frac{1}{N}
           \sum_{t=1}^{N} \max(r_t - \text{MAR}, 0)^2}
$$

The mean is taken over all $N$ valid observations, not only over those above the MAR.

* **Parameters:**
  **returns** — array-like
  : Returns series. Reductions are column-wise.

  **mar** — float, optional
  : Minimum acceptable return, in the same units as `returns`.

  **factor** — float, optional
  : Multiplies the result. To annualize a deviation of returns of a shorter period pass the square root of the number of periods in a year, e.g. `np.sqrt(BYEAR)` for daily returns.
* **Returns:**
  **out** — float or array-like
  : Float for 1d input, one value per column for 2d input. NaN when there is no valid observation.

### References

<a id="r13d03c88c427-1"></a>

[1]

[https://en.wikipedia.org/wiki/Downside_risk](https://en.wikipedia.org/wiki/Downside_risk)

### Examples

```pycon
>>> upside_deviation(np.array([0.1, -0.1, 0.3, -0.2]))
0.15811388300841897
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.kappa"></a>

### quantkit.stats.kappa(returns, mar=0.0, order=2, factor=None)

Calculate the Kappa ratio of Kaplan and Knowles.

Generalized downside risk-adjusted performance measure: the mean excess return over the minimum acceptable return $\text{MAR}$ divided by the `order`-th root of the lower partial moment of that order [\[1\]](#r5899e024835c-1):

$$
\kappa_n = \frac{\bar{r} - \text{MAR}}{\sqrt[n]{\text{LPM}_n}},
\quad
\text{LPM}_n = \frac{1}{N} \sum_{t=1}^{N} \max(\text{MAR} - r_t, 0)^n
$$

`order=1` is the Omega ratio minus one (see [`omega_ratio()`](#quantkit.stats.omega_ratio)) and `order=2` is the Sortino ratio (see [`sortino_ratio()`](#quantkit.stats.sortino_ratio)).

* **Parameters:**
  **returns** — array-like
  : Returns series. Reductions are column-wise.

  **mar** — float, optional
  : Minimum acceptable return, in the same units as `returns`.

  **order** — float, optional
  : Order $n$ of the lower partial moment. Must be positive.

  **factor** — float, optional
  : Multiplies the result. Annualizing the ratio requires the caller to pass the matching factor, e.g. `np.sqrt(BYEAR)` for the Sortino ratio (`order=2`) of daily returns.
* **Returns:**
  **out** — float or array-like
  : Float for 1d input, one value per column for 2d input. NaN when the lower partial moment is zero (no return below the MAR) or when there is no valid observation.
* **Raises:**
  ValueError
  : If `order` is not positive.

### References

<a id="r5899e024835c-1"></a>

[1]

Kaplan, P. D., & Knowles, J. A. (2004). Kappa: a generalized downside risk-adjusted performance measure. Journal of Performance Measurement, 8(3), 42-54.

### Examples

```pycon
>>> kappa(np.array([0.1, -0.1, 0.3, -0.2]), order=1)
0.3333333333333332
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.omega_ratio"></a>

### quantkit.stats.omega_ratio(returns, mar=0.0)

Calculate the Omega ratio of Shadwick and Keating.

Probability-weighted gains over probability-weighted losses relative to the minimum acceptable return $\text{MAR}$ [\[1\]](#rbf93c0d6c169-1). For a sample of returns it is the sum of the excess returns above the MAR over the sum of the shortfalls below it:

$$
\Omega = \frac{\sum_{t=1}^{N} \max(r_t - \text{MAR}, 0)}
              {\sum_{t=1}^{N} \max(\text{MAR} - r_t, 0)}
$$

It equals `1 + kappa(returns, mar, order=1)`.

* **Parameters:**
  **returns** — array-like
  : Returns series. Reductions are column-wise.

  **mar** — float, optional
  : Minimum acceptable return, in the same units as `returns`.
* **Returns:**
  **out** — float or array-like
  : Float for 1d input, one value per column for 2d input. NaN when no return is below the MAR (zero denominator) or when there is no valid observation.

### References

<a id="rbf93c0d6c169-1"></a>

[1]

Keating, C., & Shadwick, W. F. (2002). A universal performance measure. Journal of Performance Measurement, 6(3), 59-84.

### Examples

```pycon
>>> omega_ratio(np.array([0.1, -0.1, 0.3, -0.2]))
1.3333333333333333
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.sortino_ratio"></a>

### quantkit.stats.sortino_ratio(returns, mar=0.0, factor=None)

Calculate the Sortino ratio.

Risk-adjusted return that, unlike the Sharpe ratio, penalizes only the returns falling below the minimum acceptable return $\text{MAR}$ [\[1\]](#r8059a77b4122-1) [\[2\]](#r8059a77b4122-2):

$$
S = \frac{\bar{r} - \text{MAR}}{\sigma_d}
$$

where $\sigma_d$ is the [`downside_deviation()`](#quantkit.stats.downside_deviation). It is [`kappa()`](#quantkit.stats.kappa) with `order=2`.

* **Parameters:**
  **returns** — array-like
  : Returns series. Reductions are column-wise.

  **mar** — float, optional
  : Minimum acceptable return (a.k.a. target or required return), in the same units as `returns`.

  **factor** — float, optional
  : Multiplies the result. Annualizing the ratio requires the caller to pass the square root of the number of periods in a year, e.g. `np.sqrt(BYEAR)` for daily returns.
* **Returns:**
  **out** — float or array-like
  : Float for 1d input, one value per column for 2d input. NaN when no return is below the MAR (zero downside deviation) or when there is no valid observation.

### References

<a id="r8059a77b4122-1"></a>

[1]

[https://en.wikipedia.org/wiki/Sortino_ratio](https://en.wikipedia.org/wiki/Sortino_ratio)

<a id="r8059a77b4122-2"></a>

[2]

[http://www.redrockcapital.com/](http://www.redrockcapital.com/) Sortino_\_A_\_Sharper_\_Ratio_Red_Rock_Capital.pdf

### Examples

```pycon
>>> rets = np.array([0.17, 0.15, 0.23, -0.05, 0.12, 0.09, 0.13, -0.04])
>>> sortino_ratio(rets)
4.417261042993862
```

```pycon
>>> sortino_ratio(np.array([-0.1, -0.1, -0.1, -0.1]))
-1.0
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._columns"></a>

### quantkit.stats.\_columns(arr)

Return the columns of a 1D or 2D array as a list of 1D arrays.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._reduce_columns"></a>

### quantkit.stats.\_reduce_columns(prices, values)

Wrap one value per column of `prices` like `reduce_array_wrap`.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._apply_columns"></a>

### quantkit.stats.\_apply_columns(prices, func, \*args)

Apply `func(column, *args)` to every column of `prices`.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._max_drawdown_span"></a>

### quantkit.stats.\_max_drawdown_span(arr)

Locate the maximum relative drawdown of a 1D array.

* **Parameters:**
  **arr** — numpy.ndarray
  : 1D prices. Leading NaN are skipped and internal NaN are missing observations: the running maximum carries forward through them and they are never chosen.
* **Returns:**
  **peak, valley, recovery** — int
  : Positions of the running maximum in force at the valley (its last touch before the valley), of the minimum drawdown (the first one on ties) and of the first price at or above the peak after the valley. `recovery` is -1 when the price never recovers and all three are -1 when the price never falls below a previous high.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._max_drawdown_label"></a>

### quantkit.stats.\_max_drawdown_label(prices, item)

Report one item of `_max_drawdown_span` per column.

numpy input keeps the positions as floats so NaN can mean “none”; pandas input maps them to the index labels, missing (NaT or NaN) when none.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._max_drawdown_duration"></a>

### quantkit.stats.\_max_drawdown_duration(arr)

Periods from peak to valley; 0 without drawdown, NaN without data.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._max_drawdown_recovery_duration"></a>

### quantkit.stats.\_max_drawdown_recovery_duration(arr)

Periods from valley to recovery; NaN when there is none.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._longest_drawdown_duration"></a>

### quantkit.stats.\_longest_drawdown_duration(arr)

Longest run of valid observations strictly below the running max.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._average_drawdown"></a>

### quantkit.stats.\_average_drawdown(arr, periods_per_year)

Morningstar’s average drawdown of a 1D array.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.max_drawdown_peak"></a>

### quantkit.stats.max_drawdown_peak(prices)

Locate the peak the maximum drawdown fell from.

The peak is the last time the price stood at the running maximum in force at the valley of the maximum drawdown, see [`max_drawdown()`](#quantkit.stats.max_drawdown).

* **Parameters:**
  **prices** — array-like
  : Prices data.
* **Returns:**
  **out** — float, label or array-like
  : For numpy input the position as a float (NaN without drawdown), for pandas input the index label (missing without drawdown). Reduced column-wise: a Series indexed by the columns for a DataFrame.

### References

<a id="r79d0a9d91405-1"></a>

[1]

[https://en.wikipedia.org/wiki/Drawdown_(economics](https://en.wikipedia.org/wiki/Drawdown_(economics))

<a id="r79d0a9d91405-2"></a>

[2]

Enrico Schumann - Computing Drawdown Statistics [http://comisef.wikidot.com/tutorial:drawdowns](http://comisef.wikidot.com/tutorial:drawdowns)

### Examples

```pycon
>>> import numpy as np
>>> max_drawdown_peak(np.array([10, 8, 6, 9, 10, 7]))
0.0
```

```pycon
>>> max_drawdown_peak(np.array([1, 2, 3]))
nan
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.max_drawdown_valley"></a>

### quantkit.stats.max_drawdown_valley(prices)

Locate the valley (trough) of the maximum drawdown.

The valley is the first time the relative drawdown reaches its minimum, see [`max_drawdown()`](#quantkit.stats.max_drawdown).

* **Parameters:**
  **prices** — array-like
  : Prices data.
* **Returns:**
  **out** — float, label or array-like
  : For numpy input the position as a float (NaN without drawdown), for pandas input the index label (missing without drawdown). Reduced column-wise: a Series indexed by the columns for a DataFrame.

### References

<a id="r0eb77d24c4d1-1"></a>

[1]

[https://en.wikipedia.org/wiki/Drawdown_(economics](https://en.wikipedia.org/wiki/Drawdown_(economics))

<a id="r0eb77d24c4d1-2"></a>

[2]

Enrico Schumann - Computing Drawdown Statistics [http://comisef.wikidot.com/tutorial:drawdowns](http://comisef.wikidot.com/tutorial:drawdowns)

### Examples

```pycon
>>> import numpy as np
>>> max_drawdown_valley(np.array([10, 8, 6, 9, 10, 7]))
2.0
```

```pycon
>>> import pandas as pd
>>> index = pd.date_range("2020-01-01", periods=6)
>>> max_drawdown_valley(pd.Series([10, 8, 6, 9, 10, 7], index=index))
Timestamp('2020-01-03 00:00:00')
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.max_drawdown_recovery"></a>

### quantkit.stats.max_drawdown_recovery(prices)

Locate the recovery from the maximum drawdown.

The recovery is the first time after the valley the price is back at or above the peak price, see [`max_drawdown_peak()`](#quantkit.stats.max_drawdown_peak).

* **Parameters:**
  **prices** — array-like
  : Prices data.
* **Returns:**
  **out** — float, label or array-like
  : For numpy input the position as a float, for pandas input the index label. NaN (missing) when the price never recovers or there is no drawdown. Reduced column-wise: a Series indexed by the columns for a DataFrame.

### References

<a id="rb3a4060d2629-1"></a>

[1]

[https://en.wikipedia.org/wiki/Drawdown_(economics](https://en.wikipedia.org/wiki/Drawdown_(economics))

<a id="rb3a4060d2629-2"></a>

[2]

Enrico Schumann - Computing Drawdown Statistics [http://comisef.wikidot.com/tutorial:drawdowns](http://comisef.wikidot.com/tutorial:drawdowns)

### Examples

```pycon
>>> import numpy as np
>>> max_drawdown_recovery(np.array([10, 8, 6, 9, 10, 7]))
4.0
```

```pycon
>>> max_drawdown_recovery(np.array([10, 8, 6, 9]))
nan
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.max_drawdown_duration"></a>

### quantkit.stats.max_drawdown_duration(prices)

Calculate the duration of the maximum drawdown.

Number of periods from the peak to the valley of the maximum drawdown, see [`max_drawdown_peak()`](#quantkit.stats.max_drawdown_peak) and [`max_drawdown_valley()`](#quantkit.stats.max_drawdown_valley).

* **Parameters:**
  **prices** — array-like
  : Prices data.
* **Returns:**
  **out** — float or array-like
  : Periods from peak to valley, 0 when the price never falls below a previous high and NaN without valid observations.

### References

<a id="rd221a2124322-1"></a>

[1]

Bacon, C. Practical Portfolio Performance Measurement and Attribution. Wiley. 2004.

<a id="rd221a2124322-2"></a>

[2]

[https://en.wikipedia.org/wiki/Drawdown_(economics](https://en.wikipedia.org/wiki/Drawdown_(economics))

### Examples

```pycon
>>> import numpy as np
>>> max_drawdown_duration(np.array([10, 8, 6, 9, 10, 7]))
2.0
```

```pycon
>>> max_drawdown_duration(np.array([1, 2, 3]))
0.0
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.max_drawdown_recovery_duration"></a>

### quantkit.stats.max_drawdown_recovery_duration(prices)

Calculate the recovery time of the maximum drawdown.

Number of periods from the valley of the maximum drawdown to its recovery, see [`max_drawdown_recovery()`](#quantkit.stats.max_drawdown_recovery).

* **Parameters:**
  **prices** — array-like
  : Prices data.
* **Returns:**
  **out** — float or array-like
  : Periods from valley to recovery. NaN when the price never recovers or there is no drawdown.

### References

<a id="rc1e4b3677e02-1"></a>

[1]

Bacon, C. Practical Portfolio Performance Measurement and Attribution. Wiley. 2004.

<a id="rc1e4b3677e02-2"></a>

[2]

[https://en.wikipedia.org/wiki/Drawdown_(economics](https://en.wikipedia.org/wiki/Drawdown_(economics))

### Examples

```pycon
>>> import numpy as np
>>> max_drawdown_recovery_duration(np.array([10, 8, 6, 9, 10, 7]))
2.0
```

```pycon
>>> max_drawdown_recovery_duration(np.array([10, 8, 6, 9]))
nan
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.longest_drawdown_duration"></a>

### quantkit.stats.longest_drawdown_duration(prices)

Calculate the longest time under water.

Longest number of consecutive periods with the price strictly below its running maximum. It need not be the deepest drawdown. A stretch still open at the end of the series counts and internal NaN are ignored: they neither break nor extend a stretch.

* **Parameters:**
  **prices** — array-like
  : Prices data.
* **Returns:**
  **out** — float or array-like
  : Longest number of periods under water, 0 when the price never falls below a previous high and NaN without valid observations.

### References

<a id="re6bde3a2a4ea-1"></a>

[1]

Bacon, C. Practical Portfolio Performance Measurement and Attribution. Wiley. 2004.

<a id="re6bde3a2a4ea-2"></a>

[2]

[https://en.wikipedia.org/wiki/Drawdown_(economics](https://en.wikipedia.org/wiki/Drawdown_(economics))

### Examples

```pycon
>>> import numpy as np
>>> longest_drawdown_duration(np.array([10, 9, 9, 9, 10, 2, 10]))
3.0
```

```pycon
>>> longest_drawdown_duration(np.array([1, 2, 3]))
0.0
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.average_drawdown"></a>

### quantkit.stats.average_drawdown(prices, periods_per_year=BYEAR)

Calculate Morningstar’s Average Drawdown.

The valid observations are split into consecutive blocks of `periods_per_year` observations, the maximum relative drawdown of each block is computed independently (every block starts its own running maximum) and the sum is spread over the number of years covered:

$$
AvgDD = \frac{\sum_{t=1}^{n} MDD_{t}}{N / \text{periods\_per\_year}}
$$

where $N$ is the number of valid observations, so a partial last block is weighted by its length [\[1\]](#reb169a0f1987-1). This is the downside risk measure of the Sterling ratio. Drawdowns are negative in this library, so the result is negative or zero.

* **Parameters:**
  **prices** — array-like
  : Prices data.

  **periods_per_year** — int, optional
  : Observations per block (year).
* **Returns:**
  **out** — float or array-like
  : Average drawdown, 0 when the price never falls below a previous high and NaN without valid observations.

### References

<a id="reb169a0f1987-1"></a>

[1]

Morningstar - Custom Calculation Data Points, Average Drawdown [https://morningstardirect.morningstar.com/clientcomm/](https://morningstardirect.morningstar.com/clientcomm/) customcalculations.pdf

### Examples

```pycon
>>> import numpy as np
>>> prices = np.array([10, 8, 6, 9, 10, 7])
>>> average_drawdown(prices, periods_per_year=3)  # (-0.4 - 0.3) / 2
-0.35
```

```pycon
>>> average_drawdown(prices, periods_per_year=6)  # one block
-0.4
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._pairwise_complete"></a>

### quantkit.stats.\_pairwise_complete(col, bench)

Keep the rows where both `col` and `bench` are not NaN.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._is_degenerate"></a>

### quantkit.stats.\_is_degenerate(arr)

Tell whether a variance cannot be estimated from `arr`.

Either fewer than two observations or all of them identical. The identity test replaces `var == 0` because the variance of a constant array can come out as a tiny positive number when its mean is rounded.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._beta"></a>

### quantkit.stats.\_beta(col, bench)

Slope of `col` on `bench`: cov(col, bench) / var(bench).

Written as the ratio of the deviation sums so the `n - 1` cancels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._alpha"></a>

### quantkit.stats.\_alpha(col, bench, risk_free, factor)

Jensen’s alpha: mean(col - rf) - beta \* mean(bench - rf).

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._correlation"></a>

### quantkit.stats.\_correlation(col, bench)

Pearson correlation, NaN when either side has no dispersion.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._r_squared"></a>

### quantkit.stats.\_r_squared(col, bench)

Square of the Pearson correlation.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._bull_beta"></a>

### quantkit.stats.\_bull_beta(col, bench)

Beta on the rows where the benchmark went up.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._bear_beta"></a>

### quantkit.stats.\_bear_beta(col, bench)

Beta on the rows where the benchmark went down.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._reduce_pairwise"></a>

### quantkit.stats.\_reduce_pairwise(returns, benchmark, func)

Apply `func(col, bench)` to every column on its complete rows.

`returns` and `benchmark` are aligned with [`quantkit.utils.align()`](../utils/index.md#quantkit.utils.align); each column then keeps only the rows where both it and the benchmark are non-NaN before `func` reduces it to a number. The result is wrapped like the original `returns` object.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.beta"></a>

### quantkit.stats.beta(returns, benchmark)

Compute the beta of `returns` against `benchmark`.

Beta is the slope of the linear regression of the asset returns on the benchmark returns

$$
r_{i,t} = \alpha_i + \beta_i \cdot r_{b,t} + \epsilon_t
$$

whose ordinary least squares solution is

$$
\beta_i = \frac{cov(r_i, r_b)}{var(r_b)}
$$

Both moments use `ddof=1` and, for each column, only the rows where the column and the benchmark are both non-NaN (pairwise complete).

* **Parameters:**
  **returns** — array-like
  : 1D or 2D asset returns. Pandas objects are inner-joined with the benchmark on their index, see [`quantkit.utils.align()`](../utils/index.md#quantkit.utils.align).

  **benchmark** — array-like
  : 1D benchmark returns, usually a market index.
* **Returns:**
  **beta** — float or array-like
  : One value per column of `returns`. NaN when the benchmark has no variance or fewer than two complete rows are available.
* **Raises:**
  ValueError
  : If `benchmark` is not 1D or the numpy lengths differ.

### References

<a id="r1a9ab85d5ab8-1"></a>

[1]

[https://en.wikipedia.org/wiki/Beta_(finance](https://en.wikipedia.org/wiki/Beta_(finance))

### Examples

```pycon
>>> benchmark = np.array([0.5, -0.25, 0.75, -0.5])
>>> beta(2 * benchmark + 0.125, benchmark)
2.0
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.alpha"></a>

### quantkit.stats.alpha(returns, benchmark, risk_free=0.0, factor=None)

Compute Jensen’s alpha of `returns` against `benchmark`.

$$
\alpha_i = \overline{(r_i - r_f)} - \beta_i \, \overline{(r_b - r_f)}
$$

where $\beta_i$ is [`beta()`](#quantkit.stats.beta) estimated on the same pairwise complete rows and $r_f$ is the risk free rate per period.

* **Parameters:**
  **returns** — array-like
  : 1D or 2D asset returns. Pandas objects are inner-joined with the benchmark on their index, see [`quantkit.utils.align()`](../utils/index.md#quantkit.utils.align).

  **benchmark** — array-like
  : 1D benchmark returns, usually a market index.

  **risk_free** — float, optional
  : Risk free rate per period, in the same basis as the returns.

  **factor** — float, optional
  : Multiplies the result to change its basis, e.g. 12 to annualize a monthly alpha as Morningstar does.
* **Returns:**
  **alpha** — float or array-like
  : One value per column of `returns`. NaN whenever [`beta()`](#quantkit.stats.beta) is.

### References

<a id="re3afc2be5d20-1"></a>

[1]

[https://en.wikipedia.org/wiki/Jensen%27s_alpha](https://en.wikipedia.org/wiki/Jensen%27s_alpha)

### Examples

```pycon
>>> benchmark = np.array([0.5, -0.25, 0.75, -0.5])
>>> alpha(2 * benchmark + 0.125, benchmark)
0.125
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.correlation"></a>

### quantkit.stats.correlation(returns, benchmark)

Compute the Pearson correlation of `returns` with `benchmark`.

$$
\rho_i = \frac{cov(r_i, r_b)}{\sigma_{r_i} \, \sigma_{r_b}}
$$

estimated, for each column, on the rows where the column and the benchmark are both non-NaN.

* **Parameters:**
  **returns** — array-like
  : 1D or 2D asset returns. Pandas objects are inner-joined with the benchmark on their index, see [`quantkit.utils.align()`](../utils/index.md#quantkit.utils.align).

  **benchmark** — array-like
  : 1D benchmark returns, usually a market index.
* **Returns:**
  **correlation** — float or array-like
  : One value per column of `returns`, in `[-1, 1]`. NaN when either side has no variance or fewer than two complete rows are available.

### References

<a id="r1a9cddd7a8a7-1"></a>

[1]

[https://en.wikipedia.org/wiki/Pearson_correlation_coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)

### Examples

```pycon
>>> benchmark = np.array([0.5, -0.25, 0.75, -0.5])
>>> correlation(-benchmark, benchmark)
-1.0
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.r_squared"></a>

### quantkit.stats.r_squared(returns, benchmark)

Compute the coefficient of determination against `benchmark`.

The share of the variance of `returns` explained by the linear regression on the benchmark, which is the square of the Pearson [`correlation()`](#quantkit.stats.correlation):

$$
R^2_i = \rho_i^2
$$

* **Parameters:**
  **returns** — array-like
  : 1D or 2D asset returns. Pandas objects are inner-joined with the benchmark on their index, see [`quantkit.utils.align()`](../utils/index.md#quantkit.utils.align).

  **benchmark** — array-like
  : 1D benchmark returns, usually a market index.
* **Returns:**
  **r_squared** — float or array-like
  : One value per column of `returns`, in `[0, 1]`. NaN whenever [`correlation()`](#quantkit.stats.correlation) is.

### References

<a id="ra17ecad64dba-1"></a>

[1]

[https://en.wikipedia.org/wiki/Coefficient_of_determination](https://en.wikipedia.org/wiki/Coefficient_of_determination)

### Examples

```pycon
>>> benchmark = np.array([0.5, -0.25, 0.75, -0.5])
>>> r_squared(-benchmark, benchmark)
1.0
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.bull_beta"></a>

### quantkit.stats.bull_beta(returns, benchmark)

Compute the beta over the periods where the benchmark went up.

Morningstar’s Bull Beta: [`beta()`](#quantkit.stats.beta) restricted to the rows where `benchmark > 0`. Rows with a zero benchmark return belong neither to the bull nor to the bear sample.

* **Parameters:**
  **returns** — array-like
  : 1D or 2D asset returns. Pandas objects are inner-joined with the benchmark on their index, see [`quantkit.utils.align()`](../utils/index.md#quantkit.utils.align).

  **benchmark** — array-like
  : 1D benchmark returns, usually a market index.
* **Returns:**
  **bull_beta** — float or array-like
  : One value per column of `returns`. NaN when fewer than two rows have a positive benchmark or those rows have no variance.

### References

<a id="r0bc2fb490078-1"></a>

[1]

Morningstar, “Custom Calculation Data Points”, Bull Beta. [https://morningstardirect.morningstar.com/clientcomm/](https://morningstardirect.morningstar.com/clientcomm/) CustomCalculationDataPoints.pdf

### Examples

```pycon
>>> benchmark = np.array([0.5, -0.25, 0.75, -0.5])
>>> returns = np.array([1.0, -0.125, 1.5, -0.25])  # 2b up, 0.5b down
>>> bull_beta(returns, benchmark)
2.0
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.bear_beta"></a>

### quantkit.stats.bear_beta(returns, benchmark)

Compute the beta over the periods where the benchmark went down.

Morningstar’s Bear Beta: [`beta()`](#quantkit.stats.beta) restricted to the rows where `benchmark < 0`. Rows with a zero benchmark return belong neither to the bull nor to the bear sample.

* **Parameters:**
  **returns** — array-like
  : 1D or 2D asset returns. Pandas objects are inner-joined with the benchmark on their index, see [`quantkit.utils.align()`](../utils/index.md#quantkit.utils.align).

  **benchmark** — array-like
  : 1D benchmark returns, usually a market index.
* **Returns:**
  **bear_beta** — float or array-like
  : One value per column of `returns`. NaN when fewer than two rows have a negative benchmark or those rows have no variance.

### References

<a id="r3f0c386a9dd5-1"></a>

[1]

Morningstar, “Custom Calculation Data Points”, Bear Beta. [https://morningstardirect.morningstar.com/clientcomm/](https://morningstardirect.morningstar.com/clientcomm/) CustomCalculationDataPoints.pdf

### Examples

```pycon
>>> benchmark = np.array([0.5, -0.25, 0.75, -0.5])
>>> returns = np.array([1.0, -0.125, 1.5, -0.25])  # 2b up, 0.5b down
>>> bear_beta(returns, benchmark)
0.5
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._treynor"></a>

### quantkit.stats.\_treynor(col, bench, risk_free, factor)

Treynor ratio of one column: mean(col - rf) / beta(col, bench).

NaN whenever the beta cannot be used as a denominator: either it is itself NaN, or the column has no dispersion and its beta is zero up to rounding, in which case the ratio is undefined rather than infinite.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.treynor_ratio"></a>

### quantkit.stats.treynor_ratio(returns, benchmark, risk_free=0.0, factor=None)

Compute the Treynor ratio of `returns` against `benchmark`.

Morningstar’s arithmetic Treynor ratio is the mean excess return per unit of systematic risk

$$
T_i = \frac{\overline{(r_i - r_f)}}{\beta_i}
$$

where $\beta_i$ is [`beta()`](#quantkit.stats.beta) estimated on the same pairwise complete rows. It is [`sharpe_ratio()`](#quantkit.stats.sharpe_ratio) with the total risk of the asset, its standard deviation, replaced by its systematic risk, so it rewards a portfolio only for the market exposure it takes and ignores the diversifiable part of its volatility.

Morningstar annualizes the numerator before dividing; here `factor` does that, and since the numerator is linear it scales the whole ratio.

* **Parameters:**
  **returns** — array-like
  : 1D or 2D asset returns. Pandas objects are inner-joined with the benchmark on their index, see [`quantkit.utils.align()`](../utils/index.md#quantkit.utils.align).

  **benchmark** — array-like
  : 1D benchmark returns, usually a market index.

  **risk_free** — float, optional
  : Risk free rate per period, in the same basis as the returns.

  **factor** — float, optional
  : Multiplies the result to change its basis, e.g. 12 to annualize a monthly ratio as Morningstar does.
* **Returns:**
  **treynor_ratio** — float or array-like
  : One value per column of `returns`. NaN whenever [`beta()`](#quantkit.stats.beta) is, and also when the beta is zero, since the ratio is then undefined: the result is never infinite.
* **Raises:**
  ValueError
  : If `benchmark` is not 1D or the numpy lengths differ.

### References

<a id="rac687495fe3c-1"></a>

[1]

[https://en.wikipedia.org/wiki/Treynor_ratio](https://en.wikipedia.org/wiki/Treynor_ratio)

<a id="rac687495fe3c-2"></a>

[2]

Morningstar, “Custom Calculation Data Points”, Treynor Ratio. [https://morningstardirect.morningstar.com/clientcomm/](https://morningstardirect.morningstar.com/clientcomm/) CustomCalculationDataPoints.pdf

### Examples

```pycon
>>> benchmark = np.array([0.5, -0.25, 0.75, -0.5])
>>> returns = 2 * benchmark + 0.125  # beta 2, mean 0.375
>>> treynor_ratio(returns, benchmark)
0.1875
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._compound"></a>

### quantkit.stats.\_compound(arr)

Compound (geometrically linked) return of `arr`: prod(1 + r) - 1.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._capture"></a>

### quantkit.stats.\_capture(col, bench, mask)

Compound return of `col` over that of `bench` on `mask` rows.

NaN when the mask selects no row at all, so that the empty product does not silently divide 0 by 0, and when the benchmark compounds to exactly 0 over the selected rows.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._up_capture"></a>

### quantkit.stats.\_up_capture(col, bench)

Capture ratio over the rows where the benchmark did not fall.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._down_capture"></a>

### quantkit.stats.\_down_capture(col, bench)

Capture ratio over the rows where the benchmark fell.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._overall_capture"></a>

### quantkit.stats.\_overall_capture(col, bench)

Up capture over down capture, NaN when the latter is zero.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._batting_average"></a>

### quantkit.stats.\_batting_average(col, bench)

Fraction of the rows where `col` is at or above `bench`.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.up_capture"></a>

### quantkit.stats.up_capture(returns, benchmark)

Compute the up capture ratio against `benchmark`.

Morningstar’s Up Capture Ratio: how much of the benchmark’s rise the asset captured [\[1\]](#r042edd96dbbf-1). Over the rows where the benchmark did not fall, the compound return of the asset divided by the compound return of the benchmark:

$$
UC_i = \frac{\prod_{r_{b,t} \geq 0} (1 + r_{i,t}) - 1}
            {\prod_{r_{b,t} \geq 0} (1 + r_{b,t}) - 1}
$$

A period with a zero benchmark return counts as an up period, matching [`up_period_percent()`](#quantkit.stats.up_period_percent). The returns are compounded, not added, so the ratio of a single up period is the plain ratio of the two returns but the ratio of several is not.

The output is in parts per unit (1.10, not 110), like the rest of the library with `relative=True`; Morningstar quotes it as a percentage.

* **Parameters:**
  **returns** — array-like
  : 1D or 2D asset returns. Pandas objects are inner-joined with the benchmark on their index, see [`quantkit.utils.align()`](../utils/index.md#quantkit.utils.align).

  **benchmark** — array-like
  : 1D benchmark returns, usually a market index.
* **Returns:**
  **up_capture** — float or array-like
  : One value per column of `returns`. NaN when no complete row has a benchmark at or above zero, or when the benchmark compounds to exactly 0 over those rows.
* **Raises:**
  ValueError
  : If `benchmark` is not 1D or the numpy lengths differ.

### References

<a id="r042edd96dbbf-1"></a>

[1]

Morningstar, “Custom Calculation Data Points”, Up Capture Ratio. [https://morningstardirect.morningstar.com/clientcomm/](https://morningstardirect.morningstar.com/clientcomm/) CustomCalculationDataPoints.pdf

### Examples

```pycon
>>> benchmark = np.array([0.5, -0.5])
>>> returns = np.array([1.0, -0.25])  # 2b up, 0.5b down
>>> up_capture(returns, benchmark)
2.0
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.down_capture"></a>

### quantkit.stats.down_capture(returns, benchmark)

Compute the down capture ratio against `benchmark`.

Morningstar’s Down Capture Ratio: how much of the benchmark’s fall the asset suffered [\[1\]](#ra3843642031d-1). Same as [`up_capture()`](#quantkit.stats.up_capture) over the rows where the benchmark fell:

$$
DC_i = \frac{\prod_{r_{b,t} < 0} (1 + r_{i,t}) - 1}
            {\prod_{r_{b,t} < 0} (1 + r_{b,t}) - 1}
$$

Both compound returns are usually negative, so the ratio is usually positive: above 1 means the asset fell more than the benchmark, below 1 that it fell less, and a negative value that it rose while the benchmark fell. A period with a zero benchmark return is an up period and does not take part.

The output is in parts per unit (1.10, not 110), like the rest of the library with `relative=True`; Morningstar quotes it as a percentage.

* **Parameters:**
  **returns** — array-like
  : 1D or 2D asset returns. Pandas objects are inner-joined with the benchmark on their index, see [`quantkit.utils.align()`](../utils/index.md#quantkit.utils.align).

  **benchmark** — array-like
  : 1D benchmark returns, usually a market index.
* **Returns:**
  **down_capture** — float or array-like
  : One value per column of `returns`. NaN when no complete row has a negative benchmark, or when the benchmark compounds to exactly 0 over those rows.
* **Raises:**
  ValueError
  : If `benchmark` is not 1D or the numpy lengths differ.

### References

<a id="ra3843642031d-1"></a>

[1]

Morningstar, “Custom Calculation Data Points”, Down Capture Ratio. [https://morningstardirect.morningstar.com/clientcomm/](https://morningstardirect.morningstar.com/clientcomm/) CustomCalculationDataPoints.pdf

### Examples

```pycon
>>> benchmark = np.array([0.5, -0.5])
>>> returns = np.array([1.0, -0.25])  # 2b up, 0.5b down
>>> down_capture(returns, benchmark)
0.5
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.overall_capture"></a>

### quantkit.stats.overall_capture(returns, benchmark)

Compute the overall capture ratio against `benchmark`.

Morningstar’s Overall Capture Ratio, the up capture over the down capture [\[1\]](#ra6b42cf54a3f-1):

$$
OC_i = \frac{UC_i}{DC_i}
$$

Above 1 the asset takes more of the benchmark’s upside than of its downside, below 1 the other way round.

The output is in parts per unit (1.10, not 110), like the rest of the library with `relative=True`; Morningstar quotes it as a percentage.

* **Parameters:**
  **returns** — array-like
  : 1D or 2D asset returns. Pandas objects are inner-joined with the benchmark on their index, see [`quantkit.utils.align()`](../utils/index.md#quantkit.utils.align).

  **benchmark** — array-like
  : 1D benchmark returns, usually a market index.
* **Returns:**
  **overall_capture** — float or array-like
  : One value per column of `returns`. NaN whenever [`up_capture()`](#quantkit.stats.up_capture) or [`down_capture()`](#quantkit.stats.down_capture) is, and when the down capture is 0.
* **Raises:**
  ValueError
  : If `benchmark` is not 1D or the numpy lengths differ.

### References

<a id="ra6b42cf54a3f-1"></a>

[1]

Morningstar, “Custom Calculation Data Points”, Overall Capture Ratio. [https://morningstardirect.morningstar.com/clientcomm/](https://morningstardirect.morningstar.com/clientcomm/) CustomCalculationDataPoints.pdf

### Examples

```pycon
>>> benchmark = np.array([0.5, -0.5])
>>> returns = np.array([1.0, -0.25])  # up capture 2, down capture 0.5
>>> overall_capture(returns, benchmark)
4.0
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.batting_average"></a>

### quantkit.stats.batting_average(returns, benchmark)

Compute the fraction of periods that beat or match `benchmark`.

Morningstar’s Batting Average: the number of periods where the asset return is greater than or equal to the benchmark return over the number of periods compared [\[1\]](#r52f5b6887d57-1). A tie counts as a hit, and only the rows where both sides are non-NaN are compared.

$$
BA_i = \frac{\#\{t : r_{i,t} \geq r_{b,t}\}}{\#\{t\}}
$$

The size of the win or of the loss does not matter, only its sign. The output is in parts per unit (0.75, not 75), like the rest of the library with `relative=True`; Morningstar quotes it as a percentage.

* **Parameters:**
  **returns** — array-like
  : 1D or 2D asset returns. Pandas objects are inner-joined with the benchmark on their index, see [`quantkit.utils.align()`](../utils/index.md#quantkit.utils.align).

  **benchmark** — array-like
  : 1D benchmark returns, usually a market index.
* **Returns:**
  **batting_average** — float or array-like
  : Fraction in \[0, 1\], one value per column of `returns`. NaN when there is no complete row to compare.
* **Raises:**
  ValueError
  : If `benchmark` is not 1D or the numpy lengths differ.

### References

<a id="r52f5b6887d57-1"></a>

[1]

Morningstar, “Custom Calculation Data Points”, Batting Average. [https://morningstardirect.morningstar.com/clientcomm/](https://morningstardirect.morningstar.com/clientcomm/) CustomCalculationDataPoints.pdf

### Examples

```pycon
>>> benchmark = np.array([0.10, -0.10, 0.05, 0.00])
>>> returns = np.array([0.20, -0.20, 0.05, 0.10])  # win, loss, tie, win
>>> batting_average(returns, benchmark)
0.75
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.sterling_ratio"></a>

### quantkit.stats.sterling_ratio(returns, periods_per_year=BYEAR, excess=0.1)

Calculate Morningstar’s Sterling ratio.

Return earned per unit of average drawdown, cushioned by an excess risk figure of 10% [\[1\]](#r1c2a808ac5df-1). It is the annualized return divided by the absolute average drawdown of the price path implied by `returns` plus `excess`:

$$
\text{Sterling} = \frac{R_{\text{annual}}}{|AvgDD| + \text{excess}}
$$

The price path is the one [`calmar_ratio()`](#quantkit.stats.calmar_ratio) uses, `cum_returns(returns, first_price=1)` preceded by the starting capital itself, so both ratios agree on what a drawdown is and a loss on the very first return counts. That leading price is an observation like any other, so `n` returns give `n + 1` prices and [`average_drawdown()`](#quantkit.stats.average_drawdown) cuts its yearly blocks over those `n + 1` observations.

The default excess keeps the denominator positive, so the ratio is defined even without a drawdown. With `excess=0` and a price path that never falls the denominator is zero and the result is NaN; over a single block, where the average drawdown is the maximum drawdown, that case reproduces [`calmar_ratio()`](#quantkit.stats.calmar_ratio).

* **Parameters:**
  **returns** — array-like
  : Arithmetic returns series, 1-d or 2-d (columns are reduced independently).

  **periods_per_year** — int, optional
  : Number of observations that make up a year, passed to [`annualized_return()`](#quantkit.stats.annualized_return) and to [`average_drawdown()`](#quantkit.stats.average_drawdown), which uses it as its block length. `BYEAR` (business days) by default.

  **excess** — float, optional
  : Risk figure added to the absolute average drawdown, 0.10 (Morningstar’s 10%) by default.
* **Returns:**
  **out** — float or 1d-reduced-array
  : Sterling ratio of each column. NaN when the denominator is zero or the column has no valid return.

### References

<a id="r1c2a808ac5df-1"></a>

[1]

Morningstar, “Custom Calculation Data Points” (October 2016), Sterling Ratio: the compounded annual return over the average maximum drawdown minus 10%, taken in absolute value as a positive risk figure. [https://morningstardirect.morningstar.com/clientcomm/](https://morningstardirect.morningstar.com/clientcomm/) customcalculations.pdf

<a id="r1c2a808ac5df-2"></a>

[2]

[https://en.wikipedia.org/wiki/Sterling_ratio](https://en.wikipedia.org/wiki/Sterling_ratio)

### Examples

Losing half the capital in a period that is half a year long: the annualized return is $0.5^2 - 1 = -0.75$ and the price path `[1, 0.5]` is one full block with a drawdown of -0.5.

```pycon
>>> sterling_ratio(np.array([-0.5]), periods_per_year=2, excess=0.0)
-1.5
```

The default excess softens the same denominator to 0.6.

```pycon
>>> sterling_ratio(np.array([-0.5]), periods_per_year=2)
-1.25
```

Without a drawdown the excess is the whole denominator: the prices `[1, 1.25, 1.25]` never fall, so 0.25 / 0.10 = 2.5.

```pycon
>>> sterling_ratio(np.array([0.25, 0.0]), periods_per_year=2)
2.5
```

```pycon
>>> sterling_ratio(np.array([0.25, 0.0]), periods_per_year=2, excess=0.0)
nan
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._active_return"></a>

### quantkit.stats.\_active_return(col, bench)

Active return `col - bench` and its standard deviation, `ddof=1`.

A fund that moves exactly with its benchmark has a constant active return, yet subtracting two series of rounded decimals leaves a spread of a few ulp around that constant, so an exact equality test does not recognise it. A deviation below the rounding noise of the inputs is reported as exactly zero rather than as dispersion. `col` must hold at least two observations.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._tracking_error"></a>

### quantkit.stats.\_tracking_error(col, bench, factor)

Sample standard deviation of the active return, times `factor`.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats._information_ratio"></a>

### quantkit.stats.\_information_ratio(col, bench, factor)

Mean active return over its standard deviation, times `factor`.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.tracking_error"></a>

### quantkit.stats.tracking_error(returns, benchmark, factor=None)

Calculate the tracking error against `benchmark`.

Standard deviation of the active return $r_t - b_t$, the amount by which the fund deviates from its benchmark [\[1\]](#rfc760890dc9a-1) [\[2\]](#rfc760890dc9a-2):

$$
TE = \sqrt{\frac{\sum_{t=1}^{N}
           \left( (r_t - b_t) - \overline{(r - b)} \right)^2}{N - 1}}
$$

The deviation uses `ddof=1` and, for each column, only the rows where the column and the benchmark are both non-NaN (pairwise complete). A fund that moves exactly with its benchmark, `r = b + c` for a constant `c`, has a zero tracking error.

* **Parameters:**
  **returns** — array-like
  : 1D or 2D asset returns. Pandas objects are inner-joined with the benchmark on their index, see [`quantkit.utils.align()`](../utils/index.md#quantkit.utils.align).

  **benchmark** — array-like
  : 1D benchmark returns, usually the index the fund is measured against.

  **factor** — float, optional
  : Multiplies the result. Annualizing a standard deviation requires the caller to pass the square root of the number of periods in a year, e.g. `np.sqrt(BYEAR)` for daily returns.
* **Returns:**
  **tracking_error** — float or array-like
  : Float for 1d input, one value per column for 2d input. NaN when fewer than two complete rows are available.
* **Raises:**
  ValueError
  : If `benchmark` is not 1D or the numpy lengths differ.

### References

<a id="rfc760890dc9a-1"></a>

[1]

Morningstar, “Custom Calculation Data Points”, Tracking Error: the standard deviation of (P - B). [https://morningstardirect.morningstar.com/clientcomm/](https://morningstardirect.morningstar.com/clientcomm/) CustomCalculationDataPoints.pdf

<a id="rfc760890dc9a-2"></a>

[2]

[https://en.wikipedia.org/wiki/Tracking_error](https://en.wikipedia.org/wiki/Tracking_error)

### Examples

```pycon
>>> benchmark = np.array([0.01, 0.01, 0.02])
>>> tracking_error(np.array([0.02, 0.00, 0.05]), benchmark)
0.02
```

```pycon
>>> tracking_error(benchmark + 0.01, benchmark)
0.0
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.stats.information_ratio"></a>

### quantkit.stats.information_ratio(returns, benchmark, factor=None)

Calculate the arithmetic information ratio against `benchmark`.

Mean active return per unit of [`tracking_error()`](#quantkit.stats.tracking_error) [\[1\]](#r3fa14cff6139-1) [\[2\]](#r3fa14cff6139-2):

$$
IR = \frac{\overline{(r - b)}}{\sigma_{r - b}}
$$

Both moments are taken, for each column, over the rows where the column and the benchmark are both non-NaN, and the deviation uses `ddof=1`. A fund that moves exactly with its benchmark has a zero tracking error, so its ratio is undefined and comes out as NaN: a constant positive active return is not an infinitely good result, it is a result the ratio cannot rank.

* **Parameters:**
  **returns** — array-like
  : 1D or 2D asset returns. Pandas objects are inner-joined with the benchmark on their index, see [`quantkit.utils.align()`](../utils/index.md#quantkit.utils.align).

  **benchmark** — array-like
  : 1D benchmark returns, usually the index the fund is measured against.

  **factor** — float, optional
  : Multiplies the result. Annualizing the ratio requires the caller to pass the square root of the number of periods in a year, e.g. `np.sqrt(BYEAR)` for daily returns, since annualizing a standard deviation is what the square root does.
* **Returns:**
  **information_ratio** — float or array-like
  : Float for 1d input, one value per column for 2d input. NaN when the tracking error is zero (zero denominator) or fewer than two complete rows are available.
* **Raises:**
  ValueError
  : If `benchmark` is not 1D or the numpy lengths differ.

### References

<a id="r3fa14cff6139-1"></a>

[1]

Morningstar, “Custom Calculation Data Points”, Information Ratio (arithmetic). [https://morningstardirect.morningstar.com/clientcomm/](https://morningstardirect.morningstar.com/clientcomm/) CustomCalculationDataPoints.pdf

<a id="r3fa14cff6139-2"></a>

[2]

[https://en.wikipedia.org/wiki/Information_ratio](https://en.wikipedia.org/wiki/Information_ratio)

### Examples

```pycon
>>> benchmark = np.array([0.01, 0.01, 0.02])
>>> information_ratio(np.array([0.02, 0.00, 0.05]), benchmark)
0.5
```

```pycon
>>> information_ratio(benchmark + 0.01, benchmark)
nan
```

<!-- !! processed by numpydoc !! -->
