<a id="module-quantkit.expanding"></a>

<a id="quantkit-expanding"></a>

# quantkit.expanding

Financial statistics over expanding windows.

Inputs and outputs have the same NumPy or pandas container and axes. Windows end at the current row; no future observations are used. Missing values count toward window width but not `min_periods`. See the window statistics guide for alignment and missing-value rules.

<!-- !! processed by numpydoc !! -->

<a id="functions"></a>

## Functions

| [`total_returns`](#quantkit.expanding.total_returns)(prices\[, min_periods, factor, relative\])                  | Total returns over expanding windows.                  |
|------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| [`volatility`](#quantkit.expanding.volatility)(returns\[, min_periods, factor, ddof\])                           | Volatility over expanding windows.                     |
| [`drawdown`](#quantkit.expanding.drawdown)(prices\[, relative, out, min_periods\])                               | Drawdown over expanding windows.                       |
| [`max_drawdown`](#quantkit.expanding.max_drawdown)(prices\[, min_periods, relative\])                            | Max drawdown over expanding windows.                   |
| [`sharpe_ratio`](#quantkit.expanding.sharpe_ratio)(returns, risk_free\[, min_periods, factor\])                  | Sharpe ratio over expanding windows.                   |
| [`value_at_risk`](#quantkit.expanding.value_at_risk)(returns\[, min_periods, confidence\])                       | Value at risk over expanding windows.                  |
| [`max_drawup`](#quantkit.expanding.max_drawup)(prices\[, min_periods, relative\])                                | Max drawup over expanding windows.                     |
| [`annualized_return`](#quantkit.expanding.annualized_return)(returns\[, min_periods, periods_per_year\])         | Annualized return over expanding windows.              |
| [`calmar_ratio`](#quantkit.expanding.calmar_ratio)(returns\[, min_periods, periods_per_year\])                   | Calmar ratio over expanding windows.                   |
| [`average_gain`](#quantkit.expanding.average_gain)(returns\[, min_periods, method\])                             | Average gain over expanding windows.                   |
| [`average_loss`](#quantkit.expanding.average_loss)(returns\[, min_periods, method\])                             | Average loss over expanding windows.                   |
| [`gain_loss_ratio`](#quantkit.expanding.gain_loss_ratio)(returns\[, min_periods\])                               | Gain loss ratio over expanding windows.                |
| [`up_period_percent`](#quantkit.expanding.up_period_percent)(returns\[, min_periods\])                           | Up period percent over expanding windows.              |
| [`down_period_percent`](#quantkit.expanding.down_period_percent)(returns\[, min_periods\])                       | Down period percent over expanding windows.            |
| [`downside_deviation`](#quantkit.expanding.downside_deviation)(returns\[, min_periods, mar, factor\])            | Downside deviation over expanding windows.             |
| [`upside_deviation`](#quantkit.expanding.upside_deviation)(returns\[, min_periods, mar, factor\])                | Upside deviation over expanding windows.               |
| [`kappa`](#quantkit.expanding.kappa)(returns\[, min_periods, mar, order, factor\])                               | Kappa over expanding windows.                          |
| [`omega_ratio`](#quantkit.expanding.omega_ratio)(returns\[, min_periods, mar\])                                  | Omega ratio over expanding windows.                    |
| [`sortino_ratio`](#quantkit.expanding.sortino_ratio)(returns\[, min_periods, mar, factor\])                      | Sortino ratio over expanding windows.                  |
| [`max_drawdown_peak`](#quantkit.expanding.max_drawdown_peak)(prices\[, min_periods\])                            | Max drawdown peak over expanding windows.              |
| [`max_drawdown_valley`](#quantkit.expanding.max_drawdown_valley)(prices\[, min_periods\])                        | Max drawdown valley over expanding windows.            |
| [`max_drawdown_recovery`](#quantkit.expanding.max_drawdown_recovery)(prices\[, min_periods\])                    | Max drawdown recovery over expanding windows.          |
| [`max_drawdown_duration`](#quantkit.expanding.max_drawdown_duration)(prices\[, min_periods\])                    | Max drawdown duration over expanding windows.          |
| [`max_drawdown_recovery_duration`](#quantkit.expanding.max_drawdown_recovery_duration)(prices\[, min_periods\])  | Max drawdown recovery duration over expanding windows. |
| [`longest_drawdown_duration`](#quantkit.expanding.longest_drawdown_duration)(prices\[, min_periods\])            | Longest drawdown duration over expanding windows.      |
| [`average_drawdown`](#quantkit.expanding.average_drawdown)(prices\[, min_periods, periods_per_year\])            | Average drawdown over expanding windows.               |
| [`beta`](#quantkit.expanding.beta)(returns, benchmark\[, min_periods\])                                          | Beta over expanding windows.                           |
| [`alpha`](#quantkit.expanding.alpha)(returns, benchmark\[, min_periods, risk_free, factor\])                     | Alpha over expanding windows.                          |
| [`correlation`](#quantkit.expanding.correlation)(returns, benchmark\[, min_periods\])                            | Correlation over expanding windows.                    |
| [`r_squared`](#quantkit.expanding.r_squared)(returns, benchmark\[, min_periods\])                                | R squared over expanding windows.                      |
| [`bull_beta`](#quantkit.expanding.bull_beta)(returns, benchmark\[, min_periods\])                                | Bull beta over expanding windows.                      |
| [`bear_beta`](#quantkit.expanding.bear_beta)(returns, benchmark\[, min_periods\])                                | Bear beta over expanding windows.                      |
| [`treynor_ratio`](#quantkit.expanding.treynor_ratio)(returns, benchmark\[, min_periods, ...\])                   | Treynor ratio over expanding windows.                  |
| [`up_capture`](#quantkit.expanding.up_capture)(returns, benchmark\[, min_periods\])                              | Up capture over expanding windows.                     |
| [`down_capture`](#quantkit.expanding.down_capture)(returns, benchmark\[, min_periods\])                          | Down capture over expanding windows.                   |
| [`overall_capture`](#quantkit.expanding.overall_capture)(returns, benchmark\[, min_periods\])                    | Overall capture over expanding windows.                |
| [`batting_average`](#quantkit.expanding.batting_average)(returns, benchmark\[, min_periods\])                    | Batting average over expanding windows.                |
| [`sterling_ratio`](#quantkit.expanding.sterling_ratio)(returns\[, min_periods, ...\])                            | Sterling ratio over expanding windows.                 |
| [`tracking_error`](#quantkit.expanding.tracking_error)(returns, benchmark\[, min_periods, factor\])              | Tracking error over expanding windows.                 |
| [`information_ratio`](#quantkit.expanding.information_ratio)(returns, benchmark\[, min_periods, ...\])           | Information ratio over expanding windows.              |
| [`drawup`](#quantkit.expanding.drawup)(prices\[, relative, out, min_periods\])                                   | Drawup over expanding windows.                         |
| [`expected_shortfall`](#quantkit.expanding.expected_shortfall)(returns\[, min_periods, confidence\])             | Calculate expected shortfall over expanding windows.   |
| [`conditional_drawdown_at_risk`](#quantkit.expanding.conditional_drawdown_at_risk)(wealth\[, min_periods, ...\]) | Conditional drawdown at risk over expanding windows.   |

<a id="module-contents"></a>

## Module Contents

<a id="quantkit.expanding.total_returns"></a>

### quantkit.expanding.total_returns(prices, min_periods=1, factor=None, relative=True)

Total returns over expanding windows.

* **Parameters:**
  **prices** — array-like
  : Observed prices or wealth, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **factor** — float, optional
  : As in [`quantkit.stats.total_returns()`](../stats/index.md#quantkit.stats.total_returns).

  **relative** — bool, optional
  : As in [`quantkit.stats.total_returns()`](../stats/index.md#quantkit.stats.total_returns).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.volatility"></a>

### quantkit.expanding.volatility(returns, min_periods=1, factor=None, ddof=1)

Volatility over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **factor** — float, optional
  : As in [`quantkit.stats.volatility()`](../stats/index.md#quantkit.stats.volatility).

  **ddof** — int, optional
  : As in [`quantkit.stats.volatility()`](../stats/index.md#quantkit.stats.volatility).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.drawdown"></a>

### quantkit.expanding.drawdown(prices, relative=True, out=None, \*, min_periods=1)

Drawdown over expanding windows.

* **Parameters:**
  **prices** — array-like
  : Observed prices or wealth, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **relative** — bool, optional
  : As in [`quantkit.stats.drawdown()`](../stats/index.md#quantkit.stats.drawdown).

  **out** — numpy.ndarray, optional
  : Floating output buffer with the same shape as prices.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

### Notes

A missing current price produces NaN. Zero denominators produce NaN.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.max_drawdown"></a>

### quantkit.expanding.max_drawdown(prices, min_periods=1, relative=True)

Max drawdown over expanding windows.

* **Parameters:**
  **prices** — array-like
  : Observed prices or wealth, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **relative** — bool, optional
  : As in [`quantkit.stats.max_drawdown()`](../stats/index.md#quantkit.stats.max_drawdown).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

### Notes

Both extrema belong to the current window in chronological order.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.sharpe_ratio"></a>

### quantkit.expanding.sharpe_ratio(returns, risk_free, min_periods=1, factor=np.sqrt(BYEAR))

Sharpe ratio over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **risk_free** — float or array-like
  : Risk-free returns per observation; Series align by index.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **factor** — float, optional
  : As in [`quantkit.stats.sharpe_ratio()`](../stats/index.md#quantkit.stats.sharpe_ratio).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.value_at_risk"></a>

### quantkit.expanding.value_at_risk(returns, min_periods=1, confidence=0.95)

Value at risk over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **confidence** — float, optional
  : As in [`quantkit.stats.value_at_risk()`](../stats/index.md#quantkit.stats.value_at_risk).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.max_drawup"></a>

### quantkit.expanding.max_drawup(prices, min_periods=1, relative=True)

Max drawup over expanding windows.

* **Parameters:**
  **prices** — array-like
  : Observed prices or wealth, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **relative** — bool, optional
  : As in [`quantkit.stats.max_drawup()`](../stats/index.md#quantkit.stats.max_drawup).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

### Notes

Both extrema belong to the current window in chronological order.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.annualized_return"></a>

### quantkit.expanding.annualized_return(returns, min_periods=1, periods_per_year=BYEAR)

Annualized return over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **periods_per_year** — int, optional
  : As in [`quantkit.stats.annualized_return()`](../stats/index.md#quantkit.stats.annualized_return).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.calmar_ratio"></a>

### quantkit.expanding.calmar_ratio(returns, min_periods=1, periods_per_year=BYEAR)

Calmar ratio over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **periods_per_year** — int, optional
  : As in [`quantkit.stats.calmar_ratio()`](../stats/index.md#quantkit.stats.calmar_ratio).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.average_gain"></a>

### quantkit.expanding.average_gain(returns, min_periods=1, method='arith')

Average gain over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **method** — str, optional
  : As in [`quantkit.stats.average_gain()`](../stats/index.md#quantkit.stats.average_gain).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.average_loss"></a>

### quantkit.expanding.average_loss(returns, min_periods=1, method='arith')

Average loss over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **method** — str, optional
  : As in [`quantkit.stats.average_loss()`](../stats/index.md#quantkit.stats.average_loss).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.gain_loss_ratio"></a>

### quantkit.expanding.gain_loss_ratio(returns, min_periods=1)

Gain loss ratio over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.up_period_percent"></a>

### quantkit.expanding.up_period_percent(returns, min_periods=1)

Up period percent over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.down_period_percent"></a>

### quantkit.expanding.down_period_percent(returns, min_periods=1)

Down period percent over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.downside_deviation"></a>

### quantkit.expanding.downside_deviation(returns, min_periods=1, mar=0.0, factor=None)

Downside deviation over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **mar** — float, optional
  : As in [`quantkit.stats.downside_deviation()`](../stats/index.md#quantkit.stats.downside_deviation).

  **factor** — float, optional
  : As in [`quantkit.stats.downside_deviation()`](../stats/index.md#quantkit.stats.downside_deviation).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.upside_deviation"></a>

### quantkit.expanding.upside_deviation(returns, min_periods=1, mar=0.0, factor=None)

Upside deviation over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **mar** — float, optional
  : As in [`quantkit.stats.upside_deviation()`](../stats/index.md#quantkit.stats.upside_deviation).

  **factor** — float, optional
  : As in [`quantkit.stats.upside_deviation()`](../stats/index.md#quantkit.stats.upside_deviation).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.kappa"></a>

### quantkit.expanding.kappa(returns, min_periods=1, mar=0.0, order=2, factor=None)

Kappa over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **mar** — float, optional
  : As in [`quantkit.stats.kappa()`](../stats/index.md#quantkit.stats.kappa).

  **order** — int, optional
  : As in [`quantkit.stats.kappa()`](../stats/index.md#quantkit.stats.kappa).

  **factor** — float, optional
  : As in [`quantkit.stats.kappa()`](../stats/index.md#quantkit.stats.kappa).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.omega_ratio"></a>

### quantkit.expanding.omega_ratio(returns, min_periods=1, mar=0.0)

Omega ratio over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **mar** — float, optional
  : As in [`quantkit.stats.omega_ratio()`](../stats/index.md#quantkit.stats.omega_ratio).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.sortino_ratio"></a>

### quantkit.expanding.sortino_ratio(returns, min_periods=1, mar=0.0, factor=None)

Sortino ratio over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **mar** — float, optional
  : As in [`quantkit.stats.sortino_ratio()`](../stats/index.md#quantkit.stats.sortino_ratio).

  **factor** — float, optional
  : As in [`quantkit.stats.sortino_ratio()`](../stats/index.md#quantkit.stats.sortino_ratio).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.max_drawdown_peak"></a>

### quantkit.expanding.max_drawdown_peak(prices, min_periods=1)

Max drawdown peak over expanding windows.

* **Parameters:**
  **prices** — array-like
  : Observed prices or wealth, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

### Notes

Results are original pandas index labels or absolute NumPy row positions. Recovery is only reported once it has been observed.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.max_drawdown_valley"></a>

### quantkit.expanding.max_drawdown_valley(prices, min_periods=1)

Max drawdown valley over expanding windows.

* **Parameters:**
  **prices** — array-like
  : Observed prices or wealth, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

### Notes

Results are original pandas index labels or absolute NumPy row positions. Recovery is only reported once it has been observed.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.max_drawdown_recovery"></a>

### quantkit.expanding.max_drawdown_recovery(prices, min_periods=1)

Max drawdown recovery over expanding windows.

* **Parameters:**
  **prices** — array-like
  : Observed prices or wealth, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

### Notes

Results are original pandas index labels or absolute NumPy row positions. Recovery is only reported once it has been observed.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.max_drawdown_duration"></a>

### quantkit.expanding.max_drawdown_duration(prices, min_periods=1)

Max drawdown duration over expanding windows.

* **Parameters:**
  **prices** — array-like
  : Observed prices or wealth, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.max_drawdown_recovery_duration"></a>

### quantkit.expanding.max_drawdown_recovery_duration(prices, min_periods=1)

Max drawdown recovery duration over expanding windows.

* **Parameters:**
  **prices** — array-like
  : Observed prices or wealth, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.longest_drawdown_duration"></a>

### quantkit.expanding.longest_drawdown_duration(prices, min_periods=1)

Longest drawdown duration over expanding windows.

* **Parameters:**
  **prices** — array-like
  : Observed prices or wealth, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.average_drawdown"></a>

### quantkit.expanding.average_drawdown(prices, min_periods=1, periods_per_year=BYEAR)

Average drawdown over expanding windows.

* **Parameters:**
  **prices** — array-like
  : Observed prices or wealth, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **periods_per_year** — int, optional
  : As in [`quantkit.stats.average_drawdown()`](../stats/index.md#quantkit.stats.average_drawdown).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.beta"></a>

### quantkit.expanding.beta(returns, benchmark, min_periods=1)

Beta over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.alpha"></a>

### quantkit.expanding.alpha(returns, benchmark, min_periods=1, risk_free=0.0, factor=None)

Alpha over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **risk_free** — float, optional
  : As in [`quantkit.stats.alpha()`](../stats/index.md#quantkit.stats.alpha).

  **factor** — float, optional
  : As in [`quantkit.stats.alpha()`](../stats/index.md#quantkit.stats.alpha).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.correlation"></a>

### quantkit.expanding.correlation(returns, benchmark, min_periods=1)

Correlation over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.r_squared"></a>

### quantkit.expanding.r_squared(returns, benchmark, min_periods=1)

R squared over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.bull_beta"></a>

### quantkit.expanding.bull_beta(returns, benchmark, min_periods=1)

Bull beta over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.bear_beta"></a>

### quantkit.expanding.bear_beta(returns, benchmark, min_periods=1)

Bear beta over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.treynor_ratio"></a>

### quantkit.expanding.treynor_ratio(returns, benchmark, min_periods=1, risk_free=0.0, factor=None)

Treynor ratio over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **risk_free** — float, optional
  : As in [`quantkit.stats.treynor_ratio()`](../stats/index.md#quantkit.stats.treynor_ratio).

  **factor** — float, optional
  : As in [`quantkit.stats.treynor_ratio()`](../stats/index.md#quantkit.stats.treynor_ratio).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.up_capture"></a>

### quantkit.expanding.up_capture(returns, benchmark, min_periods=1)

Up capture over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.down_capture"></a>

### quantkit.expanding.down_capture(returns, benchmark, min_periods=1)

Down capture over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.overall_capture"></a>

### quantkit.expanding.overall_capture(returns, benchmark, min_periods=1)

Overall capture over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.batting_average"></a>

### quantkit.expanding.batting_average(returns, benchmark, min_periods=1)

Batting average over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.sterling_ratio"></a>

### quantkit.expanding.sterling_ratio(returns, min_periods=1, periods_per_year=BYEAR, excess=0.1)

Sterling ratio over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **periods_per_year** — int, optional
  : As in [`quantkit.stats.sterling_ratio()`](../stats/index.md#quantkit.stats.sterling_ratio).

  **excess** — float, optional
  : As in [`quantkit.stats.sterling_ratio()`](../stats/index.md#quantkit.stats.sterling_ratio).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.tracking_error"></a>

### quantkit.expanding.tracking_error(returns, benchmark, min_periods=1, factor=None)

Tracking error over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **factor** — float, optional
  : As in [`quantkit.stats.tracking_error()`](../stats/index.md#quantkit.stats.tracking_error).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.information_ratio"></a>

### quantkit.expanding.information_ratio(returns, benchmark, min_periods=1, factor=None)

Information ratio over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **factor** — float, optional
  : As in [`quantkit.stats.information_ratio()`](../stats/index.md#quantkit.stats.information_ratio).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.drawup"></a>

### quantkit.expanding.drawup(prices, relative=True, out=None, \*, min_periods=1)

Drawup over expanding windows.

* **Parameters:**
  **prices** — array-like
  : Observed prices or wealth, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **relative** — bool, optional
  : As in [`quantkit.expanding.drawup()`](#quantkit.expanding.drawup).

  **out** — numpy.ndarray, optional
  : Floating output buffer with the same shape as prices.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

### Notes

A missing current price produces NaN. Zero denominators produce NaN.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.expected_shortfall"></a>

### quantkit.expanding.expected_shortfall(returns, min_periods=1, confidence=0.95)

Calculate expected shortfall over expanding windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **confidence** — float, optional
  : As in [`quantkit.portfolio.tail_risk.expected_shortfall()`](../portfolio/tail_risk/index.md#quantkit.portfolio.tail_risk.expected_shortfall).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

### Notes

NaN within a window propagates, following the scalar tail-risk function.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.conditional_drawdown_at_risk"></a>

### quantkit.expanding.conditional_drawdown_at_risk(wealth, min_periods=1, confidence=0.95)

Conditional drawdown at risk over expanding windows.

* **Parameters:**
  **wealth** — array-like
  : Observed prices or wealth, with time along axis 0.

  **min_periods** — int, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats).

  **confidence** — float, optional
  : Tail confidence level, as in the scalar CDaR function.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

### Notes

NaN within a window propagates, following the scalar tail-risk function.

<!-- !! processed by numpydoc !! -->
