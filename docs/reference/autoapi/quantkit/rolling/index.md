<a id="module-quantkit.rolling"></a>

<a id="quantkit-rolling"></a>

# quantkit.rolling

Financial statistics over rolling windows.

Inputs and outputs have the same NumPy or pandas container and axes. Windows end at the current row; no future observations are used. Missing values count toward window width but not `min_periods`. See the window statistics guide for alignment and missing-value rules.

<!-- !! processed by numpydoc !! -->

<a id="functions"></a>

## Functions

| [`total_returns`](#quantkit.rolling.total_returns)(prices\[, window, min_periods, factor, ...\])              | Total returns over rolling windows.                  |
|---------------------------------------------------------------------------------------------------------------|------------------------------------------------------|
| [`volatility`](#quantkit.rolling.volatility)(returns\[, window, min_periods, ddof, factor\])                  | Volatility over rolling windows.                     |
| [`drawdown`](#quantkit.rolling.drawdown)(prices\[, window, min_periods, relative\])                           | Drawdown over rolling windows.                       |
| [`max_drawdown`](#quantkit.rolling.max_drawdown)(prices\[, window, min_periods, relative\])                   | Max drawdown over rolling windows.                   |
| [`sharpe_ratio`](#quantkit.rolling.sharpe_ratio)(returns, risk_free\[, window, ...\])                         | Sharpe ratio over rolling windows.                   |
| [`value_at_risk`](#quantkit.rolling.value_at_risk)(returns\[, window, min_periods, confidence\])              | Value at risk over rolling windows.                  |
| [`max_drawup`](#quantkit.rolling.max_drawup)(prices\[, window, min_periods, relative\])                       | Max drawup over rolling windows.                     |
| [`annualized_return`](#quantkit.rolling.annualized_return)(returns\[, window, min_periods, ...\])             | Annualized return over rolling windows.              |
| [`calmar_ratio`](#quantkit.rolling.calmar_ratio)(returns\[, window, min_periods, ...\])                       | Calmar ratio over rolling windows.                   |
| [`average_gain`](#quantkit.rolling.average_gain)(returns\[, window, min_periods, method\])                    | Average gain over rolling windows.                   |
| [`average_loss`](#quantkit.rolling.average_loss)(returns\[, window, min_periods, method\])                    | Average loss over rolling windows.                   |
| [`gain_loss_ratio`](#quantkit.rolling.gain_loss_ratio)(returns\[, window, min_periods\])                      | Gain loss ratio over rolling windows.                |
| [`up_period_percent`](#quantkit.rolling.up_period_percent)(returns\[, window, min_periods\])                  | Up period percent over rolling windows.              |
| [`down_period_percent`](#quantkit.rolling.down_period_percent)(returns\[, window, min_periods\])              | Down period percent over rolling windows.            |
| [`downside_deviation`](#quantkit.rolling.downside_deviation)(returns\[, window, min_periods, ...\])           | Downside deviation over rolling windows.             |
| [`upside_deviation`](#quantkit.rolling.upside_deviation)(returns\[, window, min_periods, mar, ...\])          | Upside deviation over rolling windows.               |
| [`kappa`](#quantkit.rolling.kappa)(returns\[, window, min_periods, mar, order, factor\])                      | Kappa over rolling windows.                          |
| [`omega_ratio`](#quantkit.rolling.omega_ratio)(returns\[, window, min_periods, mar\])                         | Omega ratio over rolling windows.                    |
| [`sortino_ratio`](#quantkit.rolling.sortino_ratio)(returns\[, window, min_periods, mar, factor\])             | Sortino ratio over rolling windows.                  |
| [`max_drawdown_peak`](#quantkit.rolling.max_drawdown_peak)(prices\[, window, min_periods\])                   | Max drawdown peak over rolling windows.              |
| [`max_drawdown_valley`](#quantkit.rolling.max_drawdown_valley)(prices\[, window, min_periods\])               | Max drawdown valley over rolling windows.            |
| [`max_drawdown_recovery`](#quantkit.rolling.max_drawdown_recovery)(prices\[, window, min_periods\])           | Max drawdown recovery over rolling windows.          |
| [`max_drawdown_duration`](#quantkit.rolling.max_drawdown_duration)(prices\[, window, min_periods\])           | Max drawdown duration over rolling windows.          |
| [`max_drawdown_recovery_duration`](#quantkit.rolling.max_drawdown_recovery_duration)(prices\[, window, ...\]) | Max drawdown recovery duration over rolling windows. |
| [`longest_drawdown_duration`](#quantkit.rolling.longest_drawdown_duration)(prices\[, window, min_periods\])   | Longest drawdown duration over rolling windows.      |
| [`average_drawdown`](#quantkit.rolling.average_drawdown)(prices\[, window, min_periods, ...\])                | Average drawdown over rolling windows.               |
| [`beta`](#quantkit.rolling.beta)(returns, benchmark\[, window, min_periods\])                                 | Beta over rolling windows.                           |
| [`alpha`](#quantkit.rolling.alpha)(returns, benchmark\[, window, min_periods, ...\])                          | Alpha over rolling windows.                          |
| [`correlation`](#quantkit.rolling.correlation)(returns, benchmark\[, window, min_periods\])                   | Correlation over rolling windows.                    |
| [`r_squared`](#quantkit.rolling.r_squared)(returns, benchmark\[, window, min_periods\])                       | R squared over rolling windows.                      |
| [`bull_beta`](#quantkit.rolling.bull_beta)(returns, benchmark\[, window, min_periods\])                       | Bull beta over rolling windows.                      |
| [`bear_beta`](#quantkit.rolling.bear_beta)(returns, benchmark\[, window, min_periods\])                       | Bear beta over rolling windows.                      |
| [`treynor_ratio`](#quantkit.rolling.treynor_ratio)(returns, benchmark\[, window, ...\])                       | Treynor ratio over rolling windows.                  |
| [`up_capture`](#quantkit.rolling.up_capture)(returns, benchmark\[, window, min_periods\])                     | Up capture over rolling windows.                     |
| [`down_capture`](#quantkit.rolling.down_capture)(returns, benchmark\[, window, min_periods\])                 | Down capture over rolling windows.                   |
| [`overall_capture`](#quantkit.rolling.overall_capture)(returns, benchmark\[, window, min_periods\])           | Overall capture over rolling windows.                |
| [`batting_average`](#quantkit.rolling.batting_average)(returns, benchmark\[, window, min_periods\])           | Batting average over rolling windows.                |
| [`sterling_ratio`](#quantkit.rolling.sterling_ratio)(returns\[, window, min_periods, ...\])                   | Sterling ratio over rolling windows.                 |
| [`tracking_error`](#quantkit.rolling.tracking_error)(returns, benchmark\[, window, ...\])                     | Tracking error over rolling windows.                 |
| [`information_ratio`](#quantkit.rolling.information_ratio)(returns, benchmark\[, window, ...\])               | Information ratio over rolling windows.              |
| [`drawup`](#quantkit.rolling.drawup)(prices\[, window, min_periods, relative\])                               | Drawup over rolling windows.                         |
| [`expected_shortfall`](#quantkit.rolling.expected_shortfall)(returns\[, window, min_periods, ...\])           | Calculate expected shortfall over rolling windows.   |
| [`conditional_drawdown_at_risk`](#quantkit.rolling.conditional_drawdown_at_risk)(wealth\[, window, ...\])     | Conditional drawdown at risk over rolling windows.   |

<a id="module-contents"></a>

## Module Contents

<a id="quantkit.rolling.total_returns"></a>

### quantkit.rolling.total_returns(prices, window=BYEAR, min_periods=None, factor=None, relative=True)

Total returns over rolling windows.

* **Parameters:**
  **prices** — array-like
  : Observed prices or wealth, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

  **factor** — float, optional
  : As in [`quantkit.stats.total_returns()`](../stats/index.md#quantkit.stats.total_returns).

  **relative** — bool, optional
  : As in [`quantkit.stats.total_returns()`](../stats/index.md#quantkit.stats.total_returns).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.volatility"></a>

### quantkit.rolling.volatility(returns, window=BYEAR, min_periods=2, ddof=1, factor=None)

Volatility over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

  **ddof** — int, optional
  : As in [`quantkit.stats.volatility()`](../stats/index.md#quantkit.stats.volatility).

  **factor** — float, optional
  : As in [`quantkit.stats.volatility()`](../stats/index.md#quantkit.stats.volatility).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.drawdown"></a>

### quantkit.rolling.drawdown(prices, window=BYEAR, min_periods=None, relative=True)

Drawdown over rolling windows.

* **Parameters:**
  **prices** — array-like
  : Observed prices or wealth, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

  **relative** — bool, optional
  : As in [`quantkit.stats.drawdown()`](../stats/index.md#quantkit.stats.drawdown).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

### Notes

A missing current price produces NaN. Zero denominators produce NaN.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.max_drawdown"></a>

### quantkit.rolling.max_drawdown(prices, window=BYEAR, min_periods=None, relative=True)

Max drawdown over rolling windows.

* **Parameters:**
  **prices** — array-like
  : Observed prices or wealth, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

  **relative** — bool, optional
  : As in [`quantkit.stats.max_drawdown()`](../stats/index.md#quantkit.stats.max_drawdown).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

### Notes

Both extrema belong to the current window in chronological order.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.sharpe_ratio"></a>

### quantkit.rolling.sharpe_ratio(returns, risk_free, window=BYEAR, min_periods=None, factor=np.sqrt(BYEAR))

Sharpe ratio over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **risk_free** — float or array-like
  : Risk-free returns per observation; Series align by index.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

  **factor** — float, optional
  : As in [`quantkit.stats.sharpe_ratio()`](../stats/index.md#quantkit.stats.sharpe_ratio).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.value_at_risk"></a>

### quantkit.rolling.value_at_risk(returns, window=BYEAR, min_periods=None, confidence=0.95)

Value at risk over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

  **confidence** — float, optional
  : As in [`quantkit.stats.value_at_risk()`](../stats/index.md#quantkit.stats.value_at_risk).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.max_drawup"></a>

### quantkit.rolling.max_drawup(prices, window=BYEAR, min_periods=None, relative=True)

Max drawup over rolling windows.

* **Parameters:**
  **prices** — array-like
  : Observed prices or wealth, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

  **relative** — bool, optional
  : As in [`quantkit.stats.max_drawup()`](../stats/index.md#quantkit.stats.max_drawup).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

### Notes

Both extrema belong to the current window in chronological order.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.annualized_return"></a>

### quantkit.rolling.annualized_return(returns, window=BYEAR, min_periods=None, periods_per_year=BYEAR)

Annualized return over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

  **periods_per_year** — int, optional
  : As in [`quantkit.stats.annualized_return()`](../stats/index.md#quantkit.stats.annualized_return).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.calmar_ratio"></a>

### quantkit.rolling.calmar_ratio(returns, window=BYEAR, min_periods=None, periods_per_year=BYEAR)

Calmar ratio over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

  **periods_per_year** — int, optional
  : As in [`quantkit.stats.calmar_ratio()`](../stats/index.md#quantkit.stats.calmar_ratio).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.average_gain"></a>

### quantkit.rolling.average_gain(returns, window=BYEAR, min_periods=None, method='arith')

Average gain over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

  **method** — str, optional
  : As in [`quantkit.stats.average_gain()`](../stats/index.md#quantkit.stats.average_gain).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.average_loss"></a>

### quantkit.rolling.average_loss(returns, window=BYEAR, min_periods=None, method='arith')

Average loss over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

  **method** — str, optional
  : As in [`quantkit.stats.average_loss()`](../stats/index.md#quantkit.stats.average_loss).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.gain_loss_ratio"></a>

### quantkit.rolling.gain_loss_ratio(returns, window=BYEAR, min_periods=None)

Gain loss ratio over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.up_period_percent"></a>

### quantkit.rolling.up_period_percent(returns, window=BYEAR, min_periods=None)

Up period percent over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.down_period_percent"></a>

### quantkit.rolling.down_period_percent(returns, window=BYEAR, min_periods=None)

Down period percent over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.downside_deviation"></a>

### quantkit.rolling.downside_deviation(returns, window=BYEAR, min_periods=None, mar=0.0, factor=None)

Downside deviation over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

  **mar** — float, optional
  : As in [`quantkit.stats.downside_deviation()`](../stats/index.md#quantkit.stats.downside_deviation).

  **factor** — float, optional
  : As in [`quantkit.stats.downside_deviation()`](../stats/index.md#quantkit.stats.downside_deviation).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.upside_deviation"></a>

### quantkit.rolling.upside_deviation(returns, window=BYEAR, min_periods=None, mar=0.0, factor=None)

Upside deviation over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

  **mar** — float, optional
  : As in [`quantkit.stats.upside_deviation()`](../stats/index.md#quantkit.stats.upside_deviation).

  **factor** — float, optional
  : As in [`quantkit.stats.upside_deviation()`](../stats/index.md#quantkit.stats.upside_deviation).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.kappa"></a>

### quantkit.rolling.kappa(returns, window=BYEAR, min_periods=None, mar=0.0, order=2, factor=None)

Kappa over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

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

<a id="quantkit.rolling.omega_ratio"></a>

### quantkit.rolling.omega_ratio(returns, window=BYEAR, min_periods=None, mar=0.0)

Omega ratio over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

  **mar** — float, optional
  : As in [`quantkit.stats.omega_ratio()`](../stats/index.md#quantkit.stats.omega_ratio).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.sortino_ratio"></a>

### quantkit.rolling.sortino_ratio(returns, window=BYEAR, min_periods=None, mar=0.0, factor=None)

Sortino ratio over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

  **mar** — float, optional
  : As in [`quantkit.stats.sortino_ratio()`](../stats/index.md#quantkit.stats.sortino_ratio).

  **factor** — float, optional
  : As in [`quantkit.stats.sortino_ratio()`](../stats/index.md#quantkit.stats.sortino_ratio).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.max_drawdown_peak"></a>

### quantkit.rolling.max_drawdown_peak(prices, window=BYEAR, min_periods=None)

Max drawdown peak over rolling windows.

* **Parameters:**
  **prices** — array-like
  : Observed prices or wealth, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

### Notes

Results are original pandas index labels or absolute NumPy row positions. Recovery is only reported once it has been observed.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.max_drawdown_valley"></a>

### quantkit.rolling.max_drawdown_valley(prices, window=BYEAR, min_periods=None)

Max drawdown valley over rolling windows.

* **Parameters:**
  **prices** — array-like
  : Observed prices or wealth, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

### Notes

Results are original pandas index labels or absolute NumPy row positions. Recovery is only reported once it has been observed.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.max_drawdown_recovery"></a>

### quantkit.rolling.max_drawdown_recovery(prices, window=BYEAR, min_periods=None)

Max drawdown recovery over rolling windows.

* **Parameters:**
  **prices** — array-like
  : Observed prices or wealth, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

### Notes

Results are original pandas index labels or absolute NumPy row positions. Recovery is only reported once it has been observed.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.max_drawdown_duration"></a>

### quantkit.rolling.max_drawdown_duration(prices, window=BYEAR, min_periods=None)

Max drawdown duration over rolling windows.

* **Parameters:**
  **prices** — array-like
  : Observed prices or wealth, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.max_drawdown_recovery_duration"></a>

### quantkit.rolling.max_drawdown_recovery_duration(prices, window=BYEAR, min_periods=None)

Max drawdown recovery duration over rolling windows.

* **Parameters:**
  **prices** — array-like
  : Observed prices or wealth, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.longest_drawdown_duration"></a>

### quantkit.rolling.longest_drawdown_duration(prices, window=BYEAR, min_periods=None)

Longest drawdown duration over rolling windows.

* **Parameters:**
  **prices** — array-like
  : Observed prices or wealth, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.average_drawdown"></a>

### quantkit.rolling.average_drawdown(prices, window=BYEAR, min_periods=None, periods_per_year=BYEAR)

Average drawdown over rolling windows.

* **Parameters:**
  **prices** — array-like
  : Observed prices or wealth, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

  **periods_per_year** — int, optional
  : As in [`quantkit.stats.average_drawdown()`](../stats/index.md#quantkit.stats.average_drawdown).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.beta"></a>

### quantkit.rolling.beta(returns, benchmark, window=BYEAR, min_periods=None)

Beta over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.alpha"></a>

### quantkit.rolling.alpha(returns, benchmark, window=BYEAR, min_periods=None, risk_free=0.0, factor=None)

Alpha over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

  **risk_free** — float, optional
  : As in [`quantkit.stats.alpha()`](../stats/index.md#quantkit.stats.alpha).

  **factor** — float, optional
  : As in [`quantkit.stats.alpha()`](../stats/index.md#quantkit.stats.alpha).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.correlation"></a>

### quantkit.rolling.correlation(returns, benchmark, window=BYEAR, min_periods=None)

Correlation over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.r_squared"></a>

### quantkit.rolling.r_squared(returns, benchmark, window=BYEAR, min_periods=None)

R squared over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.bull_beta"></a>

### quantkit.rolling.bull_beta(returns, benchmark, window=BYEAR, min_periods=None)

Bull beta over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.bear_beta"></a>

### quantkit.rolling.bear_beta(returns, benchmark, window=BYEAR, min_periods=None)

Bear beta over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.treynor_ratio"></a>

### quantkit.rolling.treynor_ratio(returns, benchmark, window=BYEAR, min_periods=None, risk_free=0.0, factor=None)

Treynor ratio over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

  **risk_free** — float, optional
  : As in [`quantkit.stats.treynor_ratio()`](../stats/index.md#quantkit.stats.treynor_ratio).

  **factor** — float, optional
  : As in [`quantkit.stats.treynor_ratio()`](../stats/index.md#quantkit.stats.treynor_ratio).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.up_capture"></a>

### quantkit.rolling.up_capture(returns, benchmark, window=BYEAR, min_periods=None)

Up capture over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.down_capture"></a>

### quantkit.rolling.down_capture(returns, benchmark, window=BYEAR, min_periods=None)

Down capture over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.overall_capture"></a>

### quantkit.rolling.overall_capture(returns, benchmark, window=BYEAR, min_periods=None)

Overall capture over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.batting_average"></a>

### quantkit.rolling.batting_average(returns, benchmark, window=BYEAR, min_periods=None)

Batting average over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.sterling_ratio"></a>

### quantkit.rolling.sterling_ratio(returns, window=BYEAR, min_periods=None, periods_per_year=BYEAR, excess=0.1)

Sterling ratio over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

  **periods_per_year** — int, optional
  : As in [`quantkit.stats.sterling_ratio()`](../stats/index.md#quantkit.stats.sterling_ratio).

  **excess** — float, optional
  : As in [`quantkit.stats.sterling_ratio()`](../stats/index.md#quantkit.stats.sterling_ratio).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.tracking_error"></a>

### quantkit.rolling.tracking_error(returns, benchmark, window=BYEAR, min_periods=None, factor=None)

Tracking error over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

  **factor** — float, optional
  : As in [`quantkit.stats.tracking_error()`](../stats/index.md#quantkit.stats.tracking_error).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.information_ratio"></a>

### quantkit.rolling.information_ratio(returns, benchmark, window=BYEAR, min_periods=None, factor=None)

Information ratio over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **benchmark** — array-like
  : One-dimensional benchmark aligned by index for pandas inputs.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

  **factor** — float, optional
  : As in [`quantkit.stats.information_ratio()`](../stats/index.md#quantkit.stats.information_ratio).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.drawup"></a>

### quantkit.rolling.drawup(prices, window=BYEAR, min_periods=None, relative=True)

Drawup over rolling windows.

* **Parameters:**
  **prices** — array-like
  : Observed prices or wealth, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

  **relative** — bool, optional
  : As in [`quantkit.expanding.drawup()`](../expanding/index.md#quantkit.expanding.drawup).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

### Notes

A missing current price produces NaN. Zero denominators produce NaN.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.expected_shortfall"></a>

### quantkit.rolling.expected_shortfall(returns, window=BYEAR, min_periods=None, confidence=0.95)

Calculate expected shortfall over rolling windows.

* **Parameters:**
  **returns** — array-like
  : Observed returns, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

  **confidence** — float, optional
  : As in [`quantkit.portfolio.tail_risk.expected_shortfall()`](../portfolio/tail_risk/index.md#quantkit.portfolio.tail_risk.expected_shortfall).
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

### Notes

NaN within a window propagates, following the scalar tail-risk function.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.rolling.conditional_drawdown_at_risk"></a>

### quantkit.rolling.conditional_drawdown_at_risk(wealth, window=BYEAR, min_periods=None, confidence=0.95)

Conditional drawdown at risk over rolling windows.

* **Parameters:**
  **wealth** — array-like
  : Observed prices or wealth, with time along axis 0.

  **window** — int, str or timedelta, optional
  : Positive row count or fixed duration on a sorted, unique time index. Duration windows include (timestamp - window, timestamp\].

  **min_periods** — int or None, optional
  : Minimum valid observations (jointly valid pairs for benchmark stats). None uses window size for row windows and 1 for time windows.

  **confidence** — float, optional
  : Tail confidence level, as in the scalar CDaR function.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Statistic at each row, preserving input type, shape and labels.

### Notes

NaN within a window propagates, following the scalar tail-risk function.

<!-- !! processed by numpydoc !! -->
