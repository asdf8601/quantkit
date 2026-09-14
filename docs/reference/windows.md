<a id="window-statistics"></a>

# Window statistics

Use `qnt.stats` for a summary of the full sample, `qnt.rolling` for trailing windows and `qnt.expanding` for the history available at each row. All 40 public statistics have both window versions. The window modules also include `drawup`, `expected_shortfall` and `conditional_drawdown_at_risk`.

```python
import numpy as np
import pandas as pd
import quantkit as qnt

prices = pd.Series([100.0, 50.0, 60.0, 70.0, 80.0], name="asset")
qnt.rolling.max_drawdown(prices, window=3)
# [NaN, NaN, -0.5, 0.0, 0.0]
qnt.expanding.max_drawdown(prices)
# [0.0, -0.5, -0.5, -0.5, -0.5]
```

Both the peak and subsequent trough of a rolling maximum drawdown must belong to the current window. A rolling minimum of the global drawdown does not meet that condition. `drawdown` measures the current price relative to the highest price in the window; `max_drawdown` measures the worst fall anywhere inside it. Drawdowns are negative. Drawups are positive and use the running minimum instead.

<a id="window-and-output-contract"></a>

## Window and output contract

* NumPy arrays (one or two dimensions), Series and DataFrames return the same container, shape and pandas labels. Time is axis 0; columns are evaluated independently. Arguments can be positional or named.
* Integer windows count rows, including missing observations. Their default `min_periods=None` requires `window` valid observations. Expanding windows default to `min_periods=1`.
* For backward signature compatibility, `rolling.volatility` defaults to `min_periods=2`. Pass `min_periods=window` for a complete window.
* `min_periods` counts non-missing observations, or jointly valid observations for benchmark statistics. A value of zero still requires at least one observation. Each statistic may need more data to be mathematically defined, such as two observations for sample volatility.
* Fixed durations, such as `"30D"` or `pd.Timedelta(days=30)`, require a sorted, unique DatetimeIndex or TimedeltaIndex. The window is `(timestamp - duration, timestamp]` and defaults to one valid observation. Calendar months and other non-fixed offsets are unsupported.
* All windows are right-aligned: results never use later observations. Infinities and nonnumeric inputs are rejected. Missing values are not filled.

```python
dated = pd.Series(
    [100.0, 90.0, 95.0],
    index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-04"]),
)
qnt.rolling.max_drawdown(dated, window="2D")
# [0.0, -0.1, 0.0]: January 2 is outside the final window.
```

<a id="missing-values-and-drawdown-details"></a>

## Missing values and drawdown details

Scalar statistics define each window’s formula and missing-value rules. Most ignore missing observations, but Expected Shortfall and CDaR propagate any NaN in the window, matching `qnt.portfolio.tail_risk`. Meeting `min_periods` does not override a formula’s missing-value behavior.

Point `drawdown` and `drawup` return NaN when the current price is missing, retaining the internal peak or trough for subsequent rows. Their maximum variants retain an earlier valid episode when the current price is missing, provided `min_periods` is met. A zero denominator produces NaN. This point-in-time convention differs from `qnt.stats.drawdown`, which uses the last valid price in a reduced sample.

`max_drawdown_peak`, `max_drawdown_valley` and `max_drawdown_recovery` return original pandas index labels; NumPy inputs return absolute row positions in the original array. An unrecovered episode has a missing recovery label until recovery is actually observed. Ties and episode selection follow the scalar functions.

Maximum drawdown and recovery durations count row distances, including gaps. `longest_drawdown_duration` counts valid underwater observations. `average_drawdown` retains the scalar annualized block calculation; it is not the arithmetic average of the point drawdown series.

<a id="returns-annualization-and-benchmarks"></a>

## Returns, annualization and benchmarks

A window of 252 prices spans 251 return intervals. A window of 252 returns contains 252 intervals. Include the initial wealth observation when constructing prices from returns if the first return should affect drawdown. Return-based Calmar and Sterling ratios include that initial baseline internally, following their scalar implementations.

Window size and annualization are independent. Defaults match `qnt.stats` (`BYEAR = 261` where applicable). Volatility and Sharpe use `factor` as a direct multiplier: pass `np.sqrt(252)` to annualize daily volatility or Sharpe using 252 sessions per year.

```python
returns = pd.Series([0.01, -0.02, 0.03, 0.005])
benchmark = pd.Series([0.008, -0.01, 0.02, 0.004])
vol = qnt.rolling.volatility(returns, window=3, factor=np.sqrt(252))
sharpe = qnt.expanding.sharpe_ratio(
    returns, risk_free=0.0, factor=np.sqrt(252), min_periods=2
)
beta = qnt.rolling.beta(returns, benchmark, window=3)
es = qnt.rolling.expected_shortfall(returns, window=3, confidence=0.95)
```

Benchmark Series align to the primary input’s index before windowing. Unmatched rows remain in the output and count as missing pairs; benchmark arrays align positionally and must have the same length. One benchmark is shared across DataFrame columns. Sharpe also accepts a one-dimensional risk-free series with the same alignment rules. Alpha and Treynor keep their scalar risk-free-rate parameter.

VaR returns a signed return quantile. Expected Shortfall uses the loss convention; positive values usually indicate losses, while an all-gains tail can be negative. CDaR accepts wealth, requires positive initial wealth within each evaluated window and reports a positive drawdown-tail loss. Negative wealth raises an error even when there are too few observations to report a result. A rolling CDaR window that starts at zero also raises: relative wealth is undefined at that new baseline, even if the preceding window had a positive baseline.

<a id="compatibility-and-performance"></a>

## Compatibility and performance

`qnt.rolling` is available immediately after `import quantkit as qnt`. NumPy inputs to rolling volatility and expanding drawdown/drawup now return NumPy arrays consistently, including positional calls. Rolling volatility now honors partial `min_periods` windows instead of masking the first `window - 1` rows unconditionally. Existing expanding drawdown/drawup `relative` and `out` arguments keep their positions.

Point drawdowns and expanding maximum drawdowns run in linear time for row-count windows. Complex statistics evaluate the corresponding scalar formula per window: rolling work can grow with sample size times window size, and expanding work can be quadratic. These implementations prioritize matching the scalar definitions; use bounded rolling windows for long histories when a full expanding calculation is unnecessary.

The advanced marimo notebook demonstrates these functions on the cached market snapshot with visible calculation cells.
