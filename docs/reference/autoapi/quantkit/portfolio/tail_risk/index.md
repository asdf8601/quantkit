<a id="module-quantkit.portfolio.tail_risk"></a>

<a id="quantkit-portfolio-tail-risk"></a>

# quantkit.portfolio.tail_risk

Historical tail-risk measures for portfolio time series.

<!-- !! processed by numpydoc !! -->

<a id="functions"></a>

## Functions

| [`_confidence_level`](#quantkit.portfolio.tail_risk._confidence_level)(confidence)                                   | Validate and normalize a confidence level.                |
|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| [`_empirical_upper_tail_mean`](#quantkit.portfolio.tail_risk._empirical_upper_tail_mean)(values, confidence)         | Average the upper tail of equally probable observations.  |
| [`expected_shortfall`](#quantkit.portfolio.tail_risk.expected_shortfall)(returns\[, confidence\])                    | Calculate historical expected shortfall (ES, or CVaR).    |
| [`conditional_drawdown_at_risk`](#quantkit.portfolio.tail_risk.conditional_drawdown_at_risk)(wealth\[, confidence\]) | Calculate historical conditional drawdown at risk (CDaR). |

<a id="module-contents"></a>

## Module Contents

<a id="quantkit.portfolio.tail_risk._confidence_level"></a>

### quantkit.portfolio.tail_risk.\_confidence_level(confidence)

Validate and normalize a confidence level.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.tail_risk._empirical_upper_tail_mean"></a>

### quantkit.portfolio.tail_risk.\_empirical_upper_tail_mean(values, confidence)

Average the upper tail of equally probable observations.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.tail_risk.expected_shortfall"></a>

### quantkit.portfolio.tail_risk.expected_shortfall(returns, confidence=0.95)

Calculate historical expected shortfall (ES, or CVaR).

Returns are converted to signed losses as $L_t=-r_t$.  ES is the average over the worst probability mass $1-c$ of the empirical loss distribution.  Each sample has probability $1/n$; when the tail boundary cuts through a sample, the required fraction of that boundary observation is included.  This exact tail integration differs from averaging observations selected by an interpolated quantile.

* **Parameters:**
  **returns** — numpy.ndarray or pandas.Series or pandas.DataFrame
  : A nonempty return history, shaped as time or time by portfolio.

  **confidence** — float, default 0.95
  : Finite confidence level strictly between 0 and 1.
* **Returns:**
  float or numpy.ndarray or pandas.Series
  : Signed expected loss for each portfolio.  A loss is positive, while an all-gain tail can be negative.  Any missing observation makes its portfolio result NaN.
* **Raises:**
  TypeError
  : If an input is not real numeric or `confidence` is not a real scalar.

  ValueError
  : If an input axis is empty, values include infinity, labels are duplicated, or `confidence` is outside the open interval (0, 1).

### References

<a id="r741dff284e60-1"></a>

[1]

Riskfolio-Lib, Risk Functions. [https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html](https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html)

<a id="r741dff284e60-2"></a>

[2]

Rockafellar, R. T. and Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. Journal of Risk, 2(3), 21-41.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.tail_risk.conditional_drawdown_at_risk"></a>

### quantkit.portfolio.tail_risk.conditional_drawdown_at_risk(wealth, confidence=0.95)

Calculate historical conditional drawdown at risk (CDaR).

The input is a nonnegative, flow-adjusted compounded wealth history that includes its initial baseline.  At each observed time, drawdown is $1-W_t/\max_{u\leq t}W_u$.  CDaR integrates the worst probability mass $1-c$ of these equally weighted observed drawdowns, including the initial zero drawdown and any fractional boundary observation.

This function computes its drawdown path directly.  The existing drawdown statistics use different missing-value and sign conventions and therefore are not reused here.

* **Parameters:**
  **wealth** — numpy.ndarray or pandas.Series or pandas.DataFrame
  : A nonempty wealth history, shaped as time or time by portfolio.  The first value of each portfolio must be positive; later values may be zero.

  **confidence** — float, default 0.95
  : Finite confidence level strictly between 0 and 1.
* **Returns:**
  float or numpy.ndarray or pandas.Series
  : Mean tail drawdown for each portfolio.  Any missing observation makes its portfolio result NaN.
* **Raises:**
  TypeError
  : If an input is not real numeric or `confidence` is not a real scalar.

  ValueError
  : If an input axis is empty, wealth is negative, an observed initial value is not positive, values include infinity, labels are duplicated, or `confidence` is outside the open interval (0, 1).

### References

<a id="r3fece56dccdb-1"></a>

[1]

Riskfolio-Lib, Risk Functions. [https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html](https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html)

<!-- !! processed by numpydoc !! -->
