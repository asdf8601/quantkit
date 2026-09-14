<a id="module-quantkit.portfolio.flows"></a>

<a id="quantkit-portfolio-flows"></a>

# quantkit.portfolio.flows

Portfolio profit, flow-adjusted returns, and linked returns.

<!-- !! processed by numpydoc !! -->

<a id="functions"></a>

## Functions

| [`_interval_values`](#quantkit.portfolio.flows._interval_values)(values, name)                                   | Validate a scalar or nonempty one-dimensional interval input.      |
|------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| [`_aligned_values`](#quantkit.portfolio.flows._aligned_values)(reference, values, name)                          | Validate values and align matching Series labels to the reference. |
| [`_flow_values`](#quantkit.portfolio.flows._flow_values)(beginning_value, ending_value, external_flow)           | Validate and align the three interval inputs.                      |
| [`_wrap_intervals`](#quantkit.portfolio.flows._wrap_intervals)(like, values)                                     | Wrap interval results according to the first input.                |
| [`profit_loss`](#quantkit.portfolio.flows.profit_loss)(beginning_value, ending_value\[, external_flow\])         | Calculate investment profit or loss after removing external flows. |
| [`period_return`](#quantkit.portfolio.flows.period_return)(beginning_value, ending_value\[, ...\])               | Calculate a boundary-flow-adjusted return for each interval.       |
| [`time_weighted_return`](#quantkit.portfolio.flows.time_weighted_return)(beginning_value, ending_value\[, ...\]) | Geometrically link flow-adjusted interval returns.                 |

<a id="module-contents"></a>

## Module Contents

<a id="quantkit.portfolio.flows._interval_values"></a>

### quantkit.portfolio.flows.\_interval_values(values, name)

Validate a scalar or nonempty one-dimensional interval input.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.flows._aligned_values"></a>

### quantkit.portfolio.flows.\_aligned_values(reference, values, name)

Validate values and align matching Series labels to the reference.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.flows._flow_values"></a>

### quantkit.portfolio.flows.\_flow_values(beginning_value, ending_value, external_flow)

Validate and align the three interval inputs.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.flows._wrap_intervals"></a>

### quantkit.portfolio.flows.\_wrap_intervals(like, values)

Wrap interval results according to the first input.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.flows.profit_loss"></a>

### quantkit.portfolio.flows.profit_loss(beginning_value, ending_value, external_flow=0.0)

Calculate investment profit or loss after removing external flows.

The monetary profit or loss for each interval is `ending_value - beginning_value - external_flow`. Positive flows are contributions and negative flows are withdrawals.

* **Parameters:**
  **beginning_value** — scalar, numpy.ndarray, or pandas.Series
  : Portfolio value before the interval and before any beginning flow.

  **ending_value** — scalar, numpy.ndarray, or pandas.Series
  : Portfolio value after the interval and after any ending flow. It must match the dimensionality and shape of `beginning_value`.

  **external_flow** — scalar, numpy.ndarray, or pandas.Series, default 0.0
  : Net contribution for each interval. A scalar broadcasts over an interval history; a vector must match `beginning_value`.
* **Returns:**
  float, numpy.ndarray, or pandas.Series
  : Flow-adjusted monetary profit or loss, preserving the first Series’ index and name.

### Notes

This boundary-flow convention does not infer intraperiod flow timing and does not by itself claim compliance with the GIPS standards.

### References

<a id="ra571a45b0f2d-1"></a>

[1]

CFA Institute, [GIPS Standards Handbook for Firms](https://www.gipsstandards.org/standards/gips-standards-for-firms/gips-standards-handbook-for-firms/).

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.flows.period_return"></a>

### quantkit.portfolio.flows.period_return(beginning_value, ending_value, external_flow=0.0, \*, flow_timing='end')

Calculate a boundary-flow-adjusted return for each interval.

For an end flow, the return is `(ending_value - beginning_value - external_flow) / beginning_value`. For a beginning flow, its denominator is instead `beginning_value + external_flow`.

* **Parameters:**
  **beginning_value** — scalar, numpy.ndarray, or pandas.Series
  : Portfolio value before the interval and before any beginning flow.

  **ending_value** — scalar, numpy.ndarray, or pandas.Series
  : Portfolio value after the interval and after any ending flow. It must match the dimensionality and shape of `beginning_value`.

  **external_flow** — scalar, numpy.ndarray, or pandas.Series, default 0.0
  : Net contribution for each interval. Positive values are contributions and negative values are withdrawals.

  **flow_timing** — {“beginning”, “end”}, default “end”
  : Boundary at which every supplied flow occurs.
* **Returns:**
  float, numpy.ndarray, or pandas.Series
  : Return for each interval. A missing or nonpositive denominator gives NaN. The first Series’ index and name are preserved.
* **Raises:**
  ValueError
  : If `flow_timing` is invalid or input shapes or labels do not match.

### Notes

For intraperiod flows, split the history into subperiods and supply a valuation at each flow boundary to obtain an exact time-weighted return. This explicit convention does not by itself claim GIPS compliance.

### References

<a id="r0efa9ee1141e-1"></a>

[1]

CFA Institute, [GIPS Standards Handbook for Firms](https://www.gipsstandards.org/standards/gips-standards-for-firms/gips-standards-handbook-for-firms/).

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.flows.time_weighted_return"></a>

### quantkit.portfolio.flows.time_weighted_return(beginning_value, ending_value, external_flow=0.0, \*, flow_timing='end')

Geometrically link flow-adjusted interval returns.

The time-weighted return is `product(1 + period_return) - 1` over the supplied chronological interval sequence.

* **Parameters:**
  **beginning_value** — scalar, numpy.ndarray, or pandas.Series
  : Portfolio value before each interval and before any beginning flow.

  **ending_value** — scalar, numpy.ndarray, or pandas.Series
  : Portfolio value after each interval and after any ending flow. It must match the dimensionality and shape of `beginning_value`.

  **external_flow** — scalar, numpy.ndarray, or pandas.Series, default 0.0
  : Net contribution for each interval. Positive values are contributions and negative values are withdrawals.

  **flow_timing** — {“beginning”, “end”}, default “end”
  : Boundary at which every supplied flow occurs.
* **Returns:**
  float
  : Linked return for the complete sequence. Missing interval returns propagate to the result.
* **Raises:**
  ValueError
  : If an interval return is below -1, where geometric linking would continue after negative wealth, or if validation otherwise fails.

### Notes

For intraperiod flows, split the history into subperiods and supply a valuation at each flow boundary. A return of exactly -1 is accepted and compounds to -1. This explicit convention does not by itself claim GIPS compliance.

### References

<a id="re1ba8fc0448a-1"></a>

[1]

CFA Institute, [GIPS Standards Handbook for Firms](https://www.gipsstandards.org/standards/gips-standards-for-firms/gips-standards-handbook-for-firms/).

<!-- !! processed by numpydoc !! -->
