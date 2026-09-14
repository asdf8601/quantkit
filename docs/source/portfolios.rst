Portfolio statistics
====================

``quantkit.portfolio`` operates on cash instruments valued in one currency.
Quantities are signed units; weights are fractions of portfolio capital.
Each module computes a family of statistics:

* ``valuation``: position values, gross and net asset value, NAV per share.
* ``exposure``: long, short, gross and net exposure, gross leverage.
* ``weights``: NAV weights, gross-normalized magnitudes and concentration.
* ``returns``: realized and expected returns weighted across assets.
* ``turnover``: executed trade volume relative to NAV.
* ``flows``: monetary P&L, returns adjusted for external flows and TWR.
* ``risk``: covariance-based variance, volatility and risk contributions.
* ``tail_risk``: historical Expected Shortfall and Conditional Drawdown at Risk.

Valuation and exposure
----------------------

.. code-block:: python

   import pandas as pd
   from quantkit.portfolio import valuation, exposure, weights

   quantities = pd.Series([10.0, -4.0], index=["long", "short"])
   prices = pd.Series([10.0, 10.0], index=["long", "short"])
   values = valuation.position_values(quantities, prices)
   nav = valuation.net_asset_value(values, cash=40.0)  # 100
   gav = valuation.gross_asset_value(values, cash=40.0)  # 140
   gross = exposure.gross_exposure(values)  # 140
   net = exposure.net_exposure(values)  # 60
   leverage = exposure.gross_leverage(values, nav)  # 1.4
   allocation = weights.nav_weights(values, nav)  # [1.0, -0.4]

GAV is long assets plus positive cash. NAV subtracts short positions, negative
cash financing and separately supplied additional liabilities. Do not include
short liabilities a second time in ``liabilities``. This first version has no
separate valuation for receivables or other non-position assets.
Gross exposure sums absolute position values and excludes cash; it is not GAV.
For example, with the same positions and zero cash, GAV is 100 and gross
exposure is 140.

Inputs and missing values
-------------------------

For valuation, weights, exposures and covariance-based risk, a vector
represents assets at one instant. A matrix is time by assets.
Position-preserving functions retain pandas labels and names. Reducing assets
returns one value per time row, or a scalar for a single asset vector.

The first argument determines the shape and result container. A second asset
vector can broadcast across its rows. Two pandas inputs must contain exactly
the same relevant labels; the second is reordered to the first. No assets or
timestamps are silently dropped. A Series used as an asset input has asset
labels; a Series supplied as ``cash`` or ``nav`` has time labels.

Missing observations propagate, including zero times a missing value. Empty
axes, duplicate labels, infinities and incompatible shapes are rejected.
Negative prices are rejected. Ratios requiring positive capital give NaN for
nonpositive NAV; monetary NAV itself may be negative.

Returns and existing statistics
-------------------------------

.. code-block:: python

   import numpy as np
   from quantkit import stats
   from quantkit.portfolio import returns

   asset_returns = pd.DataFrame({"a": [0.10, -0.10, 0.10],
                                 "b": [-0.05, 0.02, 0.03]})
   portfolio_returns = returns.portfolio_returns(
       asset_returns, np.array([0.6, 0.4])
   )
   sharpe = stats.sharpe_ratio(portfolio_returns, risk_free=0.0)

Weights must be those in force at the beginning of each return interval.
There is no implicit shift. Constant weights imply rebalancing every period;
constant quantities describe buy-and-hold. Include cash as a separate asset
with its own return when cash earns interest; include funding costs explicitly.
Returns should include distributions such as dividends.

Existing ``stats`` functions can consume the resulting portfolio series.
Construct a wealth series for price-based drawdown functions. ``core.returns``
on NAV is valid only when external contributions and withdrawals are absent.
The previous NAV must also be positive. Existing benchmark statistics may
inner-join dates and reducers may drop NaNs; prevalidate the aggregate series
when strict completeness is required.
``core.cum_returns`` skips NaNs under its existing convention; validate missing
periods before using it to report performance. To adjust for external flows,
use the explicit interval valuations described below. Existing annualization
and deviation conventions are unchanged.

Turnover
--------

.. code-block:: python

   from quantkit.portfolio import turnover

   result = turnover.turnover(np.array([20.0, -20.0]), nav=100.0)  # 0.20

The convention is half the sum of absolute executed trade values divided by
NAV. Pass trades before netting repeated buys and sells. Changes in weights
caused by market movement are not trades.

External flows and performance
--------------------------------

``flows`` accepts scalar interval values or one-dimensional arrays/Series of
intervals. Beginning and ending values must have the same shape. An external
flow may be scalar or one value per interval. Positive flows are contributions,
negative flows are withdrawals. Paired Series align by exact index labels.

.. code-block:: python

   from quantkit.portfolio import flows

   # Deposit 100 at the end of the first interval; then lose 10%.
   beginning = np.array([100.0, 210.0])
   ending = np.array([210.0, 189.0])
   external = np.array([100.0, 0.0])
   pnl = flows.profit_loss(beginning, ending, external)  # [10, -21]
   period_returns = flows.period_return(beginning, ending, external)
   # period_returns: [0.10, -0.10]
   twr = flows.time_weighted_return(beginning, ending, external)  # -0.01

P&L is ending value minus beginning value minus external flow. The default
``flow_timing="end"`` divides that P&L by beginning value. With
``flow_timing="beginning"``, it divides by beginning value plus the external
flow: beginning value is measured before that contribution or withdrawal.
Ending value is measured after any end flow. Choose one timing convention
for the supplied intervals.

TWR compounds the interval returns; it is not annualized. For flows within
an interval, split it into subperiods and supply valuations at the flows.
Applying a boundary convention to an intraperiod flow is not an exact TWR.
Missing observations propagate. Nonpositive invested capital gives NaN;
TWR rejects interval returns below -100%, which cannot be linked as
nonnegative wealth growth factors.

Covariance-based risk
---------------------

.. code-block:: python

   from quantkit.portfolio import risk

   allocation = pd.Series([0.5, 0.5], index=["a", "b"])
   covariance = pd.DataFrame([[0.04, 0.0], [0.0, 0.09]],
                             index=allocation.index,
                             columns=allocation.index)
   variance = risk.variance(allocation, covariance)  # 0.0325
   volatility = risk.volatility(allocation, covariance)  # sqrt(0.0325)
   contributions = risk.risk_contribution(allocation, covariance)
   # Contributions sum to portfolio volatility.

Variance is ``w.T @ covariance @ w``; volatility is its square root.
``marginal_risk_contribution`` returns ``covariance @ w / volatility``.
``risk_contribution`` multiplies each marginal contribution by its weight.
Contributions can be negative for positions that reduce portfolio risk.
They are undefined (NaN) when volatility is zero.

Weights may be an asset vector or time-by-asset matrix. A single covariance
matrix is reused for all rows; no covariance estimation or automatic
annualization is performed. Units and horizon follow the supplied covariance.
DataFrame covariance rows and columns align to asset labels. With NumPy
weights, a DataFrame covariance's row order defines the positional asset order;
its columns are reordered to its rows.

Covariance must be finite, square, symmetric and positive semidefinite.
Small numerical discrepancies are tolerated relative to matrix scale;
materially invalid covariance is rejected. Missing weights propagate by row,
while missing covariance entries are rejected because validity cannot be
checked.

Tail risk
---------

.. code-block:: python

   from quantkit.portfolio import tail_risk

   es = tail_risk.expected_shortfall(
       np.array([-0.4, -0.2, 0.1, 0.2]), confidence=0.625
   )  # 1/3
   cdar = tail_risk.conditional_drawdown_at_risk(
       np.array([100.0, 120.0, 90.0, 108.0]), confidence=0.5
   )  # 0.175

Expected Shortfall (also called CVaR here) averages the worst
``1 - confidence`` fraction of losses, using equal observation probabilities
and a fractional weight at the boundary. Losses are negative returns: the
result can be negative if even the worst outcomes are gains. This convention
does not simply average all observations beyond an interpolated quantile.

CDaR applies the same tail calculation to positive relative drawdowns from a
compounded, flow-adjusted wealth history. Include its initial positive baseline;
the initial zero drawdown is one of the observed samples. Later wealth may be
zero but cannot be negative. Do not pass raw NAV with contributions or
withdrawals, as those would be mistaken for investment gains or losses.

For tail measures, vectors represent observations over time. Matrix columns
represent separate portfolios and are reduced over time, giving one value per
column. This differs from covariance-based risk, which reduces over assets.
Confidence must lie strictly between zero and one. Any missing observation
makes that portfolio's tail measure NaN.

References
----------

* `SEC: Net Asset Value <https://www.investor.gov/introduction-investing/investing-basics/glossary/net-asset-value>`_
* `Boyd et al.: Multi-Period Trading via Convex Optimization <https://www.cvxportfolio.com/en/1.5.0/_static/cvx_portfolio.pdf>`_
* `Cvxportfolio constraints <https://www.cvxportfolio.com/en/1.5.0/constraints.html>`_
* `Cvxportfolio results <https://www.cvxportfolio.com/en/1.5.0/result.html>`_
* `GIPS Standards Handbook: calculation methodology <https://www.gipsstandards.org/standards/gips-standards-for-firms/gips-standards-handbook-for-firms/>`_
* `Riskfolio: risk functions <https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html>`_
