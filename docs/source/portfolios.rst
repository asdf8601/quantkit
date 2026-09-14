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

A vector represents assets at one instant. A matrix is time by assets.
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
periods before using it to report performance. This release does not introduce
flow accounting or change existing annualization and deviation conventions.

Turnover
--------

.. code-block:: python

   from quantkit.portfolio import turnover

   result = turnover.turnover(np.array([20.0, -20.0]), nav=100.0)  # 0.20

The convention is half the sum of absolute executed trade values divided by
NAV. Pass trades before netting repeated buys and sells. Changes in weights
caused by market movement are not trades.

References
----------

* `SEC: Net Asset Value <https://www.investor.gov/introduction-investing/investing-basics/glossary/net-asset-value>`_
* `Boyd et al.: Multi-Period Trading via Convex Optimization <https://www.cvxportfolio.com/en/1.5.0/_static/cvx_portfolio.pdf>`_
* `Cvxportfolio constraints <https://www.cvxportfolio.com/en/1.5.0/constraints.html>`_
* `Cvxportfolio results <https://www.cvxportfolio.com/en/1.5.0/result.html>`_
