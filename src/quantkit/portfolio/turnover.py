"""Turnover based on executed monetary trades, before netting."""

import numpy as np

from ._utils import (
    asset_array,
    portfolio_amount,
    positive_divide,
    wrap_reduction,
)


def turnover(trade_values, nav):
    """Return half the absolute executed trade volume divided by NAV.

    Parameters
    ----------
    trade_values : numpy.ndarray or pandas.Series or pandas.DataFrame
        Signed monetary trades, with trades on the last axis. Repeated buys
        and sells must remain separate entries; exclude cash financing legs.
    nav : float or numpy.ndarray or pandas.Series
        Pre-trade portfolio NAV. A time vector is allowed for matrix trades;
        a Series aligns to a DataFrame's time index.

    Returns
    -------
    float or numpy.ndarray or pandas.Series
        ``0.5 * sum(abs(trade_values)) / nav`` per period. Nonpositive NAV
        gives NaN. Missing trade values propagate.

    Notes
    -----
    Market-driven weight changes are not executed trades. Cash is excluded
    from traded volume. For this convention see
    https://www.cvxportfolio.com/en/1.5.0/result.html.

    Examples
    --------
    >>> turnover(np.array([20.0, -20.0]), 100.0)
    0.2
    """
    trades = asset_array(trade_values, name="trade_values")
    capital = portfolio_amount(trade_values, nav, name="nav")
    volume = 0.5 * np.sum(np.abs(trades), axis=-1)
    return wrap_reduction(trade_values, positive_divide(volume, capital))
