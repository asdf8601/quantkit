"""Portfolio position and net asset value calculations.

The functions in this module value signed positions and reduce them to
portfolio-level amounts while preserving the input container's metadata.
"""

import numpy as np
import pandas as pd

from quantkit.portfolio._utils import (
    _real_array,
    asset_array,
    pair_assets,
    portfolio_amount,
    wrap_assets,
    wrap_reduction,
)


def position_values(quantities, prices):
    """Value signed positions at nonnegative prices.

    Parameters
    ----------
    quantities : numpy.ndarray, pandas.Series, or pandas.DataFrame
        Signed units held in each asset. A matrix has time on rows and assets
        on columns.
    prices : numpy.ndarray, pandas.Series, or pandas.DataFrame
        Nonnegative monetary price for each asset. A vector may be broadcast
        across the time axis of matrix ``quantities``.

    Returns
    -------
    numpy.ndarray, pandas.Series, or pandas.DataFrame
        Position values with the type and labels of ``quantities``.

    Raises
    ------
    ValueError
        If prices are negative or the inputs have incompatible asset labels or
        shapes.
    TypeError
        If an input is not real numeric asset data.

    References
    ----------
    .. [1] Boyd et al., `Multi-Period Trading via Convex Optimization <https://www.cvxportfolio.com/en/1.5.0/_static/cvx_portfolio.pdf>`_.
    """
    quantity_values, price_values = pair_assets(quantities, prices)
    if np.any(price_values < 0):
        raise ValueError("prices must be nonnegative")
    return wrap_assets(quantities, np.multiply(quantity_values, price_values))


def gross_asset_value(position_values, cash=0.0):
    """Calculate gross asset value from positions and cash.

    Gross asset value is the sum of positive position values plus positive
    cash. Negative cash and short positions do not reduce gross asset value.

    Parameters
    ----------
    position_values : numpy.ndarray, pandas.Series, or pandas.DataFrame
        Signed monetary value of each position.
    cash : float, numpy.ndarray, or pandas.Series, default 0.0
        Cash balance. For matrix positions, it may be a scalar or one value
        per time row.

    Returns
    -------
    float, numpy.ndarray, or pandas.Series
        Gross asset value. DataFrame inputs return a time-indexed Series.

    References
    ----------
    .. [1] `Investor.gov: Net asset value <https://www.investor.gov/introduction-investing/investing-basics/glossary/net-asset-value>`_.
    """
    values = asset_array(position_values, "position_values")
    cash_values = portfolio_amount(position_values, cash, "cash")
    gross_value = np.sum(np.maximum(values, 0.0), axis=-1)
    gross_value += np.maximum(cash_values, 0.0)
    return wrap_reduction(position_values, gross_value)


def net_asset_value(position_values, cash=0.0, liabilities=0.0):
    """Calculate net asset value from positions, cash, and liabilities.

    Additional liabilities are deducted after including signed position values
    and cash. Short positions are already represented in ``position_values``.

    Parameters
    ----------
    position_values : numpy.ndarray, pandas.Series, or pandas.DataFrame
        Signed monetary value of each position.
    cash : float, numpy.ndarray, or pandas.Series, default 0.0
        Cash balance. For matrix positions, it may be a scalar or one value
        per time row.
    liabilities : float, numpy.ndarray, or pandas.Series, default 0.0
        Nonnegative additional liabilities. For matrix positions, it may be a
        scalar or one value per time row.

    Returns
    -------
    float, numpy.ndarray, or pandas.Series
        Net asset value. DataFrame inputs return a time-indexed Series.

    Raises
    ------
    ValueError
        If an additional liability is negative.

    References
    ----------
    .. [1] `Investor.gov: Net asset value <https://www.investor.gov/introduction-investing/investing-basics/glossary/net-asset-value>`_.
    """
    values = asset_array(position_values, "position_values")
    cash_values = portfolio_amount(position_values, cash, "cash")
    liability_values = portfolio_amount(
        position_values, liabilities, "liabilities"
    )
    if np.any(np.asarray(liability_values) < 0):
        raise ValueError("liabilities must be nonnegative")
    nav = np.sum(values, axis=-1) + cash_values - liability_values
    return wrap_reduction(position_values, nav)


def _scalar_or_vector(values, name):
    """Validate a scalar or nonempty one-dimensional real numeric value."""
    if isinstance(values, pd.DataFrame):
        raise ValueError(f"{name} must be scalar or one-dimensional")
    array = _real_array(values, name)
    if array.ndim > 1 or (array.ndim == 1 and array.size == 0):
        raise ValueError(f"{name} must be scalar or one-dimensional")
    if isinstance(values, pd.Series) and values.index.has_duplicates:
        raise ValueError(f"{name} has duplicate labels")
    return array


def nav_per_share(nav, shares):
    """Calculate NAV per positive outstanding share.

    Parameters
    ----------
    nav : float, numpy.ndarray, or pandas.Series
        Scalar or one-dimensional total net asset value. Negative values are
        valid and missing values propagate.
    shares : float, numpy.ndarray, or pandas.Series
        Positive, finite outstanding shares. A Series aligns to a Series
        ``nav`` by exact index label sets; a scalar broadcasts over NAV.

    Returns
    -------
    float, numpy.ndarray, or pandas.Series
        NAV divided by shares, preserving a Series index and name.

    Raises
    ------
    ValueError
        If inputs are not scalar or one-dimensional, labels or shapes do not
        match, or a finite share count is not positive.
    TypeError
        If an input is not real numeric data.

    References
    ----------
    .. [1] `Investor.gov: Net asset value <https://www.investor.gov/introduction-investing/investing-basics/glossary/net-asset-value>`_.
    """
    nav_values = _scalar_or_vector(nav, "nav")
    share_values = _scalar_or_vector(shares, "shares")

    if isinstance(nav, pd.Series) and isinstance(shares, pd.Series):
        if (
            len(nav.index) != len(shares.index)
            or not nav.index.isin(shares.index).all()
        ):
            raise ValueError("shares labels must match nav")
        share_values = _real_array(shares.reindex(nav.index), "shares")

    if nav_values.ndim == 0:
        if share_values.ndim != 0:
            raise ValueError("shares must be scalar when nav is scalar")
    elif share_values.ndim != 0 and share_values.shape != nav_values.shape:
        raise ValueError("shares must match nav shape")

    if np.any(share_values <= 0):
        raise ValueError("finite shares must be positive")

    result = np.divide(nav_values, share_values)
    if isinstance(nav, pd.Series):
        return pd.Series(result, index=nav.index, name=nav.name, copy=False)
    if result.ndim == 0:
        return float(result)
    return result
