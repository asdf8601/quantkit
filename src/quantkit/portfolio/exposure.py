"""Exposure measures for signed portfolio position values."""

import numpy as np

from quantkit.portfolio._utils import (
    asset_array,
    portfolio_amount,
    positive_divide,
    wrap_reduction,
)


def long_value(position_values):
    """Return the value of long positions.

    Long value is the sum of positive signed position values over the asset
    axis.  Cash is not included unless it is supplied as a position.

    Parameters
    ----------
    position_values : numpy.ndarray or pandas.Series or pandas.DataFrame
        Signed monetary values, shaped as assets or time by assets.

    Returns
    -------
    float or numpy.ndarray or pandas.Series
        Long value in monetary units.  Matrix inputs retain their time axis.

    Notes
    -----
    Missing values propagate through each asset-axis reduction.

    References
    ----------
    .. [1] Cvxportfolio, "Long and short positions",
       https://www.cvxportfolio.com/en/1.5.0/result.html
    """
    values = asset_array(position_values).astype(float, copy=False)
    return wrap_reduction(
        position_values, np.sum(np.maximum(values, 0.0), axis=-1)
    )


def short_value(position_values):
    """Return the positive magnitude of short positions.

    Short value is the sum of the magnitudes of negative signed position
    values over the asset axis.  It is reported as a positive monetary amount.

    Parameters
    ----------
    position_values : numpy.ndarray or pandas.Series or pandas.DataFrame
        Signed monetary values, shaped as assets or time by assets.

    Returns
    -------
    float or numpy.ndarray or pandas.Series
        Positive short value in monetary units.  Matrix inputs retain their
        time axis.

    Notes
    -----
    Missing values propagate through each asset-axis reduction.

    References
    ----------
    .. [1] Cvxportfolio, "Long and short positions",
       https://www.cvxportfolio.com/en/1.5.0/result.html
    """
    values = asset_array(position_values).astype(float, copy=False)
    return wrap_reduction(
        position_values, np.sum(np.maximum(-values, 0.0), axis=-1)
    )


def gross_exposure(position_values):
    """Return gross exposure as the sum of absolute position values.

    Parameters
    ----------
    position_values : numpy.ndarray or pandas.Series or pandas.DataFrame
        Signed monetary values, shaped as assets or time by assets.

    Returns
    -------
    float or numpy.ndarray or pandas.Series
        Gross exposure in monetary units, excluding separately held cash.

    Notes
    -----
    Missing values propagate through each asset-axis reduction.

    References
    ----------
    .. [1] Cvxportfolio, "Leverage limit",
       https://www.cvxportfolio.com/en/1.5.0/constraints.html
    """
    values = asset_array(position_values).astype(float, copy=False)
    return wrap_reduction(position_values, np.sum(np.abs(values), axis=-1))


def net_exposure(position_values):
    """Return net exposure as the signed sum of position values.

    Parameters
    ----------
    position_values : numpy.ndarray or pandas.Series or pandas.DataFrame
        Signed monetary values, shaped as assets or time by assets.

    Returns
    -------
    float or numpy.ndarray or pandas.Series
        Net exposure in monetary units, excluding separately held cash.

    Notes
    -----
    Missing values propagate through each asset-axis reduction.

    References
    ----------
    .. [1] Boyd et al., "Multi-Period Trading via Convex Optimization",
       https://www.cvxportfolio.com/en/1.5.0/_static/cvx_portfolio.pdf
    """
    values = asset_array(position_values).astype(float, copy=False)
    return wrap_reduction(position_values, np.sum(values, axis=-1))


def gross_leverage(position_values, nav):
    """Return gross exposure divided by positive net asset value.

    Parameters
    ----------
    position_values : numpy.ndarray or pandas.Series or pandas.DataFrame
        Signed monetary values, shaped as assets or time by assets.
    nav : scalar, array-like, or pandas.Series
        Portfolio net asset value.  Matrix inputs accept a scalar or one NAV
        value per time row; a pandas Series aligns to a DataFrame's index.

    Returns
    -------
    float or numpy.ndarray or pandas.Series
        Dimensionless gross leverage.  NaN is returned where NAV is missing
        or nonpositive.

    References
    ----------
    .. [1] Cvxportfolio, "Leverage limit",
       https://www.cvxportfolio.com/en/1.5.0/constraints.html
    """
    values = asset_array(position_values).astype(float, copy=False)
    gross = np.sum(np.abs(values), axis=-1)
    amount = portfolio_amount(position_values, nav, "nav")
    return wrap_reduction(position_values, positive_divide(gross, amount))
