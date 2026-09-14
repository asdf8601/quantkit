"""Portfolio weights and concentration measures."""

import numpy as np

from quantkit.portfolio._utils import (
    asset_array,
    portfolio_amount,
    positive_divide,
    wrap_assets,
    wrap_reduction,
)


def _row_denominator(values, denominator):
    """Shape a per-row denominator to broadcast across assets."""
    if values.ndim == 2 and np.ndim(denominator) == 1:
        return np.asarray(denominator).reshape(-1, 1)
    return denominator


def nav_weights(position_values, nav):
    """Return signed position values normalized by positive NAV.

    Parameters
    ----------
    position_values : numpy.ndarray or pandas.Series or pandas.DataFrame
        Signed monetary position values, shaped as assets or time by assets.
    nav : scalar, array-like, or pandas.Series
        Portfolio net asset value. Matrix inputs accept a scalar or one NAV
        value per time row; a pandas Series aligns to a DataFrame's time index.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Signed dimensionless weights with the input's container metadata. Rows
        with missing or nonpositive NAV contain NaN.

    Notes
    -----
    NAV weights may include short positions, which have negative weights.
    Cash is excluded unless it is included in ``position_values`` as an asset.

    References
    ----------
    .. [1] U.S. Securities and Exchange Commission, "Net asset value",
       https://www.investor.gov/introduction-investing/investing-basics/glossary/net-asset-value
    """
    values = asset_array(position_values, name="position_values")
    capital = portfolio_amount(position_values, nav, name="nav")
    denominator = _row_denominator(values, capital)
    return wrap_assets(position_values, positive_divide(values, denominator))


def gross_weights(position_values):
    """Return absolute position values normalized by gross exposure.

    Parameters
    ----------
    position_values : numpy.ndarray or pandas.Series or pandas.DataFrame
        Signed monetary position values, shaped as assets or time by assets.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Nonnegative dimensionless weights with the input's container metadata.
        An all-zero or missing gross exposure produces NaN.

    Notes
    -----
    Gross exposure is the sum of absolute position values and excludes cash.
    The returned weights therefore sum to one when gross exposure is positive.

    References
    ----------
    .. [1] Cvxportfolio, "Leverage limit",
       https://www.cvxportfolio.com/en/1.5.0/constraints.html
    """
    values = asset_array(position_values, name="position_values")
    magnitudes = np.abs(values)
    gross = np.sum(magnitudes, axis=-1)
    denominator = _row_denominator(values, gross)
    return wrap_assets(
        position_values, positive_divide(magnitudes, denominator)
    )


def concentration(position_values):
    """Return the Herfindahl concentration of gross position weights.

    Parameters
    ----------
    position_values : numpy.ndarray or pandas.Series or pandas.DataFrame
        Signed monetary position values, shaped as assets or time by assets.

    Returns
    -------
    float or numpy.ndarray or pandas.Series
        The dimensionless sum of squared gross weights. Matrix inputs retain
        their time axis. Undefined gross exposures produce NaN.

    Notes
    -----
    This measure uses absolute position magnitudes, so long and short
    positions contribute equally for equal absolute values.

    References
    ----------
    .. [1] Cvxportfolio, "Leverage limit",
       https://www.cvxportfolio.com/en/1.5.0/constraints.html
    """
    values = asset_array(position_values, name="position_values")
    magnitudes = np.abs(values)
    gross = np.sum(magnitudes, axis=-1)
    denominator = _row_denominator(values, gross)
    weights = positive_divide(magnitudes, denominator)
    return wrap_reduction(position_values, np.sum(weights**2, axis=-1))


def effective_number_of_assets(position_values):
    """Return the inverse Herfindahl effective number of assets.

    Parameters
    ----------
    position_values : numpy.ndarray or pandas.Series or pandas.DataFrame
        Signed monetary position values, shaped as assets or time by assets.

    Returns
    -------
    float or numpy.ndarray or pandas.Series
        ``1 / concentration(position_values)``. Matrix inputs retain their
        time axis. Undefined gross exposures produce NaN.

    Notes
    -----
    Equal gross exposure across ``n`` assets gives an effective number of
    assets equal to ``n``.

    References
    ----------
    .. [1] Cvxportfolio, "Leverage limit",
       https://www.cvxportfolio.com/en/1.5.0/constraints.html
    """
    values = asset_array(position_values, name="position_values")
    magnitudes = np.abs(values)
    gross = np.sum(magnitudes, axis=-1)
    denominator = _row_denominator(values, gross)
    weights = positive_divide(magnitudes, denominator)
    herfindahl = np.sum(weights**2, axis=-1)
    return wrap_reduction(position_values, positive_divide(1.0, herfindahl))
