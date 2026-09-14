"""Weighted portfolio return calculations."""

import numpy as np

from ._utils import pair_assets, wrap_reduction


def portfolio_returns(asset_returns, beginning_weights):
    """Aggregate asset returns using weights at each interval's beginning.

    The caller supplies weights valid at the start of every return interval.
    This function does not shift weights, normalize them, or infer a cash
    position; include cash as an explicit asset when required.

    Parameters
    ----------
    asset_returns : numpy.ndarray, pandas.Series, or pandas.DataFrame
        Per-asset returns. Matrix rows are return intervals and columns are
        assets.
    beginning_weights : numpy.ndarray, pandas.Series, or pandas.DataFrame
        Asset weights at the beginning of each corresponding interval. An
        asset vector broadcasts over matrix ``asset_returns``.

    Returns
    -------
    float, numpy.ndarray, or pandas.Series
        The weighted return for each interval, retaining the shape and time
        labels of ``asset_returns``.

    References
    ----------
    .. [1] https://www.cvxportfolio.com/en/1.5.0/_static/cvx_portfolio.pdf
    """
    returns, weights = pair_assets(asset_returns, beginning_weights)
    return wrap_reduction(asset_returns, np.sum(returns * weights, axis=-1))


def expected_return(expected_returns, weights):
    """Aggregate per-asset expected returns using asset weights.

    Expected returns and weights must use the same return frequency. This
    function does not annualize inputs, normalize weights, or infer cash.

    Parameters
    ----------
    expected_returns : numpy.ndarray, pandas.Series, or pandas.DataFrame
        Per-asset expected returns. Matrix rows may represent separate
        intervals or scenarios.
    weights : numpy.ndarray, pandas.Series, or pandas.DataFrame
        Asset weights for the corresponding expected returns. An asset vector
        broadcasts over matrix ``expected_returns``.

    Returns
    -------
    float, numpy.ndarray, or pandas.Series
        Weighted expected return values, retaining the shape and time labels
        of ``expected_returns``.

    References
    ----------
    .. [1] https://www.cvxportfolio.com/en/1.5.0/_static/cvx_portfolio.pdf
    """
    returns, asset_weights = pair_assets(expected_returns, weights)
    return wrap_reduction(
        expected_returns, np.sum(returns * asset_weights, axis=-1)
    )
