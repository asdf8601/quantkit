"""Financial statistics applied using an expanding window basis approach."""

import numpy as np

from quantkit import decorators
from quantkit.utils import array_wrap


@decorators.numpy2pandas_args_wrapper(0)
def drawdown(prices, relative=True, out=None):
    r"""Drawdown of a given prices series.

    The drawdown of a price process :math:`S` at time :math:`t` is defined as
    the drop of the asset prices from its running maximum up to time :math:`t`

    .. math::

        D_{t} = \max_{u \in [0, t]}(S_{u}) - S_{t}

    This means that, by definition, the drawdown should be a strictly positive
    series. However, the results are often presented as a negative series for
    better understanding, and that is the way we chose to implement here.

    .. math::

        D_{t} = S_{t} - \max_{u \in [0, t]}(S_{u})

    Parameters
    ----------
    prices : array-like
        Series on which the drawdown is to be calculated.
    relative : bool, optional, default: False
        Passing True makes the drawdown series relative (in parts per unit).
    out : array-like, optional, default: None
        Alternative output array in which to place the result. It must have the
        same shape and buffer length as the expected output but the type will
        be cast if necessary.

    Returns
    -------
    out : array-like
        Drawdown series.

    References
    ----------
    .. [1] Jan Vecer - Maximum Drawdown and Directional Trading
       http://www.stat.columbia.edu/~vecer/maxdrawdown3.pdf

    .. [2] Enrico Schumann - Computing Drawdown Statistics
       http://comisef.wikidot.com/tutorial:drawdowns
    """

    if out is None:
        out = np.zeros_like(prices, float)

    arr = prices.__array__()
    cummax = np.where(np.isnan(arr), np.nan, np.fmax.accumulate(arr))
    np.subtract(arr, cummax, out=out)

    if relative:
        np.divide(out, cummax, out=out)

    out = array_wrap(prices, out)
    return out
