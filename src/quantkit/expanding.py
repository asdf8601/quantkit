"""Financial statistics applied using an expanding window basis approach."""
from quantkit import decorators
from quantkit.utils import array_wrap
import numpy as np


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


@decorators.numpy2pandas_args_wrapper(0)
def drawup(prices, relative=True, out=None):
    r"""Drawup of a given prices series.

    The drawup of a price process :math:`S` at time :math:`t` is defined as
    the rise of the asset prices from its running minimum up to time
    :math:`t`, the mirror image of the drawdown

    .. math::

        U_{t} = S_{t} - \min_{u \in [0, t]}(S_{u})

    By definition the drawup is a positive series. Passing ``relative=True``
    divides it by the running minimum, giving the rise in parts per unit.
    Where the running minimum is zero the relative drawup is NaN.

    Parameters
    ----------
    prices : array-like
        Series on which the drawup is to be calculated.
    relative : bool, optional, default: True
        Passing True makes the drawup series relative (in parts per unit).
    out : array-like, optional, default: None
        Alternative output array in which to place the result. It must have the
        same shape and buffer length as the expected output but the type will
        be cast if necessary.

    Returns
    -------
    out : array-like
        Drawup series.

    References
    ----------
    .. [1] Jan Vecer - Maximum Drawdown and Directional Trading, Risk 19(12),
       2006. Maximum drawdown and maximum drawup are studied as a pair.
       http://www.stat.columbia.edu/~vecer/maxdrawdown3.pdf

    Examples
    --------
    >>> import numpy as np
    >>> drawup(np.array([4, 2, 3, 6]), relative=False)
    0    0.0
    1    0.0
    2    1.0
    3    4.0
    dtype: float64

    >>> drawup(np.array([4, 2, 3, 6]), relative=True)
    0    0.0
    1    0.0
    2    0.5
    3    2.0
    dtype: float64
    """

    if out is None:
        out = np.zeros_like(prices, float)

    arr = prices.__array__()
    cummin = np.where(np.isnan(arr), np.nan, np.fmin.accumulate(arr))
    np.subtract(arr, cummin, out=out)

    if relative:
        # a zero running minimum has no relative rise: NaN, never inf
        np.divide(out, np.where(cummin == 0, np.nan, cummin), out=out)

    out = array_wrap(prices, out)
    return out
