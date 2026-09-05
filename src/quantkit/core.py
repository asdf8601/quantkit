"""Core module contains common functions to perform finance analysis."""

import numpy as np

from quantkit.utils import array_wrap


def returns(prices, period=1, out=None):
    """Arithmetic Returns.

    Calculate the returns of a price series.

    Parameters
    ----------
    prices : array-like
        Series on which the returns are to be calculated.
    period : int
        Distance between prices to perform the returns.
    out : array-like, optional
        Alternative output array in which to place the result. It must have the
        same shape and buffer length as the expected output but the type will
        be cast if necessary.

    Returns
    -------
    returns : array-like
        Returns series.

    Examples
    --------
    >>> import numpy as np
    >>> returns(np.array([1, 2, 3]))
    array([nan , 1. , 0.5 ])
    """

    if out is None:
        out = np.zeros_like(prices, float)

    arr = prices.__array__()
    xt0 = arr[:-period]
    xti = arr[period:]

    out[:period] = np.nan
    np.divide(xti, xt0, out=out[period:])
    np.subtract(out, 1, out=out)

    out = array_wrap(prices, out)
    return out


def cum_returns(returns, first_price=None, out=None):
    """Cummulative arithmetic returns.

    Parameters
    ----------
    returns : array-like
        Return series which will be used to perform the calculation.
    first_price : float, optional
        Passing a `first_price` you can control the first element of the
        cummulative returns series performed by this function.
    out : array-like, optional
        Alternative output array in which to place the result. It must have the
        same shape and buffer length as the expected output but the type will
        be cast if necessary.

    Returns
    -------
    cumreturns : array-like
        Cummulative returns series.

    Examples
    --------
    >>> import numpy as np
    >>> cum_returns(np.array([np.nan, 1, 0.5]))
    array([0. , 1. , 2. ])

    Also you can specify the first price:

    >>> cum_returns(returns(np.array([1, 2, 3])), first_price=1)
    array([1. , 2. , 3. ])
    """
    if out is None:
        out = np.zeros_like(returns, float)

    ret_arr = returns.__array__()
    np.add(ret_arr, 1, out=out)
    np.nancumprod(out, axis=0, out=out)

    if first_price is None:
        out -= 1
    else:
        np.multiply(out, first_price, out=out)

    out = array_wrap(returns, out)
    return out


def rebase(prices, base=100, out=None):
    """Rebase prices to start in the same base.

    Parameters
    ----------
    prices : array-like
        Series on which the returns are to be calculated.
    base : float
        `prices` start point.
    out : array-like, optional
        Alternative output array in which to place the result. It must have the
        same shape and buffer length as the expected output but the type will
        be cast if necessary.

    Returns
    -------
    price_rebased : array-like
        Price series rebased.
    """

    if out is None:
        out = np.zeros_like(prices, float)

    returns(prices, out=out)
    np.add(out, 1, out=out)
    np.nancumprod(out, axis=0, out=out)
    np.multiply(out, base, out=out)

    return out
