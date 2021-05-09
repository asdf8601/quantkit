"""Core module.
"""
import numpy as np


def returns(prices, period=1, out=None):
    """Arithmetic Returns.

    Parameters
    ----------
    x : array-like
    period : int

    Returns
    -------
    out : array-like

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

    out = prices.__array_wrap__(out)
    return out


def cum_returns(returns, first_price=None, out=None):
    """Cummulative arithmetic returns.

    Parameters
    ----------
    x : array-like
    period : int

    Returns
    -------
    out : array-like

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

    out = returns.__array_wrap__(out)
    return out


def rebase(prices, base=100, out=None):
    """Rebase prices to start in the same base.

    Parameters
    ----------
    prices : array-like
    base : float
    out : array-like

    Returns
    -------
    out : array-like
    """

    if out is None:
        out = np.zeros_like(prices, float)

    returns(prices, out=out)
    np.add(out, 1, out=out)
    np.nancumprod(out, axis=0, out=out)
    np.multiply(out, base, out=out)

    return out
