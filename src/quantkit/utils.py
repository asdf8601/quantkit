"""Module of utilities."""

import numpy as np
import pandas as pd


def first_valid_index(array):
    """Find the first no null value and returns its index.

    Parameters
    ----------
    array : array-like
        Numpy or pandas (or array-like) object.

    Returns
    -------
    index : int
        The first position valid.

    Raises
    ------
    NotImplementedError
        If array has higher dimension than 2.
    """
    ndim = len(array.shape)

    if ndim == 1:
        out = pd.Series(array).first_valid_index()
    elif ndim == 2:
        out = pd.DataFrame(array).apply(lambda col: col.first_valid_index())
        if isinstance(array, np.ndarray):
            out = out.values
    else:
        raise NotImplementedError

    return out


def last_valid_index(array):
    """Find the last no null value and returns its index.

    Parameters
    ----------
    array : array-like
        Numpy or pandas (or array-like) object.

    Returns
    -------
    index : int
        The last position valid.

    Raises
    ------
    NotImplementedError
        If array has higher dimension than 2.
    """
    ndim = len(array.shape)

    if ndim == 1:
        out = pd.Series(array).last_valid_index()
    elif ndim == 2:
        out = pd.DataFrame(array).apply(lambda col: col.last_valid_index())
        if isinstance(array, np.ndarray):
            out = out.values
    else:
        raise NotImplementedError

    return out


def iloc(obj, idx):
    """Access to idx in obj.

    This is a wrapper to access in the same way to pandas and numpy objects.

    Parameters
    ----------
    obj : pandas or numpy
        Object to access.
    idx : indexer
        Indexer allowed by `obj`.

    Returns
    -------
    obj_sliced : pandas or numpy

    Raises
    ------
    NotImplementedError
        If `obj` is not a pandas or numpy object handled by this function.
    """

    if isinstance(obj, (pd.Series, pd.DataFrame)):
        out = obj.iloc[idx]
        out.name = None
    elif isinstance(obj, np.ndarray):
        out = obj[idx]
    else:
        raise NotImplementedError

    return out


def array_wrap(like, values):
    """Wrap ``values`` in the same container type as ``like``.

    Public replacement for ``like.__array_wrap__(values)``, removed from
    pandas 2.0. Pandas objects are rebuilt with the axes and name of ``like``
    without copying ``values``; numpy objects keep numpy's own protocol.

    Parameters
    ----------
    like : pandas.Series, pandas.DataFrame or numpy.ndarray
        Object whose type and axes are preserved.
    values : array-like
        Data to wrap. Must match the shape of ``like``.

    Returns
    -------
    out : same type as ``like``
    """
    if isinstance(like, pd.DataFrame):
        return pd.DataFrame(
            values, index=like.index, columns=like.columns, copy=False
        )
    if isinstance(like, pd.Series):
        return pd.Series(values, index=like.index, name=like.name, copy=False)
    return like.__array_wrap__(np.asarray(values))


def align(returns, benchmark):
    """Align ``returns`` with a 1D ``benchmark`` and return both as ndarrays.

    Pandas objects are inner-joined on their index, so only the labels known
    on both sides survive. As soon as one side is a numpy array there is no
    index to join on and both inputs must have the same length.

    Parameters
    ----------
    returns : pandas.Series, pandas.DataFrame or numpy.ndarray
        1D or 2D returns.
    benchmark : pandas.Series or numpy.ndarray
        1D benchmark returns.

    Returns
    -------
    arr_returns : numpy.ndarray
        Aligned returns, with the same number of dimensions as ``returns``.
    arr_benchmark : numpy.ndarray
        Aligned 1D benchmark.

    Raises
    ------
    ValueError
        If ``benchmark`` is not 1D, or if either input is a numpy array and
        the lengths differ.

    Examples
    --------
    >>> returns = pd.Series([1.0, 2.0, 3.0], index=[1, 2, 3])
    >>> benchmark = pd.Series([10.0, 20.0, 30.0], index=[0, 1, 2])
    >>> align(returns, benchmark)
    (array([1., 2.]), array([20., 30.]))
    """
    if benchmark.ndim != 1:
        raise ValueError(
            f"benchmark must be 1D, got {benchmark.ndim} dimensions"
        )

    is_pandas = isinstance(returns, (pd.Series, pd.DataFrame))
    if is_pandas and isinstance(benchmark, pd.Series):
        returns, benchmark = returns.align(benchmark, join="inner", axis=0)
    elif len(returns) != len(benchmark):
        raise ValueError(
            "returns and benchmark must have the same length when either "
            f"is a numpy array, got {len(returns)} and {len(benchmark)}"
        )

    return returns.__array__(), benchmark.__array__()
