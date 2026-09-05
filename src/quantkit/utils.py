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
