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
    else:
        raise NotImplementedError

    if isinstance(array, np.ndarray) and (ndim == 2):
        out = out.values

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
    else:
        raise NotImplementedError

    if isinstance(array, np.ndarray) and (ndim == 2):
        out = out.values

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

    if isinstance(obj, pd.core.generic.NDFrame):
        out = obj.iloc[idx]
        out.name = None
    elif isinstance(obj, np.ndarray):
        out = obj[idx]
    else:
        raise NotImplementedError

    return out
