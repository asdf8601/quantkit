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
