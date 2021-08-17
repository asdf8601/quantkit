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


def reduce_array_wrap(obj, res):
    """Array wrap for reducing functions.

    Allow easly wrap the result of a reducing function in the proper dimension
    of the original array-like object.

    Parameters
    ----------
    obj : pandas or numpy
        Object to access.
    res : array-like or number
        Indexer allowed by `obj`.

    Returns
    -------
    out : array-like or number

    Raises
    ------
    NotImplementedError
        If ``obj`` has higher dimension than 2 or less than 1.
    """

    ndim = obj.ndim

    if ndim == 2:
        new_obj = iloc(obj, 0)
        out = new_obj.__array_wrap__(res)
    elif ndim == 1:
        out = res
    else:
        raise NotImplementedError

    return out
