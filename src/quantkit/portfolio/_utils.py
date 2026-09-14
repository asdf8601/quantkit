"""Validation and container helpers for portfolio calculations."""

import numpy as np
import pandas as pd


def _validate_labels(obj, name):
    """Reject duplicate pandas labels in an asset input."""
    if isinstance(obj, pd.Series) and obj.index.has_duplicates:
        raise ValueError(f"{name} has duplicate labels")
    if isinstance(obj, pd.DataFrame):
        if obj.index.has_duplicates:
            raise ValueError(f"{name} has duplicate index labels")
        if obj.columns.has_duplicates:
            raise ValueError(f"{name} has duplicate column labels")


def _real_array(obj, name):
    """Return a finite real numeric array, allowing missing values."""
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        dtypes = obj.dtypes if isinstance(obj, pd.DataFrame) else [obj.dtype]
        if any(dtype.kind not in "iuf" for dtype in dtypes):
            raise TypeError(f"{name} must contain real numeric values")
        array = obj.to_numpy(dtype=float, na_value=np.nan)
    else:
        array = np.asarray(obj)
        if array.dtype.kind not in "iuf":
            raise TypeError(f"{name} must contain real numeric values")
        array = array.astype(float, copy=False)
    if np.any(np.isinf(array)):
        raise ValueError(f"{name} must not contain infinity")
    return array


def asset_array(obj, name="values"):
    """Validate asset data and return its NumPy representation.

    Parameters
    ----------
    obj : numpy.ndarray, pandas.Series, or pandas.DataFrame
        One-dimensional assets or a time-by-asset matrix.
    name : str, default "values"
        Input name used in validation errors.

    Returns
    -------
    numpy.ndarray
        The validated one- or two-dimensional asset values.

    Raises
    ------
    TypeError
        If ``obj`` is not a supported container or has non-real values.
    ValueError
        If axes are empty, labels are duplicated, or values include infinity.
    """
    if not isinstance(obj, (np.ndarray, pd.Series, pd.DataFrame)):
        raise TypeError(f"{name} must be a numpy array or pandas object")
    _validate_labels(obj, name)
    array = _real_array(obj, name)
    if array.ndim not in (1, 2):
        raise ValueError(f"{name} must be one- or two-dimensional")
    if 0 in array.shape:
        raise ValueError(f"{name} must not have an empty axis")
    return array


def _same_labels(reference, other, axis, name):
    """Require equal unique labels and return ``other`` in reference order."""
    reference_labels = getattr(reference, axis)
    other_labels = (
        other.index if isinstance(other, pd.Series) else getattr(other, axis)
    )
    if (
        len(reference_labels) != len(other_labels)
        or not reference_labels.isin(other_labels).all()
    ):
        raise ValueError(f"{name} labels must match the reference")
    if isinstance(other, pd.Series):
        return other.reindex(reference_labels)
    return other.reindex(**{axis: reference_labels})


def pair_assets(reference, other):
    """Validate and align paired asset inputs.

    The second input may be an asset vector for a matrix reference, in which
    case it is broadcast over time. Two pandas inputs align by their asset
    labels and, for DataFrames, by their time labels.

    Parameters
    ----------
    reference : numpy.ndarray, pandas.Series, or pandas.DataFrame
        The input that supplies the result shape and pandas axes.
    other : numpy.ndarray, pandas.Series, or pandas.DataFrame
        Values paired with ``reference``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Reference values and aligned values.

    Raises
    ------
    ValueError
        If dimensions, shapes, or pandas labels are incompatible.
    """
    arr_reference = asset_array(reference, "reference")
    arr_other = asset_array(other, "other")

    if arr_reference.ndim == 1:
        if arr_other.ndim != 1 or arr_other.shape != arr_reference.shape:
            raise ValueError("asset vectors must have the same shape")
    elif arr_other.ndim == 2 and arr_other.shape != arr_reference.shape:
        raise ValueError("asset matrices must have the same shape")
    elif arr_other.ndim == 1 and arr_other.shape[0] != arr_reference.shape[1]:
        raise ValueError("asset vector must match the reference asset count")
    elif arr_other.ndim != 1 and arr_other.ndim != 2:
        raise ValueError("other must be one- or two-dimensional")

    if isinstance(reference, pd.DataFrame) and isinstance(other, pd.DataFrame):
        other = _same_labels(reference, other, "index", "time")
        other = _same_labels(reference, other, "columns", "asset")
    elif isinstance(reference, pd.DataFrame) and isinstance(other, pd.Series):
        other = _same_labels(reference, other, "columns", "asset")
    elif isinstance(reference, pd.Series) and isinstance(other, pd.Series):
        other = _same_labels(reference, other, "index", "asset")

    arr_other = _real_array(other, "other")
    if arr_reference.ndim == 2 and arr_other.ndim == 1:
        arr_other = np.broadcast_to(arr_other, arr_reference.shape)
    return arr_reference, arr_other


def wrap_assets(like, values):
    """Wrap asset-shaped values in the container and axes of ``like``.

    Parameters
    ----------
    like : numpy.ndarray, pandas.Series, or pandas.DataFrame
        Input whose container metadata is preserved.
    values : array-like
        Asset-shaped values to wrap.

    Returns
    -------
    numpy.ndarray, pandas.Series, or pandas.DataFrame
        Values in the same container type as ``like``.
    """
    if isinstance(like, pd.DataFrame):
        return pd.DataFrame(
            values, index=like.index, columns=like.columns, copy=False
        )
    if isinstance(like, pd.Series):
        return pd.Series(values, index=like.index, name=like.name, copy=False)
    return np.asarray(values)


def wrap_reduction(like, values):
    """Wrap an asset-axis reduction according to its input container.

    Parameters
    ----------
    like : numpy.ndarray, pandas.Series, or pandas.DataFrame
        Input whose time axis is preserved for a DataFrame.
    values : array-like
        Reduction values over the final asset axis.

    Returns
    -------
    float, numpy.ndarray, or pandas.Series
        A scalar for vectors, time values for matrices, or a time-indexed
        Series for DataFrames.
    """
    if isinstance(like, pd.DataFrame):
        return pd.Series(values, index=like.index, name=None, copy=False)
    if like.ndim == 1:
        result = np.asarray(values)
        if result.size != 1:
            raise ValueError("vector reductions must produce one value")
        return float(result.reshape(-1)[0])
    if like.ndim == 2:
        return np.asarray(values)
    raise ValueError("like must be one- or two-dimensional")


def portfolio_amount(like, amount, name="amount"):
    """Validate and align a scalar or time-indexed portfolio amount.

    Parameters
    ----------
    like : numpy.ndarray, pandas.Series, or pandas.DataFrame
        Asset input determining whether an amount is scalar or time-indexed.
    amount : scalar, array-like, or pandas.Series
        Scalar amount, or one amount per row of a matrix input.
    name : str, default "amount"
        Input name used in validation errors.

    Returns
    -------
    float or numpy.ndarray
        A scalar for vector inputs; a scalar or length-time array for matrix
        inputs.
    """
    array = asset_array(like, "like")
    if isinstance(amount, pd.Series):
        _validate_labels(amount, name)
        if isinstance(like, pd.DataFrame):
            amount = _same_labels(like, amount, "index", name)
        elif array.ndim == 1:
            raise ValueError(f"{name} must be scalar for vector inputs")

    values = _real_array(amount, name)
    if values.ndim == 0:
        return float(values)
    if (
        array.ndim == 2
        and values.ndim == 1
        and values.shape[0] == array.shape[0]
    ):
        return values
    if array.ndim == 1:
        raise ValueError(f"{name} must be scalar for vector inputs")
    raise ValueError(f"{name} must be scalar or match the time axis")


def positive_divide(numerator, denominator):
    """Divide where the denominator is positive, returning NaN otherwise.

    Parameters
    ----------
    numerator : array-like
        Values to divide.
    denominator : array-like
        Positive divisors.

    Returns
    -------
    float or numpy.ndarray
        Quotients, with NaN where the denominator is nonpositive or missing.
    """
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    result = np.full(numerator.shape, np.nan, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    if result.ndim == 0:
        return float(result)
    return result
