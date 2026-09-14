"""NumPy and pandas wrappers.

Helpers for adapting function inputs and wrapping results as NumPy or pandas
objects. Use the decorators for transformations that preserve shape, and
``reduce_array_wrap`` for results that reduce one dimension.

Quick example
-------------
Compute one total per DataFrame column and retain the column labels:

>>> import pandas as pd
>>> from quantkit.decorators import reduce_array_wrap
>>> positions = pd.DataFrame({"stocks": [10, 20], "bonds": [5, 15]})
>>> totals = positions.to_numpy().sum(axis=0)
>>> reduce_array_wrap(positions, totals)
stocks    30
bonds     20
dtype: int64

The function reference below describes input conversion, shape-preserving
output wrapping and reduction wrapping.
"""

from functools import wraps

import numpy as np
import pandas as pd

from quantkit.utils import array_wrap


def _np2pd(np_obj):
    if isinstance(np_obj, (pd.Series, pd.DataFrame)):
        # is a pandas object
        pd_obj = np_obj

    elif np_obj.ndim == 1:
        # is a numpy object
        pd_obj = pd.Series(np_obj)

    elif np_obj.ndim == 2:
        # is a numpy object
        pd_obj = pd.DataFrame(np_obj)
    else:
        raise NotImplementedError

    return pd_obj


def numpy2pandas_args_wrapper(*pos):
    """Convert selected positional NumPy inputs to pandas objects.

    One-dimensional arrays become Series and two-dimensional arrays become
    DataFrames, both with default integer labels. Existing pandas inputs pass
    through unchanged. This decorator does not convert the return value.

    Parameters
    ----------
    *pos : int
        Zero-based positions of arguments to convert. Selected arguments must
        be passed positionally; keyword arguments are not converted.

    Returns
    -------
    decorated_func : function
        Decorator that converts selected inputs before calling the function.

    Raises
    ------
    NotImplementedError
        If a selected NumPy input has a dimension other than one or two.

    Examples
    --------
    >>> import numpy as np
    >>> @numpy2pandas_args_wrapper(0)
    ... def column_totals(values):
    ...     return values.sum()
    >>> column_totals(np.array([[1, 2], [3, 4]]))
    0    4
    1    6
    dtype: int64
    """

    def maker(func):
        @wraps(func)
        def deco(*args, **kwargs):
            new_args = []
            for arg_i, arg in enumerate(args):
                if arg_i in pos:
                    arg = _np2pd(arg)
                new_args.append(arg)

            return func(*new_args, **kwargs)

        return deco

    return maker


# TODO rename this: only wraps TS(n) -> TS(n)
# foo(x: np.array) -> np.array
# dec(foo(x: pd.DataFrame)) -> pd.DataFrame NO!!!
# dec(foo(x: pd.DataFrame)) -> np.array  SI!!!
# BUG: more than one output will fail
def array_output_wrapper(pos):
    """Wrap a result using a selected input's container and labels.

    Use this decorator for a function that returns one array-like result with
    its input's shape. For pandas inputs, preserve the index, DataFrame columns
    and Series name. For NumPy inputs, return a NumPy array.

    Parameters
    ----------
    pos : int
        Zero-based position of the input whose container and labels to use.
        This input must be passed positionally.

    Returns
    -------
    decorated_func : function
        Decorator that wraps the function's result using the selected input.

    Notes
    -----
    The function must return a single array-like result. Multiple return values
    are not supported. For dimension-reducing results, use
    ``reduce_array_wrap`` instead.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> @array_output_wrapper(0)
    ... def double(values):
    ...     return np.asarray(values) * 2
    >>> double(pd.Series([1, 2], index=["a", "b"], name="positions"))
    a    2
    b    4
    Name: positions, dtype: int64
    >>> double(np.array([1, 2]))
    array([2, 4])
    """

    def maker(func):
        @wraps(func)
        def deco(*args, **kwargs):
            argument = args[pos]
            out = func(*args, **kwargs)
            return array_wrap(argument, out)

        return deco

    return maker


def reduce_array_wrap(obj, res):
    """Wrap a reduction result using the original input's dimensions.

    For a DataFrame, return a Series indexed by its columns. For a
    two-dimensional NumPy input, return ``res`` as a NumPy array. For a
    one-dimensional input, return ``res`` unchanged, typically a scalar.
    This helper wraps an already computed result; it does not perform the
    reduction.

    Parameters
    ----------
    obj : numpy.ndarray or pandas.Series or pandas.DataFrame
        Original one- or two-dimensional input, used to determine the output
        container and labels.
    res : array-like or scalar
        Computed reduction result. For a DataFrame, supply one value per
        column in the original column order.

    Returns
    -------
    out : pandas.Series or numpy.ndarray or scalar
        Wrapped result for two-dimensional inputs, or ``res`` unchanged for
        one-dimensional inputs.

    Raises
    ------
    NotImplementedError
        If ``obj`` has a dimension other than one or two.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> values = np.array([[1, 2], [3, 4]])
    >>> reduce_array_wrap(values, values.sum(axis=0))
    array([4, 6])
    >>> frame = pd.DataFrame(values, columns=["a", "b"])
    >>> reduce_array_wrap(frame, values.sum(axis=0))
    a    4
    b    6
    dtype: int64
    >>> series = pd.Series([1, 2, 3])
    >>> int(reduce_array_wrap(series, series.sum()))
    6
    """
    ndim = obj.ndim

    if ndim == 2:
        # built from the columns, not from a row: a zero-row object has no
        # row to borrow the container from, but it still has its columns
        if isinstance(obj, pd.DataFrame):
            out = pd.Series(res, index=obj.columns, copy=False)
        else:
            out = np.asarray(res)
    elif ndim == 1:
        out = res
    else:
        raise NotImplementedError

    return out
