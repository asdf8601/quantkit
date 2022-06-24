"""Decorators module.

f(x) -> y

x : pandas.DataFrame, pandas.Series, np.array (, list)

f(x): x -> array_protocol -> x_arr -> do stuff -> out : the same type as x

Ej1
x : np.array[n,m]
y : np.array[m]

Ej2
x : pandas.DataFrame
y : pandas.Series

Ej3
x : np.array[m]
y : float

f(x): x -> array_protocol -> x_arr -> do stuff -> out : the same type as x

intput to {pandas, numpy} -> output to {pandas, numpy}:

- array_wrap_reduce  (collapse one dimension)
- array_wrap_transform  (non-collapsing)
- array_wrap_increase  (increse one dimension)
"""
import pandas as pd
from quantkit.utils import iloc
import inspect
from functools import wraps


def _np2pd(np_obj):
    if isinstance(np_obj, pd.core.generic.NDFrame):
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
    """Convert numpy to pandas output objects according to an input parameter.

    This decorator receive positions of the parameters in the decorated
    function to pick the numpy object and convert it to pandas preserving the
    orignal dimension.

    Parameters
    ----------
    *pos : int
        Parameter position in the decorated function which will be converted to
        pandas.

    Returns
    -------
    decorated_func : function
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
def array_output_wrapper(pos, name):
    """Preserves output according to the input of the decorated functions.

    Decorator which converts the decorated function's output in the same object
    as one of the input parameters. This allow us preserve the same object type
    in the output.

    Parameters
    ----------
    pos : int
        Parameter position of the decorated function to preserve object type
        from.

    Returns
    -------
    decorated_func : function
    """
    def maker(func):
        @wraps(func)
        def deco(*args, **kwargs):
            try:
                argument = args[pos]
            except IndexError:
                argument = kwargs[name]
            out = func(*args, **kwargs)
            return argument.__array_wrap__(out)

        return deco

    return maker


def reduced_array_out(pos, name):
    """Preserves output according to the input of the decorated functions.

    Decorator which converts the decorated function's output in the same object
    as one of the input parameters. This allow us preserve the same object type
    in the output.

    Parameters
    ----------
    pos : int
        Parameter position of the decorated function to preserve object type
        from.

    Returns
    -------
    decorated_func : function
    """
    def maker(func):
        @wraps(func)
        def deco(*args, **kwargs):
            try:
                argument = args[pos]
            except IndexError:
                argument = kwargs[name]
            out = func(*args, **kwargs)
            return reduce_array_wrap(argument, out)

        return deco

    return maker


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


    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd

    Numpy objects

    >>> obj = np.array([[0], [1]])
    >>> obj_reduced = np.array([22])
    >>> reduce_array_wrap(obj, obj_reduced)

    Pandas objects

    >>> obj = pd.DataFrame(np.array([[0], [1]]))
    >>> obj_reduced = np.array([22])
    >>> reduce_array_wrap(obj, obj_reduced)

    More realistic example

    >>> obj = pd.DataFrame(np.array([[0], [1]]))
    >>> obj_reduced = obj.sum()  # apply a reduction function
    >>> reduce_array_wrap(obj, obj_reduced)
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
