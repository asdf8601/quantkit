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
        def deco(*args, **kwargs):
            argument = args[pos]
            out = func(argument, *args, **kwargs)
            return argument.__array_wrap__(out)

        return deco

    return maker
