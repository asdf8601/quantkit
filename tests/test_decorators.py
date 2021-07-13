# import quantkit as qnt
from quantkit import decorators
import pandas as pd
import numpy as np


def test_numpy2pandas_dataframe():

    @decorators.numpy2pandas_args_wrapper(0)
    def foo(x):
        return x

    obj = pd.DataFrame([1])
    obtained = foo(obj)
    expected = obj

    pd.testing.assert_frame_equal(obtained, expected)


def test_numpy2pandas_series():

    @decorators.numpy2pandas_args_wrapper(0)
    def foo(x):
        return x

    obj = pd.Series([1])
    obtained = foo(obj)
    expected = obj
    pd.testing.assert_series_equal(obtained, expected)


def test_numpy2pandas_numpy_1d():

    @decorators.numpy2pandas_args_wrapper(0)
    def foo(x):
        return x

    obj = np.array([1])
    obtained = foo(obj)
    expected = pd.Series(obj)
    pd.testing.assert_series_equal(obtained, expected)


def test_numpy2pandas_numpy_2d():

    @decorators.numpy2pandas_args_wrapper(0)
    def foo(x):
        return x

    obj = np.array([[1]])
    obtained = foo(obj)
    expected = pd.DataFrame(obj)
    pd.testing.assert_frame_equal(obtained, expected)


def test_numpy2pandas_numpy_2d_first():

    @decorators.numpy2pandas_args_wrapper(0)
    def foo(x, y):
        return x, y

    obj = np.array([[1]])
    obt1, obt2 = foo(obj, obj)
    exp1, exp2 = pd.DataFrame(obj), obj

    pd.testing.assert_frame_equal(obt1, exp1)
    assert obt2 is exp2


def test_numpy2pandas_numpy_2d_second():

    @decorators.numpy2pandas_args_wrapper(1)
    def foo(x, y):
        return x, y

    obj = np.array([[1]])
    obt1, obt2 = foo(obj, obj)
    exp1, exp2 = obj, pd.DataFrame(obj)

    pd.testing.assert_frame_equal(obt2, exp2)
    assert obt1 is exp1
