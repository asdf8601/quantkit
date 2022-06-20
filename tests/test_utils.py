"""Test utilities module."""
import pandas as pd
import numpy as np
from quantkit.decorators import reduce_array_wrap


def test_reduce_pandas_nx1():
    df = pd.DataFrame([1, 2, 3])
    res = np.array([1])

    obtained = reduce_array_wrap(df, res)
    expected = pd.Series([1])

    pd.testing.assert_series_equal(obtained, expected)


def test_reduce_pandas_nx2():
    df = pd.DataFrame([[1, 1], [2, 2], [3, 3]])
    res = np.array([1, 1])

    obtained = reduce_array_wrap(df, res)
    expected = pd.Series([1, 1])

    pd.testing.assert_series_equal(obtained, expected)


def test_reduce_pandas_n():
    df = pd.Series([1, 2, 3])
    res = 1

    obtained = reduce_array_wrap(df, res)
    expected = 1

    np.testing.assert_almost_equal(obtained, expected)


def test_reduce_numpy_nx1():
    df = np.array([[1], [2], [3]])
    res = np.array([1])

    obtained = reduce_array_wrap(df, res)
    expected = np.array([1])

    np.testing.assert_almost_equal(obtained, expected)


def test_reduce_numpy_nx2():
    df = np.array([[1, 1], [2, 2], [3, 3]])
    res = np.array([1, 1])

    obtained = reduce_array_wrap(df, res)
    expected = np.array([1, 1])

    np.testing.assert_almost_equal(obtained, expected)


def test_reduce_numpy_n():

    df = np.array([1, 2, 3])
    res = 1

    obtained = reduce_array_wrap(df, res)
    expected = 1

    np.testing.assert_almost_equal(obtained, expected)
