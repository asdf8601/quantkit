"""Test utilities module."""
import pandas as pd
import numpy as np
from quantkit.decorators import reduce_array_wrap
from quantkit.utils import array_wrap


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


def test_array_wrap_series_keeps_index_and_name():
    like = pd.Series(
        [1.0, 2.0, 3.0], index=pd.date_range("2020", periods=3), name="px"
    )
    values = np.array([10.0, 20.0, 30.0])

    obtained = array_wrap(like, values)
    expected = pd.Series(values, index=like.index, name="px")

    pd.testing.assert_series_equal(obtained, expected)


def test_array_wrap_dataframe_keeps_index_and_columns():
    like = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.date_range("2020", periods=2),
        columns=["a", "b"],
    )
    values = np.array([[10.0, 20.0], [30.0, 40.0]])

    obtained = array_wrap(like, values)
    expected = pd.DataFrame(values, index=like.index, columns=like.columns)

    pd.testing.assert_frame_equal(obtained, expected)


def test_array_wrap_numpy_returns_numpy():
    like = np.array([1.0, 2.0, 3.0])
    values = np.array([10.0, 20.0, 30.0])

    obtained = array_wrap(like, values)

    assert isinstance(obtained, np.ndarray)
    np.testing.assert_array_equal(obtained, values)


def test_array_wrap_does_not_copy():
    like = pd.Series([1.0, 2.0, 3.0])
    values = np.array([10.0, 20.0, 30.0])

    obtained = array_wrap(like, values)

    assert np.shares_memory(obtained.to_numpy(), values)


def test_reduce_zero_row_numpy_gives_one_value_per_column():
    arr = np.empty((0, 2))
    res = np.array([np.nan, np.nan])

    obtained = reduce_array_wrap(arr, res)

    assert isinstance(obtained, np.ndarray)
    assert obtained.shape == (2,)  # one value per column, no row indexed
    np.testing.assert_array_equal(obtained, res)


def test_reduce_zero_row_dataframe_is_indexed_by_the_columns():
    df = pd.DataFrame(columns=["a", "b"], dtype=float)
    res = np.array([np.nan, np.nan])

    obtained = reduce_array_wrap(df, res)
    expected = pd.Series([np.nan, np.nan], index=["a", "b"])

    assert obtained.name is None
    pd.testing.assert_series_equal(obtained, expected)
