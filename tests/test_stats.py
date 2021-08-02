"""Pool of stats tests."""
from quantkit import stats
import numpy as np
import pandas as pd
import pytest
"""
TODO:

Include this behaviour in `total_returns`

Given the following input:

       col1 , col2 , col3
       ---  , ---  , ---
  0    nan  , 1    , 2
  1    nan  , nan  , 2
  2    1    , 1    , nan
  3    4    , 1    , nan

What it matters is:
         first     last
        ------     ----
- col1 : row 2 and row 3
- col2 : row 0 and row 3
- col3 : row 0 and row 1


Idea:

msk = np.isnan(arr).cumsum()

msk.min() # first
msk.max() # last

"""


def params():

    x = [
        np.array([1, 2, 3]),
        np.array([1, 0, 2, 3]),
        np.array([1, 2, 3]),
        np.array([np.nan, 1, 2, 3]),
        np.array([[np.nan, 1, 1], [1, np.nan, 3], [3, 3, np.nan]]),
        np.array([[np.nan, 1, 1, np.nan], [1, np.nan, 3, np.nan], [3, 3, np.nan, np.nan]]),
        np.array([np.nan, np.nan, 1, np.nan]),
        np.array([[np.nan], [np.nan], [1], [np.nan]]),
        pd.Series([1, 2, 3]),
        pd.Series([1, 0, 2, 3]),
        pd.Series([1, 2, 3]),
        pd.Series([np.nan, 1, 2, 3]),
        pd.DataFrame([[np.nan, 1, 1], [1, np.nan, 3], [3, 3, np.nan]]),
        pd.DataFrame([[np.nan, 1, 1, np.nan], [1, np.nan, 3, np.nan], [3, 3, np.nan, np.nan]]),
        pd.Series([np.nan, np.nan, 1, np.nan]),
        pd.DataFrame([[np.nan], [np.nan], [1], [np.nan]]),
    ]

    relative = [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    ]

    expected = [
        2.0,
        2.0,
        2.0,
        2.0,
        np.array([2.0, 2.0, 2.0]),
        np.array([2.0, 2.0, 2.0, np.nan]),
        np.nan,
        np.array([np.nan]),
        2.0,
        2.0,
        2.0,
        2.0,
        pd.Series([2.0, 2.0, 2.0]),
        pd.Series([2.0, 2.0, 2.0, np.nan]),
        np.nan,
        pd.Series([np.nan]),
    ]

    gen = zip(
        x,
        relative,
        expected,
    )
    return gen


@pytest.mark.parametrize("x, relative, expected", params())
def test_total_returns_relative(expected, x, relative):
    obtained = stats.total_returns(x, relative=relative)
    np.testing.assert_almost_equal(expected, obtained)
