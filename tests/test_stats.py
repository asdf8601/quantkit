"""Pool of stats tests."""
from quantkit import stats

import pytest
import numpy as np
import pandas as pd


def get_params():
    """Generate bunch of parameters which are used in the testing function.

    Returns
    -------
    generator

    Yields
    ------
    x : array-like
    relative : bool
    expected : array-like
    """

    x_list = [
        # numpy objects
        np.array([1, 2, 3]),
        np.array([1, 0, 2, 3]),
        np.array([1, 2, 3]),
        np.array([np.nan, 1, 2, 3]),
        np.array([[np.nan, 1, 1], [1, np.nan, 3], [3, 3, np.nan]]),
        np.array(
            [
                [np.nan, 1, 1, np.nan],
                [1, np.nan, 3, np.nan],
                [3, 3, np.nan, np.nan],
            ],
        ),
        np.array([np.nan, np.nan, 1, np.nan]),
        np.array([[np.nan], [np.nan], [1], [np.nan]]),
        # pandas objects
        pd.Series([1, 2, 3]),
        pd.Series([1, 0, 2, 3]),
        pd.Series([1, 2, 3]),
        pd.Series([np.nan, 1, 2, 3]),
        pd.DataFrame([[np.nan, 1, 1], [1, np.nan, 3], [3, 3, np.nan]]),
        pd.DataFrame(
            [
                [np.nan, 1, 1, np.nan],
                [1, np.nan, 3, np.nan],
                [3, 3, np.nan, np.nan],
            ],
        ),
        pd.Series([np.nan, np.nan, 1, np.nan]),
        pd.DataFrame([[np.nan], [np.nan], [1], [np.nan]]),
    ]

    relative = [
        # numpy object
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        # pandas object
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
        x_list,
        relative,
        expected,
    )
    return gen


@pytest.mark.parametrize("data, relative, expected", get_params())
def test_total_returns_relative(expected, data, relative):
    obtained = stats.total_returns(data, relative=relative)
    np.testing.assert_almost_equal(expected, obtained)


def test_volatility_1darray():
    rets = np.array([1, 2, 3])
    obtained = stats.volatility(rets)
    expected = 1
    assert obtained == expected, "Wrong!"


def test_volatility_2darray():
    rets = np.array([[1], [2], [3]])
    obtained = stats.volatility(rets)
    expected = 1
    assert obtained == expected, "Wrong!"


def test_volatility_daily():
    """

    ------------    -----------
    input           output
    (returns)       (volatility)
    ------------    -----------
    freq=daily   -> freq=yearly
    freq=monthly -> freq=yearly
    freq=weekly  -> freq=yearly
    freq=daily   -> freq=monthly
    ...
    ------------    -----------

    """
    # -------------------------------------------------------------------------
    # daily case: volatility(rets_bday) -> volatility_montly
    sigma_daily = 0.01
    factor_daily = np.sqrt(1)
    rets = np.array([9.98272839, 10.00005155, 10.00004624])

    obtained = stats.volatility(rets, factor=factor_daily)
    expected = sigma_daily

    np.testing.assert_almost_equal(
        obtained,
        expected,
    )


def test_volatility_monthly():
    """
    ------------    -----------
    input           output
    (returns)       (volatility)
    ------------    -----------
    freq=daily   -> freq=yearly
    freq=monthly -> freq=yearly
    freq=weekly  -> freq=yearly
    freq=daily   -> freq=monthly
    ...
    ------------    -----------

    """
    # -------------------------------------------------------------------------
    # monthly case: volatility(rets_bday) -> volatility_montly
    # FIXME: test still failing
    days_in_a_year = 252
    sigma_monthly = 0.01 * np.sqrt(days_in_a_year / 12)
    factor_monthly = np.sqrt(days_in_a_year / 12)
    rets = np.array([9.98272839, 10.00005155, 10.00004624])

    obtained = stats.volatility(rets, factor=factor_monthly)
    expected = sigma_monthly

    np.testing.assert_almost_equal(obtained, expected)


def test_volatility_yearly():
    """
    ------------    -----------
    input           output
    (returns)       (volatility)
    ------------    -----------
    freq=daily   -> freq=yearly
    freq=monthly -> freq=yearly
    freq=weekly  -> freq=yearly
    freq=daily   -> freq=monthly
    ...
    ------------    -----------
    """
    # -------------------------------------------------------------------------
    # yearly case: volatility(rets_bday) -> volatility_yearly
    days_in_a_year = 252
    sigma_yearly = 0.01 * np.sqrt(252)
    factor_yearly = np.sqrt(days_in_a_year)
    rets = np.array([9.98272839, 10.00005155, 10.00004624])

    obtained = stats.volatility(rets, factor=factor_yearly)
    expected = sigma_yearly

    np.testing.assert_almost_equal(obtained, expected)


def test_drawdown_np():
    data = np.array([4, 3, 1, 1])
    expected = -3
    obtained = stats.drawdown(prices=data, relative=False)
    np.testing.assert_almost_equal(expected, obtained)


def test_drawdown_np_relative():
    data = np.array([4, 3, 1, 1])
    expected = -0.75
    obtained = stats.drawdown(prices=data, relative=True)
    np.testing.assert_almost_equal(expected, obtained)


def test_drawdown_np_multidimensional():
    data = np.array(
        [[1, 2, 3], [4, 9, 0], [-1, 6, 7]]
    )
    expected = np.array([-10, -3, -2])
    obtained = stats.drawdown(prices=data, relative=False)
    np.testing.assert_almost_equal(expected, obtained)


def test_drawdown_np_multidimensional_relative():
    data = np.array([[1, 2, 3], [4, 9, 0], [-1, 6, 7]])
    expected = np.array([-1.1111111111, -0.3333333333, -0.2222222222])
    obtained = stats.drawdown(prices=data, relative=True)
    np.testing.assert_almost_equal(expected, obtained, decimal=10)


def test_drawdown_pd():
    data = pd.Series(np.array([4, 3, 1, 1]))
    expected = -3
    obtained = stats.drawdown(prices=data, relative=False)
    np.testing.assert_almost_equal(expected, obtained)


def test_drawdown_pd_relative():
    data = pd.Series(np.array([4, 3, 1, 1]))
    expected = -0.75
    obtained = stats.drawdown(prices=data, relative=True)
    np.testing.assert_almost_equal(expected, obtained)


def test_drawdown_pd_multidimensional():
    data = pd.DataFrame(
        np.array([[1, 2, 3], [4, 9, 0], [-1, 6, 7]])
    )
    expected = pd.Series(np.array([-10, -3, -2]))
    obtained = stats.drawdown(prices=data, relative=False)
    pd.testing.assert_series_equal(expected, obtained)


def test_drawdown_pd_multidimensional_relative():
    data = pd.DataFrame(
        np.array([[1, 2, 3], [4, 9, 0], [-1, 6, 7]])
    )
    expected = pd.Series(np.array([-1.11111111, -0.33333333, -0.22222222]))
    obtained = stats.drawdown(prices=data, relative=True)

    pd.testing.assert_series_equal(expected, obtained)


def test_drawdown_nan():
    data = np.array([4, 1, 0, 2, np.nan])
    obtained = stats.drawdown(prices=data, relative=False)


def test_drawdown_nan_relative():
    data = np.array([4, 1, 0, 2, np.nan])
    obtained = stats.drawdown(prices=data, relative=True)