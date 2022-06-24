"""Pool of stats tests."""
from quantkit import stats

import pytest
import numpy as np
import pandas as pd


def assert_array_like(obtained, expected, *args, **kwargs):
    __tracebackhide__ = True
    if isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(obtained, expected, *args, **kwargs)
    elif isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(obtained, expected, *args, **kwargs)
    else:
        np.testing.assert_almost_equal(expected, obtained, *args, **kwargs)


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
    assert_array_like(obtained, expected)


def test_volatility_1darray():
    rets = np.array([1, 2, 3])
    obtained = stats.volatility(rets)
    expected = 1
    assert_array_like(obtained, expected)


def test_volatility_2darray():
    rets = np.array([[1], [2], [3]])
    obtained = stats.volatility(rets)
    expected = 1
    assert_array_like(obtained, expected)


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

    assert_array_like(obtained, expected)


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

    assert_array_like(obtained, expected)


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

    assert_array_like(obtained, expected)


def test_drawdown_np():
    data = np.array([4, 3, 1, 1])
    expected = -3
    obtained = stats.drawdown(prices=data, relative=False)
    assert_array_like(obtained, expected)


def test_drawdown_np_relative():
    data = np.array([4, 3, 1, 1])
    expected = -0.75
    obtained = stats.drawdown(prices=data, relative=True)
    assert_array_like(obtained, expected)


def test_drawdown_np_multidimensional():
    data = np.array([[1, 2, 3], [4, 9, 0], [-1, 6, 7]])
    expected = np.array([-10, -3, -2])
    obtained = stats.drawdown(prices=data, relative=False)
    assert_array_like(obtained, expected)


def test_drawdown_np_multidimensional_relative():
    data = np.array([[1, 2, 3], [4, 9, 0], [-1, 6, 7]])
    expected = np.array([-1.1111111111, -0.3333333333, -0.2222222222])
    obtained = stats.drawdown(prices=data, relative=True)
    assert_array_like(obtained, expected)


def test_drawdown_pd():
    data = pd.Series(np.array([4, 3, 1, 1]))
    expected = -3
    obtained = stats.drawdown(prices=data, relative=False)
    assert_array_like(obtained, expected)


def test_drawdown_pd_relative():
    data = pd.Series(np.array([4, 3, 1, 1]))
    expected = -0.75
    obtained = stats.drawdown(prices=data, relative=True)
    assert_array_like(obtained, expected)


def test_drawdown_pd_multidimensional():
    data = pd.DataFrame(np.array([[1, 2, 3], [4, 9, 0], [-1, 6, 7]]))
    expected = pd.Series(np.array([-10, -3, -2]))
    obtained = stats.drawdown(prices=data, relative=False)
    assert_array_like(obtained, expected)


def test_drawdown_pd_multidimensional_relative():
    data = pd.DataFrame(np.array([[1, 2, 3], [4, 9, 0], [-1, 6, 7]]))
    expected = pd.Series(np.array([-1.11111111, -0.33333333, -0.22222222]))
    obtained = stats.drawdown(prices=data, relative=True)

    assert_array_like(obtained, expected)


def test_drawdown_nan():
    data = np.array([4, 1, 0, 2, np.nan])
    obtained = stats.drawdown(prices=data, relative=False)


def test_drawdown_nan_relative():
    data = np.array([4, 1, 0, 2, np.nan])
    obtained = stats.drawdown(prices=data, relative=True)


def test_max_drawdown_np():
    data = np.array([1, 0, 2])
    obtained = stats.max_drawdown(prices=data, relative=True)
    expected = -1
    assert_array_like(obtained, expected)


def test_max_drawdown_np_0():
    data = np.array([1, 2, 3])
    obtained = stats.max_drawdown(prices=data, relative=True)
    expected = 0
    assert_array_like(obtained, expected)


def test_max_drawdown_np_2d():
    data = np.array([[1, 0, 3]]).T  # .T to make it columnar
    obtained = stats.max_drawdown(prices=data, relative=True)
    expected = np.array([-1])
    assert_array_like(obtained, expected)


def test_max_drawdown_np_0_2d():
    data = np.array([[1, 2, 3]]).T  # .T to make it columnar
    obtained = stats.max_drawdown(prices=data, relative=True)
    expected = np.array([0])
    assert_array_like(obtained, expected)


def test_max_drawdown_pd():
    data = pd.Series([1, 0, 2])
    obtained = stats.max_drawdown(prices=data, relative=True)
    expected = -1
    assert_array_like(obtained, expected)


def test_max_drawdown_pd_0():
    data = pd.Series([1, 2, 3])
    obtained = stats.max_drawdown(prices=data, relative=True)
    expected = 0
    assert_array_like(obtained, expected)


def test_max_drawdown_pd_2d():
    data = pd.DataFrame([[1, 0, 3]]).T  # .T to make it columnar
    obtained = stats.max_drawdown(prices=data, relative=True)
    expected = pd.Series([-1], dtype=float)
    assert_array_like(obtained, expected)


def test_max_drawdown_pd_0_2d():
    data = pd.DataFrame([[1, 2, 3]]).T  # .T to make it columnar
    obtained = stats.max_drawdown(prices=data, relative=True)
    expected = pd.Series([0], dtype=float)
    assert_array_like(obtained, expected)


def test_max_drawdown_np_nan():
    data = np.array([np.nan, 1, 0, 2])
    obtained = stats.max_drawdown(prices=data, relative=True)
    expected = -1
    assert_array_like(obtained, expected)


def test_max_drawdown_np_0_nan():
    data = np.array([np.nan, 1, 2, 3])
    obtained = stats.max_drawdown(prices=data, relative=True)
    expected = 0
    assert_array_like(obtained, expected)


def test_max_drawdown_np_2d_nan():
    data = np.array([[np.nan, 1, 0, 3]]).T  # .T to make it columnar
    obtained = stats.max_drawdown(prices=data, relative=True)
    expected = np.array([-1])
    assert_array_like(obtained, expected)


def test_max_drawdown_np_0_2d_nan():
    data = np.array([[np.nan, 1, 2, 3]]).T  # .T to make it columnar
    obtained = stats.max_drawdown(prices=data, relative=True)
    expected = np.array([0])
    assert_array_like(obtained, expected)


def test_max_drawdown_pd_nan():
    data = pd.Series([np.nan, 1, 0, 2])
    obtained = stats.max_drawdown(prices=data, relative=True)
    expected = -1
    assert_array_like(obtained, expected)


def test_max_drawdown_pd_0_nan():
    data = pd.Series([np.nan, 1, 2, 3])
    obtained = stats.max_drawdown(prices=data, relative=True)
    expected = 0
    assert_array_like(obtained, expected)


def test_max_drawdown_pd_2d_nan():
    data = pd.DataFrame([[np.nan, 1, 0, 3]]).T  # .T to make it columnar
    obtained = stats.max_drawdown(prices=data, relative=True)
    expected = pd.Series([-1], dtype=float)
    assert_array_like(obtained, expected)


def test_max_drawdown_pd_0_2d_nan():
    data = pd.DataFrame([[np.nan, 1, 2, 3]]).T  # .T to make it columnar
    obtained = stats.max_drawdown(prices=data, relative=True)
    expected = pd.Series([0], dtype=float)
    assert_array_like(obtained, expected)


def test_sharpe_ratio():
    ra = np.array([0, 2])
    rb = 0

    obtained = stats.sharpe_ratio(ra, rb, factor=1)
    expected = 1

    assert_array_like(obtained, expected)


def test_sharpe_ratio_dataframe():
    ra = pd.DataFrame(np.array([0, 2]))
    rb = 0

    obtained = stats.sharpe_ratio(ra, rb, factor=1)
    expected = pd.Series([1], dtype=float)

    assert_array_like(obtained, expected)


def test_sortino_ratio():
    # example taken from
    # http://www.redrockcapital.com/Sortino__A__Sharper__Ratio_Red_Rock_Capital.pdf
    rets = np.array([0.17, 0.15, 0.23, -0.05, 0.12, 0.09, 0.13, -0.04])
    bench = 0.0
    obtained = stats.sortino_ratio(rets, bench, factor=1)
    expected = 4.417261042993862
    assert_array_like(obtained, expected)


def test_sortino_ratio_case_idem_rets():
    rets = np.array([-10, -10, -10, -10])
    bench = 0.0
    obtained = stats.sortino_ratio(rets, bench, factor=1)
    expected = -1
    assert_array_like(obtained, expected)


def test_sortino_ratio_series():
    # example taken from
    # http://www.redrockcapital.com/Sortino__A__Sharper__Ratio_Red_Rock_Capital.pdf
    rets = pd.Series(
        np.array([0.17, 0.15, 0.23, -0.05, 0.12, 0.09, 0.13, -0.04])
    )
    bench = 0.0
    obtained = stats.sortino_ratio(rets, bench, factor=1)
    expected = 4.417261042993862
    assert_array_like(obtained, expected)


def test_sortino_ratio_case_idem_rets_series():
    # series -> float
    rets = pd.Series(np.array([-10, -10, -10, -10]))
    bench = 0.0
    obtained = stats.sortino_ratio(rets, bench, factor=1)
    expected = -1
    assert_array_like(obtained, expected)


def test_sortino_ratio_case_idem_rets_dataframe():
    # df -> series
    rets = pd.DataFrame(np.array([-10, -10, -10, -10]))
    bench = 0.0
    obtained = stats.sortino_ratio(rets, bench, factor=1)
    expected = pd.Series([-1], dtype=float)
    assert_array_like(obtained, expected)
