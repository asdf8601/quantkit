import pandas as pd
import numpy as np
from quantkit import rolling


def test_volatility_np():
    ret = np.array([0.1, 0.1, 0.1])
    obtained = rolling.volatility(ret, 2)
    expected = np.array([np.nan, 0., 0.])
    np.testing.assert_almost_equal(obtained, expected)


def test_volatility2_np():
    days_in_a_year = 252
    factor_monthly = np.sqrt(days_in_a_year / 12)
    rets = np.array([9.98272839, 10.00005155, 10.00004624])

    obtained = rolling.volatility(
        rets, window=3, min_periods=3, factor=factor_monthly
    )
    sigma_monthly = 0.01 * np.sqrt(days_in_a_year / 12)
    expected = np.array([np.nan, np.nan, sigma_monthly])

    np.testing.assert_almost_equal(obtained, expected)


def test_volatility_pd():
    ret = pd.Series([0.1, 0.1, 0.1])
    obtained = rolling.volatility(ret, 2)
    expected = pd.Series([np.nan, 0., 0.])
    pd.testing.assert_series_equal(obtained, expected)


def test_volatility2_pd():
    days_in_a_year = 252
    factor_monthly = np.sqrt(days_in_a_year / 12)
    rets = pd.Series([9.98272839, 10.00005155, 10.00004624])

    obtained = rolling.volatility(
        rets, window=3, min_periods=3, factor=factor_monthly
    )
    sigma_monthly = 0.01 * np.sqrt(days_in_a_year / 12)
    expected = pd.Series([np.nan, np.nan, sigma_monthly])

    pd.testing.assert_series_equal(obtained, expected)


def test_volatility_default_minimum_allows_partial_windows():
    returns = np.array([0.0, 0.02, 0.04])
    obtained = rolling.volatility(returns, window=3)
    np.testing.assert_allclose(
        obtained, [np.nan, np.sqrt(0.0002), 0.02], equal_nan=True
    )
    assert isinstance(obtained, np.ndarray)


def test_volatility_keyword_array_matches_positional_call():
    returns = np.array([[0.0, 0.02], [0.02, 0.0], [0.04, 0.04]])
    obtained = rolling.volatility(returns=returns, window=2, ddof=0)
    np.testing.assert_allclose(
        obtained, [[np.nan, np.nan], [0.01, 0.01], [0.01, 0.02]],
        equal_nan=True,
    )
    assert isinstance(obtained, np.ndarray)
