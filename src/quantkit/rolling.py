"""Financial statistics over rolling windows.

Inputs and outputs have the same NumPy or pandas container and axes.
Windows end at the current row; no future observations are used.
Missing values count toward window width but not ``min_periods``.
See the window statistics guide for alignment and missing-value rules.
"""

import numpy as np

from quantkit._window_drawdown import drawdown_path
from quantkit._windows import evaluate
from quantkit.conventions import BYEAR

__all__ = [
    "total_returns",
    "volatility",
    "drawdown",
    "max_drawdown",
    "sharpe_ratio",
    "value_at_risk",
    "max_drawup",
    "annualized_return",
    "calmar_ratio",
    "average_gain",
    "average_loss",
    "gain_loss_ratio",
    "up_period_percent",
    "down_period_percent",
    "downside_deviation",
    "upside_deviation",
    "kappa",
    "omega_ratio",
    "sortino_ratio",
    "max_drawdown_peak",
    "max_drawdown_valley",
    "max_drawdown_recovery",
    "max_drawdown_duration",
    "max_drawdown_recovery_duration",
    "longest_drawdown_duration",
    "average_drawdown",
    "beta",
    "alpha",
    "correlation",
    "r_squared",
    "bull_beta",
    "bear_beta",
    "treynor_ratio",
    "up_capture",
    "down_capture",
    "overall_capture",
    "batting_average",
    "sterling_ratio",
    "tracking_error",
    "information_ratio",
    "drawup",
    "expected_shortfall",
    "conditional_drawdown_at_risk",
]


def total_returns(
    prices, window=BYEAR, min_periods=None, factor=None, relative=True
):
    """Total returns over rolling windows.

    Parameters
    ----------
    prices : array-like
        Observed prices or wealth, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    factor : float, optional
        As in :func:`quantkit.stats.total_returns`.
    relative : bool, optional
        As in :func:`quantkit.stats.total_returns`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        prices,
        "total_returns",
        window=window,
        min_periods=min_periods,
        parameters={"factor": factor, "relative": relative},
    )


def volatility(returns, window=BYEAR, min_periods=2, ddof=1, factor=None):
    """Volatility over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    ddof : int, optional
        As in :func:`quantkit.stats.volatility`.
    factor : float, optional
        As in :func:`quantkit.stats.volatility`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "volatility",
        window=window,
        min_periods=min_periods,
        parameters={"factor": factor, "ddof": ddof},
    )


def drawdown(prices, window=BYEAR, min_periods=None, relative=True):
    """Drawdown over rolling windows.

    Parameters
    ----------
    prices : array-like
        Observed prices or wealth, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    relative : bool, optional
        As in :func:`quantkit.stats.drawdown`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.

    Notes
    -----
    A missing current price produces NaN. Zero denominators produce NaN.
    """
    return drawdown_path(
        prices,
        window=window,
        min_periods=min_periods,
        relative=relative,
    )


def max_drawdown(prices, window=BYEAR, min_periods=None, relative=True):
    """Max drawdown over rolling windows.

    Parameters
    ----------
    prices : array-like
        Observed prices or wealth, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    relative : bool, optional
        As in :func:`quantkit.stats.max_drawdown`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.

    Notes
    -----
    Both extrema belong to the current window in chronological order.
    """
    return drawdown_path(
        prices,
        window=window,
        min_periods=min_periods,
        relative=relative,
        maximum=True,
    )


def sharpe_ratio(
    returns, risk_free, window=BYEAR, min_periods=None, factor=np.sqrt(BYEAR)
):
    """Sharpe ratio over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    risk_free : float or array-like
        Risk-free returns per observation; Series align by index.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    factor : float, optional
        As in :func:`quantkit.stats.sharpe_ratio`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "sharpe_ratio",
        window=window,
        min_periods=min_periods,
        parameters={"risk_free": risk_free, "factor": factor},
    )


def value_at_risk(returns, window=BYEAR, min_periods=None, confidence=0.95):
    """Value at risk over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    confidence : float, optional
        As in :func:`quantkit.stats.value_at_risk`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "value_at_risk",
        window=window,
        min_periods=min_periods,
        parameters={"confidence": confidence},
    )


def max_drawup(prices, window=BYEAR, min_periods=None, relative=True):
    """Max drawup over rolling windows.

    Parameters
    ----------
    prices : array-like
        Observed prices or wealth, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    relative : bool, optional
        As in :func:`quantkit.stats.max_drawup`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.

    Notes
    -----
    Both extrema belong to the current window in chronological order.
    """
    return drawdown_path(
        prices,
        window=window,
        min_periods=min_periods,
        relative=relative,
        upward=True,
        maximum=True,
    )


def annualized_return(
    returns, window=BYEAR, min_periods=None, periods_per_year=BYEAR
):
    """Annualized return over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    periods_per_year : int, optional
        As in :func:`quantkit.stats.annualized_return`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "annualized_return",
        window=window,
        min_periods=min_periods,
        parameters={"periods_per_year": periods_per_year},
    )


def calmar_ratio(
    returns, window=BYEAR, min_periods=None, periods_per_year=BYEAR
):
    """Calmar ratio over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    periods_per_year : int, optional
        As in :func:`quantkit.stats.calmar_ratio`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "calmar_ratio",
        window=window,
        min_periods=min_periods,
        parameters={"periods_per_year": periods_per_year},
    )


def average_gain(returns, window=BYEAR, min_periods=None, method="arith"):
    """Average gain over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    method : str, optional
        As in :func:`quantkit.stats.average_gain`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "average_gain",
        window=window,
        min_periods=min_periods,
        parameters={"method": method},
    )


def average_loss(returns, window=BYEAR, min_periods=None, method="arith"):
    """Average loss over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    method : str, optional
        As in :func:`quantkit.stats.average_loss`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "average_loss",
        window=window,
        min_periods=min_periods,
        parameters={"method": method},
    )


def gain_loss_ratio(returns, window=BYEAR, min_periods=None):
    """Gain loss ratio over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "gain_loss_ratio",
        window=window,
        min_periods=min_periods,
    )


def up_period_percent(returns, window=BYEAR, min_periods=None):
    """Up period percent over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "up_period_percent",
        window=window,
        min_periods=min_periods,
    )


def down_period_percent(returns, window=BYEAR, min_periods=None):
    """Down period percent over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "down_period_percent",
        window=window,
        min_periods=min_periods,
    )


def downside_deviation(
    returns, window=BYEAR, min_periods=None, mar=0.0, factor=None
):
    """Downside deviation over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    mar : float, optional
        As in :func:`quantkit.stats.downside_deviation`.
    factor : float, optional
        As in :func:`quantkit.stats.downside_deviation`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "downside_deviation",
        window=window,
        min_periods=min_periods,
        parameters={"mar": mar, "factor": factor},
    )


def upside_deviation(
    returns, window=BYEAR, min_periods=None, mar=0.0, factor=None
):
    """Upside deviation over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    mar : float, optional
        As in :func:`quantkit.stats.upside_deviation`.
    factor : float, optional
        As in :func:`quantkit.stats.upside_deviation`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "upside_deviation",
        window=window,
        min_periods=min_periods,
        parameters={"mar": mar, "factor": factor},
    )


def kappa(
    returns, window=BYEAR, min_periods=None, mar=0.0, order=2, factor=None
):
    """Kappa over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    mar : float, optional
        As in :func:`quantkit.stats.kappa`.
    order : int, optional
        As in :func:`quantkit.stats.kappa`.
    factor : float, optional
        As in :func:`quantkit.stats.kappa`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "kappa",
        window=window,
        min_periods=min_periods,
        parameters={"mar": mar, "order": order, "factor": factor},
    )


def omega_ratio(returns, window=BYEAR, min_periods=None, mar=0.0):
    """Omega ratio over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    mar : float, optional
        As in :func:`quantkit.stats.omega_ratio`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "omega_ratio",
        window=window,
        min_periods=min_periods,
        parameters={"mar": mar},
    )


def sortino_ratio(
    returns, window=BYEAR, min_periods=None, mar=0.0, factor=None
):
    """Sortino ratio over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    mar : float, optional
        As in :func:`quantkit.stats.sortino_ratio`.
    factor : float, optional
        As in :func:`quantkit.stats.sortino_ratio`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "sortino_ratio",
        window=window,
        min_periods=min_periods,
        parameters={"mar": mar, "factor": factor},
    )


def max_drawdown_peak(prices, window=BYEAR, min_periods=None):
    """Max drawdown peak over rolling windows.

    Parameters
    ----------
    prices : array-like
        Observed prices or wealth, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.

    Notes
    -----
    Results are original pandas index labels or absolute NumPy row positions.
    Recovery is only reported once it has been observed.
    """
    return evaluate(
        prices,
        "max_drawdown_peak",
        window=window,
        min_periods=min_periods,
        label=True,
    )


def max_drawdown_valley(prices, window=BYEAR, min_periods=None):
    """Max drawdown valley over rolling windows.

    Parameters
    ----------
    prices : array-like
        Observed prices or wealth, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.

    Notes
    -----
    Results are original pandas index labels or absolute NumPy row positions.
    Recovery is only reported once it has been observed.
    """
    return evaluate(
        prices,
        "max_drawdown_valley",
        window=window,
        min_periods=min_periods,
        label=True,
    )


def max_drawdown_recovery(prices, window=BYEAR, min_periods=None):
    """Max drawdown recovery over rolling windows.

    Parameters
    ----------
    prices : array-like
        Observed prices or wealth, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.

    Notes
    -----
    Results are original pandas index labels or absolute NumPy row positions.
    Recovery is only reported once it has been observed.
    """
    return evaluate(
        prices,
        "max_drawdown_recovery",
        window=window,
        min_periods=min_periods,
        label=True,
    )


def max_drawdown_duration(prices, window=BYEAR, min_periods=None):
    """Max drawdown duration over rolling windows.

    Parameters
    ----------
    prices : array-like
        Observed prices or wealth, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        prices,
        "max_drawdown_duration",
        window=window,
        min_periods=min_periods,
    )


def max_drawdown_recovery_duration(prices, window=BYEAR, min_periods=None):
    """Max drawdown recovery duration over rolling windows.

    Parameters
    ----------
    prices : array-like
        Observed prices or wealth, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        prices,
        "max_drawdown_recovery_duration",
        window=window,
        min_periods=min_periods,
    )


def longest_drawdown_duration(prices, window=BYEAR, min_periods=None):
    """Longest drawdown duration over rolling windows.

    Parameters
    ----------
    prices : array-like
        Observed prices or wealth, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        prices,
        "longest_drawdown_duration",
        window=window,
        min_periods=min_periods,
    )


def average_drawdown(
    prices, window=BYEAR, min_periods=None, periods_per_year=BYEAR
):
    """Average drawdown over rolling windows.

    Parameters
    ----------
    prices : array-like
        Observed prices or wealth, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    periods_per_year : int, optional
        As in :func:`quantkit.stats.average_drawdown`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        prices,
        "average_drawdown",
        window=window,
        min_periods=min_periods,
        parameters={"periods_per_year": periods_per_year},
    )


def beta(returns, benchmark, window=BYEAR, min_periods=None):
    """Beta over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "beta",
        window=window,
        min_periods=min_periods,
        benchmark=benchmark,
    )


def alpha(
    returns,
    benchmark,
    window=BYEAR,
    min_periods=None,
    risk_free=0.0,
    factor=None,
):
    """Alpha over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    risk_free : float, optional
        As in :func:`quantkit.stats.alpha`.
    factor : float, optional
        As in :func:`quantkit.stats.alpha`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "alpha",
        window=window,
        min_periods=min_periods,
        benchmark=benchmark,
        parameters={"risk_free": risk_free, "factor": factor},
    )


def correlation(returns, benchmark, window=BYEAR, min_periods=None):
    """Correlation over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "correlation",
        window=window,
        min_periods=min_periods,
        benchmark=benchmark,
    )


def r_squared(returns, benchmark, window=BYEAR, min_periods=None):
    """R squared over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "r_squared",
        window=window,
        min_periods=min_periods,
        benchmark=benchmark,
    )


def bull_beta(returns, benchmark, window=BYEAR, min_periods=None):
    """Bull beta over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "bull_beta",
        window=window,
        min_periods=min_periods,
        benchmark=benchmark,
    )


def bear_beta(returns, benchmark, window=BYEAR, min_periods=None):
    """Bear beta over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "bear_beta",
        window=window,
        min_periods=min_periods,
        benchmark=benchmark,
    )


def treynor_ratio(
    returns,
    benchmark,
    window=BYEAR,
    min_periods=None,
    risk_free=0.0,
    factor=None,
):
    """Treynor ratio over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    risk_free : float, optional
        As in :func:`quantkit.stats.treynor_ratio`.
    factor : float, optional
        As in :func:`quantkit.stats.treynor_ratio`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "treynor_ratio",
        window=window,
        min_periods=min_periods,
        benchmark=benchmark,
        parameters={"risk_free": risk_free, "factor": factor},
    )


def up_capture(returns, benchmark, window=BYEAR, min_periods=None):
    """Up capture over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "up_capture",
        window=window,
        min_periods=min_periods,
        benchmark=benchmark,
    )


def down_capture(returns, benchmark, window=BYEAR, min_periods=None):
    """Down capture over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "down_capture",
        window=window,
        min_periods=min_periods,
        benchmark=benchmark,
    )


def overall_capture(returns, benchmark, window=BYEAR, min_periods=None):
    """Overall capture over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "overall_capture",
        window=window,
        min_periods=min_periods,
        benchmark=benchmark,
    )


def batting_average(returns, benchmark, window=BYEAR, min_periods=None):
    """Batting average over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "batting_average",
        window=window,
        min_periods=min_periods,
        benchmark=benchmark,
    )


def sterling_ratio(
    returns, window=BYEAR, min_periods=None, periods_per_year=BYEAR, excess=0.1
):
    """Sterling ratio over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    periods_per_year : int, optional
        As in :func:`quantkit.stats.sterling_ratio`.
    excess : float, optional
        As in :func:`quantkit.stats.sterling_ratio`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "sterling_ratio",
        window=window,
        min_periods=min_periods,
        parameters={"periods_per_year": periods_per_year, "excess": excess},
    )


def tracking_error(
    returns, benchmark, window=BYEAR, min_periods=None, factor=None
):
    """Tracking error over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    factor : float, optional
        As in :func:`quantkit.stats.tracking_error`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "tracking_error",
        window=window,
        min_periods=min_periods,
        benchmark=benchmark,
        parameters={"factor": factor},
    )


def information_ratio(
    returns, benchmark, window=BYEAR, min_periods=None, factor=None
):
    """Information ratio over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    factor : float, optional
        As in :func:`quantkit.stats.information_ratio`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "information_ratio",
        window=window,
        min_periods=min_periods,
        benchmark=benchmark,
        parameters={"factor": factor},
    )


def drawup(prices, window=BYEAR, min_periods=None, relative=True):
    """Drawup over rolling windows.

    Parameters
    ----------
    prices : array-like
        Observed prices or wealth, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    relative : bool, optional
        As in :func:`quantkit.expanding.drawup`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.

    Notes
    -----
    A missing current price produces NaN. Zero denominators produce NaN.
    """
    return drawdown_path(
        prices,
        window=window,
        min_periods=min_periods,
        relative=relative,
        upward=True,
    )


def expected_shortfall(
    returns, window=BYEAR, min_periods=None, confidence=0.95
):
    """Calculate expected shortfall over rolling windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    confidence : float, optional
        As in :func:`quantkit.portfolio.tail_risk.expected_shortfall`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.

    Notes
    -----
    NaN within a window propagates, following the scalar tail-risk function.
    """
    return evaluate(
        returns,
        "expected_shortfall",
        window=window,
        min_periods=min_periods,
        parameters={"confidence": confidence},
    )


def conditional_drawdown_at_risk(
    wealth, window=BYEAR, min_periods=None, confidence=0.95
):
    """Conditional drawdown at risk over rolling windows.

    Parameters
    ----------
    wealth : array-like
        Observed prices or wealth, with time along axis 0.
    window : int, str or timedelta, optional
        Positive row count or fixed duration on a sorted, unique time index.
        Duration windows include (timestamp - window, timestamp].
    min_periods : int or None, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
        None uses window size for row windows and 1 for time windows.
    confidence : float, optional
        Tail confidence level, as in the scalar CDaR function.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.

    Notes
    -----
    NaN within a window propagates, following the scalar tail-risk function.
    """
    return evaluate(
        wealth,
        "conditional_drawdown_at_risk",
        window=window,
        min_periods=min_periods,
        parameters={"confidence": confidence},
    )
