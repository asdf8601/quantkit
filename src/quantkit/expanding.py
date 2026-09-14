"""Financial statistics over expanding windows.

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


def total_returns(prices, min_periods=1, factor=None, relative=True):
    """Total returns over expanding windows.

    Parameters
    ----------
    prices : array-like
        Observed prices or wealth, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
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
        min_periods=min_periods,
        parameters={"factor": factor, "relative": relative},
    )


def volatility(returns, min_periods=1, factor=None, ddof=1):
    """Volatility over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
    factor : float, optional
        As in :func:`quantkit.stats.volatility`.
    ddof : int, optional
        As in :func:`quantkit.stats.volatility`.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "volatility",
        min_periods=min_periods,
        parameters={"factor": factor, "ddof": ddof},
    )


def drawdown(prices, relative=True, out=None, *, min_periods=1):
    """Drawdown over expanding windows.

    Parameters
    ----------
    prices : array-like
        Observed prices or wealth, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
    relative : bool, optional
        As in :func:`quantkit.stats.drawdown`.
    out : numpy.ndarray, optional
        Floating output buffer with the same shape as prices.

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
        min_periods=min_periods,
        relative=relative,
        out=out,
    )


def max_drawdown(prices, min_periods=1, relative=True):
    """Max drawdown over expanding windows.

    Parameters
    ----------
    prices : array-like
        Observed prices or wealth, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
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
        min_periods=min_periods,
        relative=relative,
        maximum=True,
    )


def sharpe_ratio(returns, risk_free, min_periods=1, factor=np.sqrt(BYEAR)):
    """Sharpe ratio over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    risk_free : float or array-like
        Risk-free returns per observation; Series align by index.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
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
        min_periods=min_periods,
        parameters={"risk_free": risk_free, "factor": factor},
    )


def value_at_risk(returns, min_periods=1, confidence=0.95):
    """Value at risk over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
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
        min_periods=min_periods,
        parameters={"confidence": confidence},
    )


def max_drawup(prices, min_periods=1, relative=True):
    """Max drawup over expanding windows.

    Parameters
    ----------
    prices : array-like
        Observed prices or wealth, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
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
        min_periods=min_periods,
        relative=relative,
        upward=True,
        maximum=True,
    )


def annualized_return(returns, min_periods=1, periods_per_year=BYEAR):
    """Annualized return over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
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
        min_periods=min_periods,
        parameters={"periods_per_year": periods_per_year},
    )


def calmar_ratio(returns, min_periods=1, periods_per_year=BYEAR):
    """Calmar ratio over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
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
        min_periods=min_periods,
        parameters={"periods_per_year": periods_per_year},
    )


def average_gain(returns, min_periods=1, method="arith"):
    """Average gain over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
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
        min_periods=min_periods,
        parameters={"method": method},
    )


def average_loss(returns, min_periods=1, method="arith"):
    """Average loss over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
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
        min_periods=min_periods,
        parameters={"method": method},
    )


def gain_loss_ratio(returns, min_periods=1):
    """Gain loss ratio over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "gain_loss_ratio",
        min_periods=min_periods,
    )


def up_period_percent(returns, min_periods=1):
    """Up period percent over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "up_period_percent",
        min_periods=min_periods,
    )


def down_period_percent(returns, min_periods=1):
    """Down period percent over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "down_period_percent",
        min_periods=min_periods,
    )


def downside_deviation(returns, min_periods=1, mar=0.0, factor=None):
    """Downside deviation over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
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
        min_periods=min_periods,
        parameters={"mar": mar, "factor": factor},
    )


def upside_deviation(returns, min_periods=1, mar=0.0, factor=None):
    """Upside deviation over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
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
        min_periods=min_periods,
        parameters={"mar": mar, "factor": factor},
    )


def kappa(returns, min_periods=1, mar=0.0, order=2, factor=None):
    """Kappa over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
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
        min_periods=min_periods,
        parameters={"mar": mar, "order": order, "factor": factor},
    )


def omega_ratio(returns, min_periods=1, mar=0.0):
    """Omega ratio over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
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
        min_periods=min_periods,
        parameters={"mar": mar},
    )


def sortino_ratio(returns, min_periods=1, mar=0.0, factor=None):
    """Sortino ratio over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
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
        min_periods=min_periods,
        parameters={"mar": mar, "factor": factor},
    )


def max_drawdown_peak(prices, min_periods=1):
    """Max drawdown peak over expanding windows.

    Parameters
    ----------
    prices : array-like
        Observed prices or wealth, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).

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
        min_periods=min_periods,
        label=True,
    )


def max_drawdown_valley(prices, min_periods=1):
    """Max drawdown valley over expanding windows.

    Parameters
    ----------
    prices : array-like
        Observed prices or wealth, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).

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
        min_periods=min_periods,
        label=True,
    )


def max_drawdown_recovery(prices, min_periods=1):
    """Max drawdown recovery over expanding windows.

    Parameters
    ----------
    prices : array-like
        Observed prices or wealth, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).

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
        min_periods=min_periods,
        label=True,
    )


def max_drawdown_duration(prices, min_periods=1):
    """Max drawdown duration over expanding windows.

    Parameters
    ----------
    prices : array-like
        Observed prices or wealth, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        prices,
        "max_drawdown_duration",
        min_periods=min_periods,
    )


def max_drawdown_recovery_duration(prices, min_periods=1):
    """Max drawdown recovery duration over expanding windows.

    Parameters
    ----------
    prices : array-like
        Observed prices or wealth, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        prices,
        "max_drawdown_recovery_duration",
        min_periods=min_periods,
    )


def longest_drawdown_duration(prices, min_periods=1):
    """Longest drawdown duration over expanding windows.

    Parameters
    ----------
    prices : array-like
        Observed prices or wealth, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        prices,
        "longest_drawdown_duration",
        min_periods=min_periods,
    )


def average_drawdown(prices, min_periods=1, periods_per_year=BYEAR):
    """Average drawdown over expanding windows.

    Parameters
    ----------
    prices : array-like
        Observed prices or wealth, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
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
        min_periods=min_periods,
        parameters={"periods_per_year": periods_per_year},
    )


def beta(returns, benchmark, min_periods=1):
    """Beta over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "beta",
        min_periods=min_periods,
        benchmark=benchmark,
    )


def alpha(returns, benchmark, min_periods=1, risk_free=0.0, factor=None):
    """Alpha over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
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
        min_periods=min_periods,
        benchmark=benchmark,
        parameters={"risk_free": risk_free, "factor": factor},
    )


def correlation(returns, benchmark, min_periods=1):
    """Correlation over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "correlation",
        min_periods=min_periods,
        benchmark=benchmark,
    )


def r_squared(returns, benchmark, min_periods=1):
    """R squared over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "r_squared",
        min_periods=min_periods,
        benchmark=benchmark,
    )


def bull_beta(returns, benchmark, min_periods=1):
    """Bull beta over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "bull_beta",
        min_periods=min_periods,
        benchmark=benchmark,
    )


def bear_beta(returns, benchmark, min_periods=1):
    """Bear beta over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "bear_beta",
        min_periods=min_periods,
        benchmark=benchmark,
    )


def treynor_ratio(
    returns, benchmark, min_periods=1, risk_free=0.0, factor=None
):
    """Treynor ratio over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
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
        min_periods=min_periods,
        benchmark=benchmark,
        parameters={"risk_free": risk_free, "factor": factor},
    )


def up_capture(returns, benchmark, min_periods=1):
    """Up capture over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "up_capture",
        min_periods=min_periods,
        benchmark=benchmark,
    )


def down_capture(returns, benchmark, min_periods=1):
    """Down capture over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "down_capture",
        min_periods=min_periods,
        benchmark=benchmark,
    )


def overall_capture(returns, benchmark, min_periods=1):
    """Overall capture over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "overall_capture",
        min_periods=min_periods,
        benchmark=benchmark,
    )


def batting_average(returns, benchmark, min_periods=1):
    """Batting average over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Statistic at each row, preserving input type, shape and labels.
    """
    return evaluate(
        returns,
        "batting_average",
        min_periods=min_periods,
        benchmark=benchmark,
    )


def sterling_ratio(returns, min_periods=1, periods_per_year=BYEAR, excess=0.1):
    """Sterling ratio over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
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
        min_periods=min_periods,
        parameters={"periods_per_year": periods_per_year, "excess": excess},
    )


def tracking_error(returns, benchmark, min_periods=1, factor=None):
    """Tracking error over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
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
        min_periods=min_periods,
        benchmark=benchmark,
        parameters={"factor": factor},
    )


def information_ratio(returns, benchmark, min_periods=1, factor=None):
    """Information ratio over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    benchmark : array-like
        One-dimensional benchmark aligned by index for pandas inputs.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
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
        min_periods=min_periods,
        benchmark=benchmark,
        parameters={"factor": factor},
    )


def drawup(prices, relative=True, out=None, *, min_periods=1):
    """Drawup over expanding windows.

    Parameters
    ----------
    prices : array-like
        Observed prices or wealth, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
    relative : bool, optional
        As in :func:`quantkit.expanding.drawup`.
    out : numpy.ndarray, optional
        Floating output buffer with the same shape as prices.

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
        min_periods=min_periods,
        relative=relative,
        upward=True,
        out=out,
    )


def expected_shortfall(returns, min_periods=1, confidence=0.95):
    """Calculate expected shortfall over expanding windows.

    Parameters
    ----------
    returns : array-like
        Observed returns, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
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
        min_periods=min_periods,
        parameters={"confidence": confidence},
    )


def conditional_drawdown_at_risk(wealth, min_periods=1, confidence=0.95):
    """Conditional drawdown at risk over expanding windows.

    Parameters
    ----------
    wealth : array-like
        Observed prices or wealth, with time along axis 0.
    min_periods : int, optional
        Minimum valid observations (jointly valid pairs for benchmark stats).
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
        min_periods=min_periods,
        parameters={"confidence": confidence},
    )
