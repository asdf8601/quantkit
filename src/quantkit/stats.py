"""Stats module.

All functions here receive a time series (1-dimension or 2-dimension) and
returns a number for each column:

f(x_t) -> y

WHERE:

    f() : function
    x_t : time series
    y : float
"""
from quantkit import expanding
from quantkit.utils import first_valid_index, last_valid_index
from quantkit.decorators import reduce_array_wrap
from quantkit.conventions import ArrayLike, BYEAR

import numpy as np


def total_returns(prices, factor=None, relative=True):
    """Calculate arithmetic total return.

    Given a prices series this function returns the [total return][1] (a.k.a
    RoR) as ouput.

    Parameters
    ----------
    prices : array-like
        Prices data.
    factor : float, optional
        The annualization factor is the (1/t) term from the annualized total
        return as show the following equation:
        r = (1+R)^(1/t)-1 = sqrt[t](1+R)-1
    relative : bool, optional
        If True returns the output in part per units.

    Raises
    ------
    NotImplementedError
        When prices has more than 2 dimensions.

    Returns
    -------
    total_return : float

    References
    ----------
    .. [1]: https://en.wikipedia.org/wiki/Rate_of_return

    """
    arr = prices.__array__()
    ndim = arr.ndim

    first_idx = np.nan_to_num(first_valid_index(arr)).astype(int)
    last_idx = np.nan_to_num(last_valid_index(arr)).astype(int)
    replace_with_nan = np.isclose(first_idx, last_idx)

    if ndim == 2:
        col_idx = np.arange(arr.shape[1])
        first_idx = first_idx, col_idx
        last_idx = last_idx, col_idx

    first = arr[first_idx]
    last = arr[last_idx]
    tot_ret = last - first

    if relative:
        # it has sense this parameter here?
        tot_ret /= first

    if factor is not None:
        # r = (1+R)^(1/t)-1 = sqrt[t](1+R)-1
        tot_ret = (tot_ret + 1) ** factor - 1

    # ------------------------------------------------------------------------
    # RETURNS

    if not np.any(replace_with_nan):
        return tot_ret

    if ndim == 1:
        tot_ret = np.nan
    elif ndim == 2:
        tot_ret[replace_with_nan] = np.nan
    else:
        raise NotImplementedError

    return tot_ret


def volatility(returns, factor=None, ddof=1):
    r"""Calculate the volatility of the arithmetic returns.

    Calculates the volatility :math:`\sigma` over a returns series :math:`r_t`
    and change the basis of the math:`\sigma` using the factor :math:`F`.

    .. math:

        \sigma = \sqrt{\frac{\sum{(r_t - \bar{r})^2}}{n - \text{ddof}}} \cdot F

    The factor can be help to change the basis of the :math:`\sigma`, for
    example: converting a `\sigma_{\text{daily}}` in `\sigma_{\text{yearly}}`
    multiplying it by a factor :math:`F`:

    .. math:

        \sigma _{\text{yearly}} = \sigma_{\text{daily}}{ \sqrt{\frac{252}{12}}}

    The factor above is equal to :math:`F = \sqrt{\frac{252}{12}}`. For more
    information check out the [ref][1].

    Parameters
    ----------
    returns : array-like
        Data over Volatility will be calculated.
    factor : float, optional
        Normalization factor of the volatility result. Use this to change the
        basis from daily to montly or yearly.
    ddof : int, optional
        Degree of freedom. This help to use the unbiased estimation of the
        standard diviation, [see][2].

    Returns
    -------
    volatility : array-like

    References
    ----------
    .. [1]: https://en.wikipedia.org/wiki/Volatility_(finance)
            #Mathematical_definition
    .. [2]: https://en.wikipedia.org/wiki/
            Unbiased_estimation_of_standard_deviation

    Examples
    --------
    >>> returns = np.array([1, 2, 3]),
    >>> volatility(returns, factor=None, ddof=1):
    1

    >>> returns = pd.Series([1, 2, 3]),
    >>> volatility(returns, factor=None, ddof=1):
    1
    """
    # TODO: automatic factor calculation assuming an index frequency
    if factor is None:
        factor = 1

    # TODO: Now I'm assuming that the frequency is bday always
    arr = returns.__array__()

    vol = np.nanstd(arr, ddof=ddof, axis=0) * factor
    out = reduce_array_wrap(returns, vol)

    return out


def drawdown(prices, relative=True):
    r"""Compute the drawdown statistic.

    The drawdown of a price process :math:`S` at time :math:`t` is defined as
    the drop of the asset prices from its running maximum up to time :math:`t`

    .. math::

       D_{t} = \max_{u \in [0, t]}(S_{u}) - S_{t}

    For the whole period, the running maximum is the maximum of the period.

    Parameters
    ----------
    prices : array-like
        Series on which the drawdown is to be calculated.
    relative : bool, optional
        Passing True makes the drawdown series relative (in parts per unit).

    Returns
    -------
    out : float or array-like
        Drawdown value.

    References
    ----------
    .. [1] Jan Vecer - Maximum Drawdown and Directional Trading
       http://www.stat.columbia.edu/~vecer/maxdrawdown3.pdf
    """
    arr = prices.__array__()
    last_idx = last_valid_index(array=arr)
    ndim = arr.ndim

    # if 2 dimensions, indexing has to be carried out in both axis
    if ndim == 2:
        col_idx = np.arange(arr.shape[1])
        last_idx = last_idx, col_idx

    last_element = arr[last_idx]
    if relative:
        out = np.divide(last_element, np.nanmax(prices))
        out -= 1
    else:
        out = last_element - np.nanmax(prices)

    out = reduce_array_wrap(prices, out)
    return out


def max_drawdown(prices, relative=True):
    r"""Calculate the Maximum Drawdown.

    Max drawdown is the maximum potential loss of the trading system. Drawdown
    is defined as the distance between a given point and the highest point
    before it on the equity curve:

    .. math::

        D_{t} = \max_{u \in [0, t]}(S_{u}) - S_{t}

    "Max drawdown" is the largest drawdown value observed in the given time
    series for the whole period.

    .. math::

        MD_{T} = \max_{t \in [0, T] ( \max_{u \in [0, k]}(S_{u}) - S_{t} )

    Parameters
    ----------
    prices : array-like
        Series on which the drawdown is to be calculated.
    relative : bool, optional
        Passing True makes the drawdown series relative (in parts per unit).

    Returns
    -------
    out : float or array-like
        Maximum Drawdown value.

    Examples
    --------
    >>> prices = np.array([1, 2, 3])]
    >>> max_drawdown(prices)
    0

    >>> prices = np.array([1, 0, 3])]
    >>> max_drawdown(prices)
    -1
    """
    dd = expanding.drawdown(prices, relative=relative)
    out = np.nanmin(dd, axis=0)
    out = reduce_array_wrap(prices, out)
    return out


def sharpe_ratio(
    returns: ArrayLike, risk_free: float, factor: float = np.sqrt(BYEAR)
):
    """Calculate shape ratio.

    It measures the performance of an investment such as a security or
    portfolio compared to a risk-free asset, after adjusting for its risk.

    It is defined as the difference beteween the returns of the investement and
    the risk-free return, divided by the standard deviation of the investment
    returns.

    It represents the additional amount of return that an investor recieves per
    unit of increase in risk[1]_.

    Parameters
    ----------
    returns : array-like
        Asset return series.
    risk_free : number or array-like
        Risk free return.
    factor : float
        Annualization factor which multiplies the raw sharpe ratio.

    Returns
    -------
    out : 1d-reduced-array

    References
    ----------
    .. [1]: https://en.wikipedia.org/wiki/Sharpe_ratio

    """
    ret_excess = returns - risk_free
    e_ret_excess = np.nanmean(ret_excess, axis=-1)
    sigma = np.nanstd(ret_excess)

    return (e_ret_excess / sigma) * factor


# ---------------------------------------------------------------------------
# maximum drawdown details: peak, valley, recovery, durations and Morningstar's
# average drawdown. Everything is computed column by column on 1D arrays.

_PEAK, _VALLEY, _RECOVERY = 0, 1, 2


def _columns(arr):
    """Return the columns of a 1D or 2D array as a list of 1D arrays."""
    if arr.ndim == 1:
        return [arr]
    if arr.ndim == 2:
        return [arr[:, col] for col in range(arr.shape[1])]
    raise NotImplementedError


def _reduce_columns(prices, values):
    """Wrap one value per column of ``prices`` like ``reduce_array_wrap``."""
    if prices.ndim == 1:
        values = values[0]
    return reduce_array_wrap(prices, values)


def _apply_columns(prices, func, *args):
    """Apply ``func(column, *args)`` to every column of ``prices``."""
    arr = prices.__array__()
    values = np.array([func(col, *args) for col in _columns(arr)], float)
    return _reduce_columns(prices, values)


def _max_drawdown_span(arr):
    """Locate the maximum relative drawdown of a 1D array.

    Parameters
    ----------
    arr : numpy.ndarray
        1D prices. Leading NaN are skipped and internal NaN are missing
        observations: the running maximum carries forward through them and
        they are never chosen.

    Returns
    -------
    peak, valley, recovery : int
        Positions of the running maximum in force at the valley (its last
        touch before the valley), of the minimum drawdown (the first one on
        ties) and of the first price at or above the peak after the valley.
        ``recovery`` is -1 when the price never recovers and all three are
        -1 when the price never falls below a previous high.
    """
    dd = expanding.drawdown(arr, relative=True).__array__()
    if np.all(np.isnan(dd)) or np.nanmin(dd) >= 0:
        return -1, -1, -1

    valley = int(np.nanargmin(dd))
    peak = int(np.flatnonzero(dd[: valley + 1] == 0)[-1])
    recovered = np.flatnonzero(arr[valley + 1:] >= arr[peak])
    recovery = int(valley + 1 + recovered[0]) if recovered.size else -1
    return peak, valley, recovery


def _max_drawdown_label(prices, item):
    """Report one item of ``_max_drawdown_span`` per column.

    numpy input keeps the positions as floats so NaN can mean "none"; pandas
    input maps them to the index labels, missing (NaT or NaN) when none.
    """
    arr = prices.__array__()
    positions = np.array(
        [_max_drawdown_span(col)[item] for col in _columns(arr)], int
    )
    missing = positions < 0

    if isinstance(prices, np.ndarray):
        out = np.where(missing, np.nan, positions)
    elif prices.index.size == 0:
        out = np.full(positions.shape, np.nan)
    else:
        out = prices.index[np.maximum(positions, 0)].where(~missing)

    return _reduce_columns(prices, out)


def _max_drawdown_duration(arr):
    """Periods from peak to valley; 0 without drawdown, NaN without data."""
    if np.all(np.isnan(arr)):
        return np.nan
    peak, valley, _ = _max_drawdown_span(arr)
    return valley - peak if peak >= 0 else 0


def _max_drawdown_recovery_duration(arr):
    """Periods from valley to recovery; NaN when there is none."""
    _, valley, recovery = _max_drawdown_span(arr)
    return recovery - valley if recovery >= 0 else np.nan


def _longest_drawdown_duration(arr):
    """Longest run of valid observations strictly below the running max."""
    dd = expanding.drawdown(arr, relative=True).__array__()
    under_water = dd[~np.isnan(dd)] < 0
    if under_water.size == 0:
        return np.nan

    longest = run = 0
    for is_under in under_water:
        run = run + 1 if is_under else 0
        longest = max(longest, run)
    return longest


def _average_drawdown(arr, periods_per_year):
    """Morningstar's average drawdown of a 1D array."""
    valid = arr[~np.isnan(arr)]
    n_valid = valid.size
    if n_valid == 0:
        return np.nan

    starts = range(0, n_valid, periods_per_year)
    blocks = [valid[start:start + periods_per_year] for start in starts]
    mdd_sum = sum(max_drawdown(block) for block in blocks)
    return mdd_sum / (n_valid / periods_per_year)


def max_drawdown_peak(prices):
    """Locate the peak the maximum drawdown fell from.

    The peak is the last time the price stood at the running maximum in
    force at the valley of the maximum drawdown, see :func:`max_drawdown`.

    Parameters
    ----------
    prices : array-like
        Prices data.

    Returns
    -------
    out : float, label or array-like
        For numpy input the position as a float (NaN without drawdown), for
        pandas input the index label (missing without drawdown). Reduced
        column-wise: a Series indexed by the columns for a DataFrame.

    Examples
    --------
    >>> import numpy as np
    >>> max_drawdown_peak(np.array([10, 8, 6, 9, 10, 7]))
    0.0

    >>> max_drawdown_peak(np.array([1, 2, 3]))
    nan

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Drawdown_(economics)
    .. [2] Enrico Schumann - Computing Drawdown Statistics
       http://comisef.wikidot.com/tutorial:drawdowns
    """
    return _max_drawdown_label(prices, _PEAK)


def max_drawdown_valley(prices):
    """Locate the valley (trough) of the maximum drawdown.

    The valley is the first time the relative drawdown reaches its minimum,
    see :func:`max_drawdown`.

    Parameters
    ----------
    prices : array-like
        Prices data.

    Returns
    -------
    out : float, label or array-like
        For numpy input the position as a float (NaN without drawdown), for
        pandas input the index label (missing without drawdown). Reduced
        column-wise: a Series indexed by the columns for a DataFrame.

    Examples
    --------
    >>> import numpy as np
    >>> max_drawdown_valley(np.array([10, 8, 6, 9, 10, 7]))
    2.0

    >>> import pandas as pd
    >>> index = pd.date_range("2020-01-01", periods=6)
    >>> max_drawdown_valley(pd.Series([10, 8, 6, 9, 10, 7], index=index))
    Timestamp('2020-01-03 00:00:00')

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Drawdown_(economics)
    .. [2] Enrico Schumann - Computing Drawdown Statistics
       http://comisef.wikidot.com/tutorial:drawdowns
    """
    return _max_drawdown_label(prices, _VALLEY)


def max_drawdown_recovery(prices):
    """Locate the recovery from the maximum drawdown.

    The recovery is the first time after the valley the price is back at or
    above the peak price, see :func:`max_drawdown_peak`.

    Parameters
    ----------
    prices : array-like
        Prices data.

    Returns
    -------
    out : float, label or array-like
        For numpy input the position as a float, for pandas input the index
        label. NaN (missing) when the price never recovers or there is no
        drawdown. Reduced column-wise: a Series indexed by the columns for a
        DataFrame.

    Examples
    --------
    >>> import numpy as np
    >>> max_drawdown_recovery(np.array([10, 8, 6, 9, 10, 7]))
    4.0

    >>> max_drawdown_recovery(np.array([10, 8, 6, 9]))
    nan

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Drawdown_(economics)
    .. [2] Enrico Schumann - Computing Drawdown Statistics
       http://comisef.wikidot.com/tutorial:drawdowns
    """
    return _max_drawdown_label(prices, _RECOVERY)


def max_drawdown_duration(prices):
    """Calculate the duration of the maximum drawdown.

    Number of periods from the peak to the valley of the maximum drawdown,
    see :func:`max_drawdown_peak` and :func:`max_drawdown_valley`.

    Parameters
    ----------
    prices : array-like
        Prices data.

    Returns
    -------
    out : float or array-like
        Periods from peak to valley, 0 when the price never falls below a
        previous high and NaN without valid observations.

    Examples
    --------
    >>> import numpy as np
    >>> max_drawdown_duration(np.array([10, 8, 6, 9, 10, 7]))
    2.0

    >>> max_drawdown_duration(np.array([1, 2, 3]))
    0.0

    References
    ----------
    .. [1] Bacon, C. Practical Portfolio Performance Measurement and
       Attribution. Wiley. 2004.
    .. [2] https://en.wikipedia.org/wiki/Drawdown_(economics)
    """
    return _apply_columns(prices, _max_drawdown_duration)


def max_drawdown_recovery_duration(prices):
    """Calculate the recovery time of the maximum drawdown.

    Number of periods from the valley of the maximum drawdown to its
    recovery, see :func:`max_drawdown_recovery`.

    Parameters
    ----------
    prices : array-like
        Prices data.

    Returns
    -------
    out : float or array-like
        Periods from valley to recovery. NaN when the price never recovers
        or there is no drawdown.

    Examples
    --------
    >>> import numpy as np
    >>> max_drawdown_recovery_duration(np.array([10, 8, 6, 9, 10, 7]))
    2.0

    >>> max_drawdown_recovery_duration(np.array([10, 8, 6, 9]))
    nan

    References
    ----------
    .. [1] Bacon, C. Practical Portfolio Performance Measurement and
       Attribution. Wiley. 2004.
    .. [2] https://en.wikipedia.org/wiki/Drawdown_(economics)
    """
    return _apply_columns(prices, _max_drawdown_recovery_duration)


def longest_drawdown_duration(prices):
    """Calculate the longest time under water.

    Longest number of consecutive periods with the price strictly below its
    running maximum. It need not be the deepest drawdown. A stretch still
    open at the end of the series counts and internal NaN are ignored: they
    neither break nor extend a stretch.

    Parameters
    ----------
    prices : array-like
        Prices data.

    Returns
    -------
    out : float or array-like
        Longest number of periods under water, 0 when the price never falls
        below a previous high and NaN without valid observations.

    Examples
    --------
    >>> import numpy as np
    >>> longest_drawdown_duration(np.array([10, 9, 9, 9, 10, 2, 10]))
    3.0

    >>> longest_drawdown_duration(np.array([1, 2, 3]))
    0.0

    References
    ----------
    .. [1] Bacon, C. Practical Portfolio Performance Measurement and
       Attribution. Wiley. 2004.
    .. [2] https://en.wikipedia.org/wiki/Drawdown_(economics)
    """
    return _apply_columns(prices, _longest_drawdown_duration)


def average_drawdown(prices, periods_per_year=BYEAR):
    r"""Calculate Morningstar's Average Drawdown.

    The valid observations are split into consecutive blocks of
    ``periods_per_year`` observations, the maximum relative drawdown of each
    block is computed independently (every block starts its own running
    maximum) and the sum is spread over the number of years covered:

    .. math::

        AvgDD = \frac{\sum_{t=1}^{n} MDD_{t}}{N / \text{periods\_per\_year}}

    where :math:`N` is the number of valid observations, so a partial last
    block is weighted by its length [1]_. This is the downside risk measure
    of the Sterling ratio. Drawdowns are negative in this library, so the
    result is negative or zero.

    Parameters
    ----------
    prices : array-like
        Prices data.
    periods_per_year : int, optional
        Observations per block (year).

    Returns
    -------
    out : float or array-like
        Average drawdown, 0 when the price never falls below a previous high
        and NaN without valid observations.

    Examples
    --------
    >>> import numpy as np
    >>> prices = np.array([10, 8, 6, 9, 10, 7])
    >>> average_drawdown(prices, periods_per_year=3)  # (-0.4 - 0.3) / 2
    -0.35

    >>> average_drawdown(prices, periods_per_year=6)  # one block
    -0.4

    References
    ----------
    .. [1] Morningstar - Custom Calculation Data Points, Average Drawdown
       https://morningstardirect.morningstar.com/clientcomm/
       customcalculations.pdf
    """
    return _apply_columns(prices, _average_drawdown, periods_per_year)
