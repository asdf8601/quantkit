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


def _nanmean(arr):
    """Column-wise mean ignoring NaN.

    Same as ``np.nanmean(arr, axis=0)`` but returns NaN silently, instead of
    warning, when a column has no valid observation.
    """
    count = np.sum(~np.isnan(arr), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(arr, axis=0) / count


def downside_deviation(returns, mar=0.0, factor=None):
    r"""Calculate the downside deviation of the returns.

    Root mean square of the shortfalls below the minimum acceptable return
    :math:`\text{MAR}`:

    .. math::

        \sigma_d = \sqrt{\frac{1}{N}
                   \sum_{t=1}^{N} \min(r_t - \text{MAR}, 0)^2}

    The mean is taken over all :math:`N` valid observations, not only over
    those below the MAR. This is the Sortino / Morningstar convention [1]_
    [2]_ and makes the statistic the denominator of :func:`sortino_ratio`.

    Parameters
    ----------
    returns : array-like
        Returns series. Reductions are column-wise.
    mar : float, optional
        Minimum acceptable return, in the same units as ``returns``.
    factor : float, optional
        Multiplies the result. To annualize a deviation of returns of a
        shorter period pass the square root of the number of periods in a
        year, e.g. ``np.sqrt(BYEAR)`` for daily returns.

    Returns
    -------
    out : float or array-like
        Float for 1d input, one value per column for 2d input. NaN when
        there is no valid observation.

    Examples
    --------
    >>> downside_deviation(np.array([0.1, -0.1, 0.3, -0.2]))
    0.1118033988749895

    References
    ----------
    .. [1] Sortino, F. A., & Price, L. N. (1994). Performance measurement in
           a downside risk framework. The Journal of Investing, 3(3), 59-64.
    .. [2] https://en.wikipedia.org/wiki/Downside_risk
    """
    if factor is None:
        factor = 1

    diff = returns.__array__() - mar
    dev = np.sqrt(_nanmean(np.minimum(diff, 0) ** 2)) * factor

    return reduce_array_wrap(returns, dev)


def upside_deviation(returns, mar=0.0, factor=None):
    r"""Calculate the upside deviation of the returns.

    Root mean square of the excess returns above the minimum acceptable
    return :math:`\text{MAR}`, the mirror image of :func:`downside_deviation`:

    .. math::

        \sigma_u = \sqrt{\frac{1}{N}
                   \sum_{t=1}^{N} \max(r_t - \text{MAR}, 0)^2}

    The mean is taken over all :math:`N` valid observations, not only over
    those above the MAR.

    Parameters
    ----------
    returns : array-like
        Returns series. Reductions are column-wise.
    mar : float, optional
        Minimum acceptable return, in the same units as ``returns``.
    factor : float, optional
        Multiplies the result. To annualize a deviation of returns of a
        shorter period pass the square root of the number of periods in a
        year, e.g. ``np.sqrt(BYEAR)`` for daily returns.

    Returns
    -------
    out : float or array-like
        Float for 1d input, one value per column for 2d input. NaN when
        there is no valid observation.

    Examples
    --------
    >>> upside_deviation(np.array([0.1, -0.1, 0.3, -0.2]))
    0.15811388300841897

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Downside_risk
    """
    if factor is None:
        factor = 1

    diff = returns.__array__() - mar
    dev = np.sqrt(_nanmean(np.maximum(diff, 0) ** 2)) * factor

    return reduce_array_wrap(returns, dev)


def kappa(returns, mar=0.0, order=2, factor=None):
    r"""Calculate the Kappa ratio of Kaplan and Knowles.

    Generalized downside risk-adjusted performance measure: the mean excess
    return over the minimum acceptable return :math:`\text{MAR}` divided by
    the ``order``-th root of the lower partial moment of that order [1]_:

    .. math::

        \kappa_n = \frac{\bar{r} - \text{MAR}}{\sqrt[n]{\text{LPM}_n}},
        \quad
        \text{LPM}_n = \frac{1}{N} \sum_{t=1}^{N} \max(\text{MAR} - r_t, 0)^n

    ``order=1`` is the Omega ratio minus one (see :func:`omega_ratio`) and
    ``order=2`` is the Sortino ratio (see :func:`sortino_ratio`).

    Parameters
    ----------
    returns : array-like
        Returns series. Reductions are column-wise.
    mar : float, optional
        Minimum acceptable return, in the same units as ``returns``.
    order : float, optional
        Order :math:`n` of the lower partial moment. Must be positive.
    factor : float, optional
        Multiplies the result. Annualizing the ratio requires the caller to
        pass the matching factor, e.g. ``np.sqrt(BYEAR)`` for the Sortino
        ratio (``order=2``) of daily returns.

    Returns
    -------
    out : float or array-like
        Float for 1d input, one value per column for 2d input. NaN when the
        lower partial moment is zero (no return below the MAR) or when there
        is no valid observation.

    Raises
    ------
    ValueError
        If ``order`` is not positive.

    Examples
    --------
    >>> kappa(np.array([0.1, -0.1, 0.3, -0.2]), order=1)
    0.3333333333333332

    References
    ----------
    .. [1] Kaplan, P. D., & Knowles, J. A. (2004). Kappa: a generalized
           downside risk-adjusted performance measure. Journal of Performance
           Measurement, 8(3), 42-54.
    """
    if order <= 0:
        raise ValueError(f"order must be positive, got {order}")
    if factor is None:
        factor = 1

    diff = returns.__array__() - mar
    lpm = _nanmean(np.maximum(-diff, 0) ** order)
    den = lpm ** (1 / order)
    den = np.where(den == 0, np.nan, den)  # zero denominator -> NaN, not inf
    out = _nanmean(diff) / den * factor

    return reduce_array_wrap(returns, out)


def omega_ratio(returns, mar=0.0):
    r"""Calculate the Omega ratio of Shadwick and Keating.

    Probability-weighted gains over probability-weighted losses relative to
    the minimum acceptable return :math:`\text{MAR}` [1]_. For a sample of
    returns it is the sum of the excess returns above the MAR over the sum
    of the shortfalls below it:

    .. math::

        \Omega = \frac{\sum_{t=1}^{N} \max(r_t - \text{MAR}, 0)}
                      {\sum_{t=1}^{N} \max(\text{MAR} - r_t, 0)}

    It equals ``1 + kappa(returns, mar, order=1)``.

    Parameters
    ----------
    returns : array-like
        Returns series. Reductions are column-wise.
    mar : float, optional
        Minimum acceptable return, in the same units as ``returns``.

    Returns
    -------
    out : float or array-like
        Float for 1d input, one value per column for 2d input. NaN when no
        return is below the MAR (zero denominator) or when there is no valid
        observation.

    Examples
    --------
    >>> omega_ratio(np.array([0.1, -0.1, 0.3, -0.2]))
    1.3333333333333333

    References
    ----------
    .. [1] Keating, C., & Shadwick, W. F. (2002). A universal performance
           measure. Journal of Performance Measurement, 6(3), 59-84.
    """
    diff = returns.__array__() - mar
    gains = np.nansum(np.maximum(diff, 0), axis=0)
    losses = np.nansum(np.maximum(-diff, 0), axis=0)
    losses = np.where(losses == 0, np.nan, losses)  # 0 -> NaN, not inf
    out = gains / losses

    return reduce_array_wrap(returns, out)


def sortino_ratio(returns, mar=0.0, factor=None):
    r"""Calculate the Sortino ratio.

    Risk-adjusted return that, unlike the Sharpe ratio, penalizes only the
    returns falling below the minimum acceptable return :math:`\text{MAR}`
    [1]_ [2]_:

    .. math::

        S = \frac{\bar{r} - \text{MAR}}{\sigma_d}

    where :math:`\sigma_d` is the :func:`downside_deviation`. It is
    :func:`kappa` with ``order=2``.

    Parameters
    ----------
    returns : array-like
        Returns series. Reductions are column-wise.
    mar : float, optional
        Minimum acceptable return (a.k.a. target or required return), in the
        same units as ``returns``.
    factor : float, optional
        Multiplies the result. Annualizing the ratio requires the caller to
        pass the square root of the number of periods in a year, e.g.
        ``np.sqrt(BYEAR)`` for daily returns.

    Returns
    -------
    out : float or array-like
        Float for 1d input, one value per column for 2d input. NaN when no
        return is below the MAR (zero downside deviation) or when there is
        no valid observation.

    Examples
    --------
    >>> rets = np.array([0.17, 0.15, 0.23, -0.05, 0.12, 0.09, 0.13, -0.04])
    >>> sortino_ratio(rets)
    4.417261042993862

    >>> sortino_ratio(np.array([-0.1, -0.1, -0.1, -0.1]))
    -1.0

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Sortino_ratio
    .. [2] http://www.redrockcapital.com/
           Sortino__A__Sharper__Ratio_Red_Rock_Capital.pdf
    """
    return kappa(returns, mar=mar, order=2, factor=factor)
