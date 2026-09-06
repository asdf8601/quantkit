"""Stats module.

All functions here receive a time series (1-dimension or 2-dimension) and
returns a number for each column:

f(x_t) -> y

WHERE:

    f() : function
    x_t : time series
    y : float
"""

import warnings

import numpy as np

from quantkit import expanding
from quantkit.conventions import BYEAR, ArrayLike
from quantkit.core import cum_returns
from quantkit.decorators import reduce_array_wrap
from quantkit.utils import align, first_valid_index, last_valid_index


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
    .. [1] https://en.wikipedia.org/wiki/Rate_of_return

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
    .. [1] https://en.wikipedia.org/wiki/Volatility_(finance)
            #Mathematical_definition
    .. [2] https://en.wikipedia.org/wiki/
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
    .. [1] Jan Vecer - Maximum Drawdown and Directional Trading, Risk 19(12),
       88-92, 2006.
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
    returns: ArrayLike,
    risk_free: float | ArrayLike,
    factor: float = np.sqrt(BYEAR),
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
    .. [1] https://en.wikipedia.org/wiki/Sharpe_ratio

    """
    ret_excess = returns - risk_free
    e_ret_excess = np.nanmean(ret_excess, axis=-1)
    sigma = np.nanstd(ret_excess)

    return (e_ret_excess / sigma) * factor


def value_at_risk(returns, confidence=0.95):
    r"""Calculate the historical value at risk (VaR).

    The historical VaR at confidence level :math:`c` is the empirical
    :math:`(1 - c)` quantile of the returns, i.e. the return that is only
    undercut with probability :math:`1 - c` in the sample:

    .. math::

        \text{VaR}_{c} = Q_{1 - c}(r_t)

    The quantile is computed with numpy's default linear interpolation
    between order statistics, ignoring NaN.

    The sign of the return is kept, so a loss is a NEGATIVE number, the same
    convention as :func:`drawdown`. Morningstar (and most risk reports)
    quote VaR as a positive loss; negate the output if that convention is
    needed.

    Parameters
    ----------
    returns : array-like
        Asset return series.
    confidence : float, optional
        Confidence level, strictly between 0 and 1. The default 0.95 gives
        the 5th percentile of the returns.

    Returns
    -------
    out : float or array-like
        Value at risk of each column. NaN for a column without valid
        observations.

    Raises
    ------
    ValueError
        If ``confidence`` is not in the open interval (0, 1).

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Value_at_risk
    .. [2] Jorion, P. (2006). Value at Risk: The New Benchmark for Managing
            Financial Risk. McGraw-Hill.

    Examples
    --------
    >>> returns = np.array([-0.05, -0.02, 0.0, 0.01, 0.03])
    >>> value_at_risk(returns, confidence=0.75)
    -0.02

    >>> returns = pd.DataFrame({"a": returns, "b": returns + 0.1})
    >>> value_at_risk(returns, confidence=0.75)
    a   -0.02
    b    0.08
    dtype: float64
    """
    if not 0 < confidence < 1:
        raise ValueError(
            f"confidence must be in the open interval (0, 1): {confidence}"
        )

    arr = returns.__array__()
    percentile = (1 - confidence) * 100

    with warnings.catch_warnings():
        # nanpercentile warns on empty and all-NaN columns; NaN is the
        # documented result there, so the warning is noise
        warnings.simplefilter("ignore", RuntimeWarning)
        var = np.nanpercentile(arr, percentile, axis=0)

    out = reduce_array_wrap(returns, var)

    return out


def max_drawup(prices, relative=True):
    r"""Calculate the Maximum Drawup.

    The drawup is the rise of the prices from their running minimum, the
    mirror image of the drawdown:

    .. math::

        U_{t} = S_{t} - \min_{u \in [0, t]}(S_{u})

    "Max drawup" is the largest drawup value observed in the given time
    series for the whole period.

    .. math::

        MU_{T} = \max_{t \in [0, T]} ( S_{t} - \min_{u \in [0, t]}(S_{u}) )

    Parameters
    ----------
    prices : array-like
        Series on which the drawup is to be calculated.
    relative : bool, optional
        Passing True makes the drawup relative (in parts per unit) to the
        running minimum.

    Returns
    -------
    out : float or array-like
        Maximum Drawup value. NaN when there is no valid observation.

    References
    ----------
    .. [1] Jan Vecer - Maximum Drawdown and Directional Trading, Risk 19(12),
       2006. Maximum drawdown and maximum drawup are studied as a pair.
       http://www.stat.columbia.edu/~vecer/maxdrawdown3.pdf

    Examples
    --------
    >>> prices = np.array([4, 2, 3, 6])
    >>> max_drawup(prices, relative=False)
    4.0

    >>> max_drawup(prices)
    2.0
    """
    du = expanding.drawup(prices, relative=relative)
    arr = du.__array__()

    if arr.shape[0] == 0:
        # no observations: nanmax would raise, return NaN instead
        out = np.full(arr.shape[1:], np.nan)
    else:
        out = np.nanmax(arr, axis=0)

    out = reduce_array_wrap(prices, out)
    return out


def annualized_return(returns, periods_per_year=BYEAR):
    r"""Calculate the geometric annualized return.

    Compounds the returns of the whole period and rescales the result to a
    one year basis, so that series of different lengths are comparable:

    .. math::

        R_{\text{annual}} = \left( \prod_{t=1}^{n} (1 + r_t) \right)
            ^{\frac{P}{n}} - 1

    where :math:`P` is ``periods_per_year`` and :math:`n` is the number of
    non-NaN returns of each column. NaN values are dropped from both the
    product and the count.

    Parameters
    ----------
    returns : array-like
        Arithmetic returns series, 1-d or 2-d (columns are reduced
        independently).
    periods_per_year : float, optional
        Number of returns that make up a year, ``BYEAR`` (business days) by
        default. Use 12 for monthly and 52 for weekly returns.

    Returns
    -------
    out : float or 1d-reduced-array
        Annualized return of each column. NaN when a column has no valid
        return.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Rate_of_return#Annualization

    Examples
    --------
    Doubling twice over two years annualizes to +100%.

    >>> annualized_return(np.array([1.0, 1.0]), periods_per_year=1)
    1.0

    Doubling in half a year annualizes to +300%.

    >>> annualized_return(np.array([1.0]), periods_per_year=2)
    3.0
    """
    arr = returns.__array__()
    n = np.sum(~np.isnan(arr), axis=0)
    growth = np.nanprod(1 + arr, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        # periods_per_year / n is inf where n == 0; masked right after
        out = np.where(n == 0, np.nan, growth ** (periods_per_year / n) - 1)

    if out.ndim == 0:
        out = out[()]

    return reduce_array_wrap(returns, out)


def calmar_ratio(returns, periods_per_year=BYEAR):
    r"""Calculate the Calmar ratio.

    Return earned per unit of maximum drawdown suffered: the annualized
    return divided by the absolute maximum drawdown of the price path
    implied by ``returns``, starting from a capital of 1 [1]_.

    .. math::

        \text{Calmar} = \frac{R_{\text{annual}}}{|MD|}

    The price path is ``cum_returns(returns, first_price=1)`` preceded by
    the starting capital itself, so a loss on the very first return counts
    as a drawdown. Morningstar computes the ratio over the trailing 36
    months; here the caller chooses the window by slicing ``returns``
    before calling.

    Parameters
    ----------
    returns : array-like
        Arithmetic returns series, 1-d or 2-d (columns are reduced
        independently).
    periods_per_year : float, optional
        Number of returns that make up a year, passed to
        :func:`annualized_return`. ``BYEAR`` (business days) by default.

    Returns
    -------
    out : float or 1d-reduced-array
        Calmar ratio of each column. NaN when there is no drawdown to
        divide by or no valid return.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Calmar_ratio
    .. [2] Young, T. W. (1991). Calmar Ratio: A Smoother Tool. Futures,
            20(1), 40.

    Examples
    --------
    Halve the capital, then quadruple it: the year ends +100% after a 50%
    drawdown.

    >>> calmar_ratio(np.array([-0.5, 3.0]), periods_per_year=2)
    2.0

    Without a drawdown the ratio is undefined.

    >>> calmar_ratio(np.array([0.1, 0.2]), periods_per_year=2)
    nan
    """
    arr = returns.__array__()
    ann_ret = annualized_return(arr, periods_per_year=periods_per_year)
    ann_ret = np.asarray(ann_ret, dtype=float)

    # start the price path at the initial capital so that a loss on the first
    # return is a drawdown too
    prices = cum_returns(arr, first_price=1)
    start = np.ones((1,) + arr.shape[1:])
    prices = np.concatenate([start, prices], axis=0)
    mdd = max_drawdown(prices, relative=True)
    mdd = np.abs(np.asarray(mdd, dtype=float))

    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(mdd == 0, np.nan, ann_ret / mdd)

    if out.ndim == 0:
        out = out[()]

    return reduce_array_wrap(returns, out)


def _nan_divide(numerator, denominator):
    """Divide returning NaN, never inf, where the denominator is zero."""
    return numerator / np.where(denominator != 0, denominator, np.nan)


def _masked_mean(arr, mask, method):
    """Column-wise mean of ``arr`` over the periods selected by ``mask``.

    Periods outside the mask, NaN included, do not contribute. A column with
    no selected period gives NaN.
    """
    if method not in ("arith", "geo"):
        raise ValueError(f"method must be 'arith' or 'geo', got {method!r}")

    selected = np.where(mask, arr, 0)
    if method == "geo":
        # prod(1 + r) ** (1 / n) - 1 == expm1(mean(log1p(r))). log1p(-1) is
        # -inf for a total loss and expm1 maps it back to -1.
        with np.errstate(divide="ignore"):
            selected = np.log1p(selected)

    mean = _nan_divide(selected.sum(axis=0), mask.sum(axis=0))

    if method == "geo":
        mean = np.expm1(mean)

    return mean


def average_gain(returns, method="arith"):
    r"""Calculate the average gain.

    The mean of the returns over the periods with a gain, :math:`r_t > 0`.
    Periods with a zero or negative return and NaN are left out.

    .. math::

        \text{arith} = \frac{1}{n_g} \sum_{r_t > 0} r_t

        \text{geo} = \left( \prod_{r_t > 0} (1 + r_t) \right)^{1 / n_g} - 1

    The geometric mean is Morningstar's definition of Average Gain [1]_.

    Parameters
    ----------
    returns : array-like
        Returns series, 1 or 2 dimensions. Reduces along axis 0.
    method : {"arith", "geo"}, optional
        Arithmetic (default) or geometric mean of the gains.

    Returns
    -------
    out : float or array-like
        Average gain per column. NaN when there is no gain.

    Raises
    ------
    ValueError
        If ``method`` is neither "arith" nor "geo".

    References
    ----------
    .. [1] Morningstar, Custom Calculation Data Points, October 2016.
       https://morningstardirect.morningstar.com/clientcomm/customcalculations.pdf

    Examples
    --------
    >>> returns = np.array([0.10, -0.05, 0.20])
    >>> average_gain(returns)
    0.15

    >>> average_gain(returns, method="geo")
    0.1489125293076057
    """
    arr = returns.__array__()
    out = _masked_mean(arr, arr > 0, method)
    return reduce_array_wrap(returns, out)


def average_loss(returns, method="arith"):
    r"""Calculate the average loss.

    The mean of the returns over the periods with a loss, :math:`r_t < 0`, so
    the result is negative. Periods with a zero or positive return and NaN are
    left out.

    .. math::

        \text{arith} = \frac{1}{n_l} \sum_{r_t < 0} r_t

        \text{geo} = \left( \prod_{r_t < 0} (1 + r_t) \right)^{1 / n_l} - 1

    The geometric mean is Morningstar's definition of Average Loss [1]_.

    Parameters
    ----------
    returns : array-like
        Returns series, 1 or 2 dimensions. Reduces along axis 0.
    method : {"arith", "geo"}, optional
        Arithmetic (default) or geometric mean of the losses.

    Returns
    -------
    out : float or array-like
        Average loss per column, negative. NaN when there is no loss.

    Raises
    ------
    ValueError
        If ``method`` is neither "arith" nor "geo".

    References
    ----------
    .. [1] Morningstar, Custom Calculation Data Points, October 2016.
       https://morningstardirect.morningstar.com/clientcomm/customcalculations.pdf

    Examples
    --------
    >>> returns = np.array([-0.10, 0.05, -0.20])
    >>> average_loss(returns)
    -0.15

    >>> average_loss(returns, method="geo")
    -0.1514718625761430
    """
    arr = returns.__array__()
    out = _masked_mean(arr, arr < 0, method)
    return reduce_array_wrap(returns, out)


def gain_loss_ratio(returns):
    r"""Calculate Morningstar's Gain/Loss Ratio.

    The ratio between the arithmetic average gain and the arithmetic average
    loss, in absolute value, multiplied by the ratio between the number of
    periods with a gain and the number of periods with a loss [1]_. Zero
    returns and NaN are neither gains nor losses.

    .. math::

        GL = \left| \frac{\bar{r}_g}{\bar{r}_l} \right| \frac{n_g}{n_l}
           = \frac{\sum_{r_t > 0} r_t}{\left| \sum_{r_t < 0} r_t \right|}

    Both forms are the same number: the counts cancel out and the ratio is
    the sum of the gains over the absolute sum of the losses.

    Parameters
    ----------
    returns : array-like
        Returns series, 1 or 2 dimensions. Reduces along axis 0.

    Returns
    -------
    out : float or array-like
        Gain/Loss Ratio per column. NaN when there is no loss, 0 when there
        are losses but no gain.

    References
    ----------
    .. [1] Morningstar, Custom Calculation Data Points, October 2016.
       https://morningstardirect.morningstar.com/clientcomm/customcalculations.pdf

    Examples
    --------
    >>> returns = np.array([0.1, 0.2, -0.1])
    >>> gain_loss_ratio(returns)
    3.0
    """
    arr = returns.__array__()
    gains = np.where(arr > 0, arr, 0).sum(axis=0)
    losses = -np.where(arr < 0, arr, 0).sum(axis=0)
    out = _nan_divide(gains, losses)
    return reduce_array_wrap(returns, out)


def up_period_percent(returns):
    r"""Calculate the fraction of periods with a return at or above zero.

    Number of periods whose return is greater than or equal to 0 over the
    number of valid periods [1]_. A zero return counts as an up period and
    NaN is left out of both counts.

    .. math::

        UP = \frac{\#\{t : r_t \geq 0\}}{\#\{t : r_t \text{ is not NaN}\}}

    Despite the name, the output is in parts per unit (0.75, not 75), like
    the rest of the library with ``relative=True``.

    Parameters
    ----------
    returns : array-like
        Returns series, 1 or 2 dimensions. Reduces along axis 0.

    Returns
    -------
    out : float or array-like
        Fraction in [0, 1] per column. NaN when there is no valid period. It
        adds up to 1 with :func:`down_period_percent`.

    References
    ----------
    .. [1] Morningstar, Custom Calculation Data Points, October 2016.
       https://morningstardirect.morningstar.com/clientcomm/customcalculations.pdf

    Examples
    --------
    >>> returns = np.array([0.1, 0.0, -0.1, 0.2])
    >>> up_period_percent(returns)
    0.75
    """
    arr = returns.__array__()
    out = _nan_divide((arr >= 0).sum(axis=0), (~np.isnan(arr)).sum(axis=0))
    return reduce_array_wrap(returns, out)


def down_period_percent(returns):
    r"""Calculate the fraction of periods with a return below zero.

    Number of periods whose return is less than 0 over the number of valid
    periods [1]_. NaN is left out of both counts.

    .. math::

        DOWN = \frac{\#\{t : r_t < 0\}}{\#\{t : r_t \text{ is not NaN}\}}

    Despite the name, the output is in parts per unit (0.25, not 25), like
    the rest of the library with ``relative=True``.

    Parameters
    ----------
    returns : array-like
        Returns series, 1 or 2 dimensions. Reduces along axis 0.

    Returns
    -------
    out : float or array-like
        Fraction in [0, 1] per column. NaN when there is no valid period. It
        adds up to 1 with :func:`up_period_percent`.

    References
    ----------
    .. [1] Morningstar, Custom Calculation Data Points, October 2016.
       https://morningstardirect.morningstar.com/clientcomm/customcalculations.pdf

    Examples
    --------
    >>> returns = np.array([0.1, 0.0, -0.1, 0.2])
    >>> down_period_percent(returns)
    0.25
    """
    arr = returns.__array__()
    out = _nan_divide((arr < 0).sum(axis=0), (~np.isnan(arr)).sum(axis=0))
    return reduce_array_wrap(returns, out)


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
    after_valley = valley + 1
    recovered = np.flatnonzero(arr[after_valley:] >= arr[peak])
    recovery = int(after_valley + recovered[0]) if recovered.size else -1
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

    edges = range(0, n_valid + periods_per_year, periods_per_year)
    blocks = [valid[start:stop] for start, stop in zip(edges, edges[1:])]
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


def _pairwise_complete(col, bench):
    """Keep the rows where both ``col`` and ``bench`` are not NaN."""
    mask = ~(np.isnan(col) | np.isnan(bench))
    return col[mask], bench[mask]


def _is_degenerate(arr):
    """Tell whether a variance cannot be estimated from ``arr``.

    Either fewer than two observations or all of them identical. The
    identity test replaces ``var == 0`` because the variance of a constant
    array can come out as a tiny positive number when its mean is rounded.
    """
    return arr.size < 2 or bool(np.all(arr == arr[0]))


def _beta(col, bench):
    """Slope of ``col`` on ``bench``: cov(col, bench) / var(bench).

    Written as the ratio of the deviation sums so the ``n - 1`` cancels.
    """
    if _is_degenerate(bench):
        return np.nan
    dev_col = col - col.mean()
    dev_bench = bench - bench.mean()
    return np.dot(dev_col, dev_bench) / np.dot(dev_bench, dev_bench)


def _alpha(col, bench, risk_free, factor):
    """Jensen's alpha: mean(col - rf) - beta * mean(bench - rf)."""
    beta_ = _beta(col, bench)
    if np.isnan(beta_):
        return np.nan
    excess = np.mean(col - risk_free) - beta_ * np.mean(bench - risk_free)
    return excess * factor


def _correlation(col, bench):
    """Pearson correlation, NaN when either side has no dispersion."""
    if _is_degenerate(col) or _is_degenerate(bench):
        return np.nan
    dev_col = col - col.mean()
    dev_bench = bench - bench.mean()
    scale = np.sqrt(np.dot(dev_col, dev_col) * np.dot(dev_bench, dev_bench))
    return np.dot(dev_col, dev_bench) / scale


def _r_squared(col, bench):
    """Square of the Pearson correlation."""
    return _correlation(col, bench) ** 2


def _bull_beta(col, bench):
    """Beta on the rows where the benchmark went up."""
    up = bench > 0
    return _beta(col[up], bench[up])


def _bear_beta(col, bench):
    """Beta on the rows where the benchmark went down."""
    down = bench < 0
    return _beta(col[down], bench[down])


def _reduce_pairwise(returns, benchmark, func):
    """Apply ``func(col, bench)`` to every column on its complete rows.

    ``returns`` and ``benchmark`` are aligned with
    :func:`quantkit.utils.align`; each column then keeps only the rows where
    both it and the benchmark are non-NaN before ``func`` reduces it to a
    number. The result is wrapped like the original ``returns`` object.
    """
    arr_returns, arr_benchmark = align(returns, benchmark)
    ndim = arr_returns.ndim

    if ndim == 1:
        res = func(*_pairwise_complete(arr_returns, arr_benchmark))
    elif ndim == 2:
        res = np.array(
            [
                func(*_pairwise_complete(col, arr_benchmark))
                for col in arr_returns.T
            ]
        )
    else:
        raise NotImplementedError

    return reduce_array_wrap(returns, res)


def beta(returns, benchmark):
    r"""Compute the beta of ``returns`` against ``benchmark``.

    Beta is the slope of the linear regression of the asset returns on the
    benchmark returns

    .. math::

        r_{i,t} = \alpha_i + \beta_i \cdot r_{b,t} + \epsilon_t

    whose ordinary least squares solution is

    .. math::

        \beta_i = \frac{cov(r_i, r_b)}{var(r_b)}

    Both moments use ``ddof=1`` and, for each column, only the rows where
    the column and the benchmark are both non-NaN (pairwise complete).

    Parameters
    ----------
    returns : array-like
        1D or 2D asset returns. Pandas objects are inner-joined with the
        benchmark on their index, see :func:`quantkit.utils.align`.
    benchmark : array-like
        1D benchmark returns, usually a market index.

    Returns
    -------
    beta : float or array-like
        One value per column of ``returns``. NaN when the benchmark has no
        variance or fewer than two complete rows are available.

    Raises
    ------
    ValueError
        If ``benchmark`` is not 1D or the numpy lengths differ.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Beta_(finance)

    Examples
    --------
    >>> benchmark = np.array([0.5, -0.25, 0.75, -0.5])
    >>> beta(2 * benchmark + 0.125, benchmark)
    2.0
    """
    return _reduce_pairwise(returns, benchmark, _beta)


def alpha(returns, benchmark, risk_free=0.0, factor=None):
    r"""Compute Jensen's alpha of ``returns`` against ``benchmark``.

    .. math::

        \alpha_i = \overline{(r_i - r_f)} - \beta_i \, \overline{(r_b - r_f)}

    where :math:`\beta_i` is :func:`beta` estimated on the same pairwise
    complete rows and :math:`r_f` is the risk free rate per period.

    Parameters
    ----------
    returns : array-like
        1D or 2D asset returns. Pandas objects are inner-joined with the
        benchmark on their index, see :func:`quantkit.utils.align`.
    benchmark : array-like
        1D benchmark returns, usually a market index.
    risk_free : float, optional
        Risk free rate per period, in the same basis as the returns.
    factor : float, optional
        Multiplies the result to change its basis, e.g. 12 to annualize a
        monthly alpha as Morningstar does.

    Returns
    -------
    alpha : float or array-like
        One value per column of ``returns``. NaN whenever :func:`beta` is.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Jensen%27s_alpha

    Examples
    --------
    >>> benchmark = np.array([0.5, -0.25, 0.75, -0.5])
    >>> alpha(2 * benchmark + 0.125, benchmark)
    0.125
    """
    if factor is None:
        factor = 1

    def _jensen(col, bench):
        return _alpha(col, bench, risk_free, factor)

    return _reduce_pairwise(returns, benchmark, _jensen)


def correlation(returns, benchmark):
    r"""Compute the Pearson correlation of ``returns`` with ``benchmark``.

    .. math::

        \rho_i = \frac{cov(r_i, r_b)}{\sigma_{r_i} \, \sigma_{r_b}}

    estimated, for each column, on the rows where the column and the
    benchmark are both non-NaN.

    Parameters
    ----------
    returns : array-like
        1D or 2D asset returns. Pandas objects are inner-joined with the
        benchmark on their index, see :func:`quantkit.utils.align`.
    benchmark : array-like
        1D benchmark returns, usually a market index.

    Returns
    -------
    correlation : float or array-like
        One value per column of ``returns``, in ``[-1, 1]``. NaN when either
        side has no variance or fewer than two complete rows are available.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    Examples
    --------
    >>> benchmark = np.array([0.5, -0.25, 0.75, -0.5])
    >>> correlation(-benchmark, benchmark)
    -1.0
    """
    return _reduce_pairwise(returns, benchmark, _correlation)


def r_squared(returns, benchmark):
    r"""Compute the coefficient of determination against ``benchmark``.

    The share of the variance of ``returns`` explained by the linear
    regression on the benchmark, which is the square of the Pearson
    :func:`correlation`:

    .. math::

        R^2_i = \rho_i^2

    Parameters
    ----------
    returns : array-like
        1D or 2D asset returns. Pandas objects are inner-joined with the
        benchmark on their index, see :func:`quantkit.utils.align`.
    benchmark : array-like
        1D benchmark returns, usually a market index.

    Returns
    -------
    r_squared : float or array-like
        One value per column of ``returns``, in ``[0, 1]``. NaN whenever
        :func:`correlation` is.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Coefficient_of_determination

    Examples
    --------
    >>> benchmark = np.array([0.5, -0.25, 0.75, -0.5])
    >>> r_squared(-benchmark, benchmark)
    1.0
    """
    return _reduce_pairwise(returns, benchmark, _r_squared)


def bull_beta(returns, benchmark):
    """Compute the beta over the periods where the benchmark went up.

    Morningstar's Bull Beta: :func:`beta` restricted to the rows where
    ``benchmark > 0``. Rows with a zero benchmark return belong neither to
    the bull nor to the bear sample.

    Parameters
    ----------
    returns : array-like
        1D or 2D asset returns. Pandas objects are inner-joined with the
        benchmark on their index, see :func:`quantkit.utils.align`.
    benchmark : array-like
        1D benchmark returns, usually a market index.

    Returns
    -------
    bull_beta : float or array-like
        One value per column of ``returns``. NaN when fewer than two rows
        have a positive benchmark or those rows have no variance.

    References
    ----------
    .. [1] Morningstar, "Custom Calculation Data Points", Bull Beta.
            https://morningstardirect.morningstar.com/clientcomm/
            CustomCalculationDataPoints.pdf

    Examples
    --------
    >>> benchmark = np.array([0.5, -0.25, 0.75, -0.5])
    >>> returns = np.array([1.0, -0.125, 1.5, -0.25])  # 2b up, 0.5b down
    >>> bull_beta(returns, benchmark)
    2.0
    """
    return _reduce_pairwise(returns, benchmark, _bull_beta)


def bear_beta(returns, benchmark):
    """Compute the beta over the periods where the benchmark went down.

    Morningstar's Bear Beta: :func:`beta` restricted to the rows where
    ``benchmark < 0``. Rows with a zero benchmark return belong neither to
    the bull nor to the bear sample.

    Parameters
    ----------
    returns : array-like
        1D or 2D asset returns. Pandas objects are inner-joined with the
        benchmark on their index, see :func:`quantkit.utils.align`.
    benchmark : array-like
        1D benchmark returns, usually a market index.

    Returns
    -------
    bear_beta : float or array-like
        One value per column of ``returns``. NaN when fewer than two rows
        have a negative benchmark or those rows have no variance.

    References
    ----------
    .. [1] Morningstar, "Custom Calculation Data Points", Bear Beta.
            https://morningstardirect.morningstar.com/clientcomm/
            CustomCalculationDataPoints.pdf

    Examples
    --------
    >>> benchmark = np.array([0.5, -0.25, 0.75, -0.5])
    >>> returns = np.array([1.0, -0.125, 1.5, -0.25])  # 2b up, 0.5b down
    >>> bear_beta(returns, benchmark)
    0.5
    """
    return _reduce_pairwise(returns, benchmark, _bear_beta)


def _treynor(col, bench, risk_free, factor):
    """Treynor ratio of one column: mean(col - rf) / beta(col, bench).

    NaN whenever the beta cannot be used as a denominator: either it is
    itself NaN, or the column has no dispersion and its beta is zero up to
    rounding, in which case the ratio is undefined rather than infinite.
    """
    if _is_degenerate(col):
        return np.nan
    beta_ = _beta(col, bench)
    if np.isnan(beta_) or beta_ == 0:
        return np.nan
    return np.mean(col - risk_free) / beta_ * factor


def treynor_ratio(returns, benchmark, risk_free=0.0, factor=None):
    r"""Compute the Treynor ratio of ``returns`` against ``benchmark``.

    Morningstar's arithmetic Treynor ratio is the mean excess return per
    unit of systematic risk

    .. math::

        T_i = \frac{\overline{(r_i - r_f)}}{\beta_i}

    where :math:`\beta_i` is :func:`beta` estimated on the same pairwise
    complete rows. It is :func:`sharpe_ratio` with the total risk of the
    asset, its standard deviation, replaced by its systematic risk, so it
    rewards a portfolio only for the market exposure it takes and ignores
    the diversifiable part of its volatility.

    Morningstar annualizes the numerator before dividing; here ``factor``
    does that, and since the numerator is linear it scales the whole ratio.

    Parameters
    ----------
    returns : array-like
        1D or 2D asset returns. Pandas objects are inner-joined with the
        benchmark on their index, see :func:`quantkit.utils.align`.
    benchmark : array-like
        1D benchmark returns, usually a market index.
    risk_free : float, optional
        Risk free rate per period, in the same basis as the returns.
    factor : float, optional
        Multiplies the result to change its basis, e.g. 12 to annualize a
        monthly ratio as Morningstar does.

    Returns
    -------
    treynor_ratio : float or array-like
        One value per column of ``returns``. NaN whenever :func:`beta` is,
        and also when the beta is zero, since the ratio is then undefined:
        the result is never infinite.

    Raises
    ------
    ValueError
        If ``benchmark`` is not 1D or the numpy lengths differ.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Treynor_ratio
    .. [2] Morningstar, "Custom Calculation Data Points", Treynor Ratio.
            https://morningstardirect.morningstar.com/clientcomm/
            CustomCalculationDataPoints.pdf

    Examples
    --------
    >>> benchmark = np.array([0.5, -0.25, 0.75, -0.5])
    >>> returns = 2 * benchmark + 0.125  # beta 2, mean 0.375
    >>> treynor_ratio(returns, benchmark)
    0.1875
    """
    if factor is None:
        factor = 1

    def _ratio(col, bench):
        return _treynor(col, bench, risk_free, factor)

    return _reduce_pairwise(returns, benchmark, _ratio)


def _compound(arr):
    """Compound (geometrically linked) return of ``arr``: prod(1 + r) - 1."""
    return np.prod(1 + arr) - 1


def _capture(col, bench, mask):
    """Compound return of ``col`` over that of ``bench`` on ``mask`` rows.

    NaN when the mask selects no row at all, so that the empty product does
    not silently divide 0 by 0, and when the benchmark compounds to exactly
    0 over the selected rows.
    """
    if not mask.any():
        return np.nan

    denominator = _compound(bench[mask])
    if denominator == 0:
        return np.nan

    return _compound(col[mask]) / denominator


def _up_capture(col, bench):
    """Capture ratio over the rows where the benchmark did not fall."""
    return _capture(col, bench, bench >= 0)


def _down_capture(col, bench):
    """Capture ratio over the rows where the benchmark fell."""
    return _capture(col, bench, bench < 0)


def _overall_capture(col, bench):
    """Up capture over down capture, NaN when the latter is zero."""
    down = _down_capture(col, bench)
    if down == 0:
        return np.nan
    return _up_capture(col, bench) / down


def _batting_average(col, bench):
    """Fraction of the rows where ``col`` is at or above ``bench``."""
    if col.size == 0:
        return np.nan
    return (col >= bench).sum() / col.size


def up_capture(returns, benchmark):
    r"""Compute the up capture ratio against ``benchmark``.

    Morningstar's Up Capture Ratio: how much of the benchmark's rise the
    asset captured [1]_. Over the rows where the benchmark did not fall,
    the compound return of the asset divided by the compound return of the
    benchmark:

    .. math::

        UC_i = \frac{\prod_{r_{b,t} \geq 0} (1 + r_{i,t}) - 1}
                    {\prod_{r_{b,t} \geq 0} (1 + r_{b,t}) - 1}

    A period with a zero benchmark return counts as an up period, matching
    :func:`up_period_percent`. The returns are compounded, not added, so
    the ratio of a single up period is the plain ratio of the two returns
    but the ratio of several is not.

    The output is in parts per unit (1.10, not 110), like the rest of the
    library with ``relative=True``; Morningstar quotes it as a percentage.

    Parameters
    ----------
    returns : array-like
        1D or 2D asset returns. Pandas objects are inner-joined with the
        benchmark on their index, see :func:`quantkit.utils.align`.
    benchmark : array-like
        1D benchmark returns, usually a market index.

    Returns
    -------
    up_capture : float or array-like
        One value per column of ``returns``. NaN when no complete row has a
        benchmark at or above zero, or when the benchmark compounds to
        exactly 0 over those rows.

    Raises
    ------
    ValueError
        If ``benchmark`` is not 1D or the numpy lengths differ.

    References
    ----------
    .. [1] Morningstar, "Custom Calculation Data Points", Up Capture Ratio.
            https://morningstardirect.morningstar.com/clientcomm/
            CustomCalculationDataPoints.pdf

    Examples
    --------
    >>> benchmark = np.array([0.5, -0.5])
    >>> returns = np.array([1.0, -0.25])  # 2b up, 0.5b down
    >>> up_capture(returns, benchmark)
    2.0
    """
    return _reduce_pairwise(returns, benchmark, _up_capture)


def down_capture(returns, benchmark):
    r"""Compute the down capture ratio against ``benchmark``.

    Morningstar's Down Capture Ratio: how much of the benchmark's fall the
    asset suffered [1]_. Same as :func:`up_capture` over the rows where the
    benchmark fell:

    .. math::

        DC_i = \frac{\prod_{r_{b,t} < 0} (1 + r_{i,t}) - 1}
                    {\prod_{r_{b,t} < 0} (1 + r_{b,t}) - 1}

    Both compound returns are usually negative, so the ratio is usually
    positive: above 1 means the asset fell more than the benchmark, below 1
    that it fell less, and a negative value that it rose while the benchmark
    fell. A period with a zero benchmark return is an up period and does not
    take part.

    The output is in parts per unit (1.10, not 110), like the rest of the
    library with ``relative=True``; Morningstar quotes it as a percentage.

    Parameters
    ----------
    returns : array-like
        1D or 2D asset returns. Pandas objects are inner-joined with the
        benchmark on their index, see :func:`quantkit.utils.align`.
    benchmark : array-like
        1D benchmark returns, usually a market index.

    Returns
    -------
    down_capture : float or array-like
        One value per column of ``returns``. NaN when no complete row has a
        negative benchmark, or when the benchmark compounds to exactly 0
        over those rows.

    Raises
    ------
    ValueError
        If ``benchmark`` is not 1D or the numpy lengths differ.

    References
    ----------
    .. [1] Morningstar, "Custom Calculation Data Points", Down Capture
            Ratio. https://morningstardirect.morningstar.com/clientcomm/
            CustomCalculationDataPoints.pdf

    Examples
    --------
    >>> benchmark = np.array([0.5, -0.5])
    >>> returns = np.array([1.0, -0.25])  # 2b up, 0.5b down
    >>> down_capture(returns, benchmark)
    0.5
    """
    return _reduce_pairwise(returns, benchmark, _down_capture)


def overall_capture(returns, benchmark):
    r"""Compute the overall capture ratio against ``benchmark``.

    Morningstar's Overall Capture Ratio, the up capture over the down
    capture [1]_:

    .. math::

        OC_i = \frac{UC_i}{DC_i}

    Above 1 the asset takes more of the benchmark's upside than of its
    downside, below 1 the other way round.

    The output is in parts per unit (1.10, not 110), like the rest of the
    library with ``relative=True``; Morningstar quotes it as a percentage.

    Parameters
    ----------
    returns : array-like
        1D or 2D asset returns. Pandas objects are inner-joined with the
        benchmark on their index, see :func:`quantkit.utils.align`.
    benchmark : array-like
        1D benchmark returns, usually a market index.

    Returns
    -------
    overall_capture : float or array-like
        One value per column of ``returns``. NaN whenever
        :func:`up_capture` or :func:`down_capture` is, and when the down
        capture is 0.

    Raises
    ------
    ValueError
        If ``benchmark`` is not 1D or the numpy lengths differ.

    References
    ----------
    .. [1] Morningstar, "Custom Calculation Data Points", Overall Capture
            Ratio. https://morningstardirect.morningstar.com/clientcomm/
            CustomCalculationDataPoints.pdf

    Examples
    --------
    >>> benchmark = np.array([0.5, -0.5])
    >>> returns = np.array([1.0, -0.25])  # up capture 2, down capture 0.5
    >>> overall_capture(returns, benchmark)
    4.0
    """
    return _reduce_pairwise(returns, benchmark, _overall_capture)


def batting_average(returns, benchmark):
    r"""Compute the fraction of periods that beat or match ``benchmark``.

    Morningstar's Batting Average: the number of periods where the asset
    return is greater than or equal to the benchmark return over the number
    of periods compared [1]_. A tie counts as a hit, and only the rows where
    both sides are non-NaN are compared.

    .. math::

        BA_i = \frac{\#\{t : r_{i,t} \geq r_{b,t}\}}{\#\{t\}}

    The size of the win or of the loss does not matter, only its sign. The
    output is in parts per unit (0.75, not 75), like the rest of the library
    with ``relative=True``; Morningstar quotes it as a percentage.

    Parameters
    ----------
    returns : array-like
        1D or 2D asset returns. Pandas objects are inner-joined with the
        benchmark on their index, see :func:`quantkit.utils.align`.
    benchmark : array-like
        1D benchmark returns, usually a market index.

    Returns
    -------
    batting_average : float or array-like
        Fraction in [0, 1], one value per column of ``returns``. NaN when
        there is no complete row to compare.

    Raises
    ------
    ValueError
        If ``benchmark`` is not 1D or the numpy lengths differ.

    References
    ----------
    .. [1] Morningstar, "Custom Calculation Data Points", Batting Average.
            https://morningstardirect.morningstar.com/clientcomm/
            CustomCalculationDataPoints.pdf

    Examples
    --------
    >>> benchmark = np.array([0.10, -0.10, 0.05, 0.00])
    >>> returns = np.array([0.20, -0.20, 0.05, 0.10])  # win, loss, tie, win
    >>> batting_average(returns, benchmark)
    0.75
    """
    return _reduce_pairwise(returns, benchmark, _batting_average)


def sterling_ratio(returns, periods_per_year=BYEAR, excess=0.10):
    r"""Calculate Morningstar's Sterling ratio.

    Return earned per unit of average drawdown, cushioned by an excess risk
    figure of 10% [1]_. It is the annualized return divided by the absolute
    average drawdown of the price path implied by ``returns`` plus
    ``excess``:

    .. math::

        \text{Sterling} = \frac{R_{\text{annual}}}{|AvgDD| + \text{excess}}

    The price path is the one :func:`calmar_ratio` uses,
    ``cum_returns(returns, first_price=1)`` preceded by the starting capital
    itself, so both ratios agree on what a drawdown is and a loss on the
    very first return counts. That leading price is an observation like any
    other, so ``n`` returns give ``n + 1`` prices and
    :func:`average_drawdown` cuts its yearly blocks over those ``n + 1``
    observations.

    The default excess keeps the denominator positive, so the ratio is
    defined even without a drawdown. With ``excess=0`` and a price path that
    never falls the denominator is zero and the result is NaN; over a single
    block, where the average drawdown is the maximum drawdown, that case
    reproduces :func:`calmar_ratio`.

    Parameters
    ----------
    returns : array-like
        Arithmetic returns series, 1-d or 2-d (columns are reduced
        independently).
    periods_per_year : int, optional
        Number of observations that make up a year, passed to
        :func:`annualized_return` and to :func:`average_drawdown`, which
        uses it as its block length. ``BYEAR`` (business days) by default.
    excess : float, optional
        Risk figure added to the absolute average drawdown, 0.10
        (Morningstar's 10%) by default.

    Returns
    -------
    out : float or 1d-reduced-array
        Sterling ratio of each column. NaN when the denominator is zero or
        the column has no valid return.

    References
    ----------
    .. [1] Morningstar, "Custom Calculation Data Points" (October 2016),
            Sterling Ratio: the compounded annual return over the average
            maximum drawdown minus 10%, taken in absolute value as a
            positive risk figure.
            https://morningstardirect.morningstar.com/clientcomm/
            customcalculations.pdf
    .. [2] https://en.wikipedia.org/wiki/Sterling_ratio

    Examples
    --------
    Losing half the capital in a period that is half a year long: the
    annualized return is :math:`0.5^2 - 1 = -0.75` and the price path
    ``[1, 0.5]`` is one full block with a drawdown of -0.5.

    >>> sterling_ratio(np.array([-0.5]), periods_per_year=2, excess=0.0)
    -1.5

    The default excess softens the same denominator to 0.6.

    >>> sterling_ratio(np.array([-0.5]), periods_per_year=2)
    -1.25

    Without a drawdown the excess is the whole denominator: the prices
    ``[1, 1.25, 1.25]`` never fall, so 0.25 / 0.10 = 2.5.

    >>> sterling_ratio(np.array([0.25, 0.0]), periods_per_year=2)
    2.5

    >>> sterling_ratio(np.array([0.25, 0.0]), periods_per_year=2, excess=0.0)
    nan
    """
    arr = returns.__array__()
    ann_ret = annualized_return(arr, periods_per_year=periods_per_year)
    ann_ret = np.asarray(ann_ret, dtype=float)

    # start the price path at the initial capital so that a loss on the first
    # return is a drawdown too, exactly as calmar_ratio does
    prices = cum_returns(arr, first_price=1)
    start = np.ones((1,) + arr.shape[1:])
    prices = np.concatenate([start, prices], axis=0)
    avg_dd = average_drawdown(prices, periods_per_year=periods_per_year)
    avg_dd = np.abs(np.asarray(avg_dd, dtype=float))

    out = _nan_divide(ann_ret, avg_dd + excess)

    if out.ndim == 0:
        out = out[()]

    return reduce_array_wrap(returns, out)


# ---------------------------------------------------------------------------
# active return statistics: tracking error and information ratio


def _active_return(col, bench):
    """Active return ``col - bench`` and its standard deviation, ``ddof=1``.

    A fund that moves exactly with its benchmark has a constant active
    return, yet subtracting two series of rounded decimals leaves a spread
    of a few ulp around that constant, so an exact equality test does not
    recognise it. A deviation below the rounding noise of the inputs is
    reported as exactly zero rather than as dispersion. ``col`` must hold at
    least two observations.
    """
    active = col - bench
    std = np.std(active, ddof=1)
    scale = max(np.abs(col).max(), np.abs(bench).max())
    tol = 4 * np.finfo(float).eps * scale
    return active, 0.0 if std <= tol else std


def _tracking_error(col, bench, factor):
    """Sample standard deviation of the active return, times ``factor``."""
    if col.size < 2:
        return np.nan
    _, std = _active_return(col, bench)
    return std * factor


def _information_ratio(col, bench, factor):
    """Mean active return over its standard deviation, times ``factor``."""
    if col.size < 2:
        return np.nan
    active, std = _active_return(col, bench)
    if std == 0:  # 0 -> NaN, not inf
        return np.nan
    return np.mean(active) / std * factor


def tracking_error(returns, benchmark, factor=None):
    r"""Calculate the tracking error against ``benchmark``.

    Standard deviation of the active return :math:`r_t - b_t`, the amount by
    which the fund deviates from its benchmark [1]_ [2]_:

    .. math::

        TE = \sqrt{\frac{\sum_{t=1}^{N}
                   \left( (r_t - b_t) - \overline{(r - b)} \right)^2}{N - 1}}

    The deviation uses ``ddof=1`` and, for each column, only the rows where
    the column and the benchmark are both non-NaN (pairwise complete). A
    fund that moves exactly with its benchmark, ``r = b + c`` for a constant
    ``c``, has a zero tracking error.

    Parameters
    ----------
    returns : array-like
        1D or 2D asset returns. Pandas objects are inner-joined with the
        benchmark on their index, see :func:`quantkit.utils.align`.
    benchmark : array-like
        1D benchmark returns, usually the index the fund is measured against.
    factor : float, optional
        Multiplies the result. Annualizing a standard deviation requires the
        caller to pass the square root of the number of periods in a year,
        e.g. ``np.sqrt(BYEAR)`` for daily returns.

    Returns
    -------
    tracking_error : float or array-like
        Float for 1d input, one value per column for 2d input. NaN when
        fewer than two complete rows are available.

    Raises
    ------
    ValueError
        If ``benchmark`` is not 1D or the numpy lengths differ.

    References
    ----------
    .. [1] Morningstar, "Custom Calculation Data Points", Tracking Error:
           the standard deviation of (P - B).
           https://morningstardirect.morningstar.com/clientcomm/
           CustomCalculationDataPoints.pdf
    .. [2] https://en.wikipedia.org/wiki/Tracking_error

    Examples
    --------
    >>> benchmark = np.array([0.01, 0.01, 0.02])
    >>> tracking_error(np.array([0.02, 0.00, 0.05]), benchmark)
    0.02

    >>> tracking_error(benchmark + 0.01, benchmark)
    0.0
    """
    if factor is None:
        factor = 1

    def _func(col, bench):
        return _tracking_error(col, bench, factor)

    return _reduce_pairwise(returns, benchmark, _func)


def information_ratio(returns, benchmark, factor=None):
    r"""Calculate the arithmetic information ratio against ``benchmark``.

    Mean active return per unit of :func:`tracking_error` [1]_ [2]_:

    .. math::

        IR = \frac{\overline{(r - b)}}{\sigma_{r - b}}

    Both moments are taken, for each column, over the rows where the column
    and the benchmark are both non-NaN, and the deviation uses ``ddof=1``.
    A fund that moves exactly with its benchmark has a zero tracking error,
    so its ratio is undefined and comes out as NaN: a constant positive
    active return is not an infinitely good result, it is a result the ratio
    cannot rank.

    Parameters
    ----------
    returns : array-like
        1D or 2D asset returns. Pandas objects are inner-joined with the
        benchmark on their index, see :func:`quantkit.utils.align`.
    benchmark : array-like
        1D benchmark returns, usually the index the fund is measured against.
    factor : float, optional
        Multiplies the result. Annualizing the ratio requires the caller to
        pass the square root of the number of periods in a year, e.g.
        ``np.sqrt(BYEAR)`` for daily returns, since annualizing a standard
        deviation is what the square root does.

    Returns
    -------
    information_ratio : float or array-like
        Float for 1d input, one value per column for 2d input. NaN when the
        tracking error is zero (zero denominator) or fewer than two complete
        rows are available.

    Raises
    ------
    ValueError
        If ``benchmark`` is not 1D or the numpy lengths differ.

    References
    ----------
    .. [1] Morningstar, "Custom Calculation Data Points", Information Ratio
           (arithmetic).
           https://morningstardirect.morningstar.com/clientcomm/
           CustomCalculationDataPoints.pdf
    .. [2] https://en.wikipedia.org/wiki/Information_ratio

    Examples
    --------
    >>> benchmark = np.array([0.01, 0.01, 0.02])
    >>> information_ratio(np.array([0.02, 0.00, 0.05]), benchmark)
    0.5

    >>> information_ratio(benchmark + 0.01, benchmark)
    nan
    """
    if factor is None:
        factor = 1

    def _func(col, bench):
        return _information_ratio(col, bench, factor)

    return _reduce_pairwise(returns, benchmark, _func)
