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
from quantkit.utils import first_valid_index, last_valid_index


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
    .. [1]: https://en.wikipedia.org/wiki/Sharpe_ratio

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
    .. [1]: https://en.wikipedia.org/wiki/Value_at_risk
    .. [2]: Jorion, P. (2006). Value at Risk: The New Benchmark for Managing
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
    .. [1]: https://en.wikipedia.org/wiki/Rate_of_return#Annualization

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
    .. [1]: https://en.wikipedia.org/wiki/Calmar_ratio
    .. [2]: Young, T. W. (1991). Calmar Ratio: A Smoother Tool. Futures,
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
