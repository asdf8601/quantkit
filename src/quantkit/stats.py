"""Stats module.

All functions here receive a time series (1-dimension or 2-dimension) and
returns a number for each column:

f(x_t) -> y

WHERE:

    f() : function
    x_t : time series
    y : float
"""

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
