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
