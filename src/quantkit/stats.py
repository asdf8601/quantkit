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
    .. [1]: https://en.wikipedia.org/wiki/Beta_(finance)

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
    .. [1]: https://en.wikipedia.org/wiki/Jensen%27s_alpha

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
    .. [1]: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

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
    .. [1]: https://en.wikipedia.org/wiki/Coefficient_of_determination

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
    .. [1]: Morningstar, "Custom Calculation Data Points", Bull Beta.
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
    .. [1]: Morningstar, "Custom Calculation Data Points", Bear Beta.
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
