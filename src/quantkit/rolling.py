from quantkit.decorators import numpy2pandas_args_wrapper, array_output_wrapper
from quantkit.conventions import BYEAR
from quantkit.stats import volatility as stat_vol
import numpy as np


@numpy2pandas_args_wrapper(0)
@array_output_wrapper(0)
def volatility(returns, window=BYEAR, min_periods=2, ddof=1, factor=None):
    """Volatility calculation.

    Parameters
    ----------
    returns : array-like
        Returns series.
    window : int or pandas.offset
        Number of elements to take in account on each iteration.
    ddof : int, optional
        Degree of freedom used in the std calculation.
    factor : float, optional
        Annualization factor.

    Returns
    -------
    out : array-like
        Rolling volatility series.

    Examples
    --------
    >>> ret = np.array([0.1, 0.1, 0.1])
    >>> volatility(ret, 2)
    array([np.nan, 0., 0.])

    References
    ----------
    Bacon, C.Practical Portfolio Performance Measurement and Attribution.
    Wiley. 2004. p. 27
    """
    kw = {"ddof": ddof, "factor": factor}
    out = returns.rolling(window, min_periods=min_periods).apply(
        stat_vol, kwargs=kw
    )
    out[: window - 1] = np.nan
    return out
