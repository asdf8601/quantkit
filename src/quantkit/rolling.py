from quantkit.decorators import numpy2pandas_args_wrapper, array_output_wrapper
from quantkit.conventions import BYEAR


@numpy2pandas_args_wrapper(0)
@array_output_wrapper(0)
def volatility(returns, window=BYEAR, min_win=2, ddof=1):
    """Volatility calculation."""
    vol = returns.std(window, min_win=min_win, ddof=ddof)
    return vol
