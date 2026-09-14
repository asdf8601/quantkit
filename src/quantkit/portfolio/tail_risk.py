"""Historical tail-risk measures for portfolio time series."""

from numbers import Real

import numpy as np

from quantkit.decorators import reduce_array_wrap
from quantkit.portfolio._utils import asset_array


def _confidence_level(confidence):
    """Validate and normalize a confidence level."""
    if isinstance(confidence, (bool, np.bool_)) or not isinstance(
        confidence, Real
    ):
        raise TypeError("confidence must be a finite real scalar")
    confidence = float(confidence)
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be in the open interval (0, 1)")
    return confidence


def _empirical_upper_tail_mean(values, confidence):
    """Average the upper tail of equally probable observations."""
    sample_size = values.shape[0]
    tail_mass = (1 - confidence) * sample_size
    whole_observations = int(np.floor(tail_mass))
    boundary_weight = tail_mass - whole_observations

    ordered = np.sort(values, axis=0)[::-1]
    total = np.sum(ordered[:whole_observations], axis=0)
    if boundary_weight:
        total = total + boundary_weight * ordered[whole_observations]

    result = total / tail_mass
    return np.where(np.isnan(values).any(axis=0), np.nan, result)


def expected_shortfall(returns, confidence=0.95):
    r"""Calculate historical expected shortfall (ES, or CVaR).

    Returns are converted to signed losses as :math:`L_t=-r_t`.  ES is the
    average over the worst probability mass :math:`1-c` of the empirical
    loss distribution.  Each sample has probability :math:`1/n`; when the
    tail boundary cuts through a sample, the required fraction of that
    boundary observation is included.  This exact tail integration differs
    from averaging observations selected by an interpolated quantile.

    Parameters
    ----------
    returns : numpy.ndarray or pandas.Series or pandas.DataFrame
        A nonempty return history, shaped as time or time by portfolio.
    confidence : float, default 0.95
        Finite confidence level strictly between 0 and 1.

    Returns
    -------
    float or numpy.ndarray or pandas.Series
        Signed expected loss for each portfolio.  A loss is positive, while
        an all-gain tail can be negative.  Any missing observation makes its
        portfolio result NaN.

    Raises
    ------
    TypeError
        If an input is not real numeric or ``confidence`` is not a real
        scalar.
    ValueError
        If an input axis is empty, values include infinity, labels are
        duplicated, or ``confidence`` is outside the open interval (0, 1).

    References
    ----------
    .. [1] Riskfolio-Lib, Risk Functions.
       https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html
    .. [2] Rockafellar, R. T. and Uryasev, S. (2000). Optimization of
       Conditional Value-at-Risk. Journal of Risk, 2(3), 21-41.
    """
    level = _confidence_level(confidence)
    values = asset_array(returns, "returns")
    result = _empirical_upper_tail_mean(-values, level)
    if values.ndim == 1:
        result = float(result)
    return reduce_array_wrap(returns, result)


def conditional_drawdown_at_risk(wealth, confidence=0.95):
    r"""Calculate historical conditional drawdown at risk (CDaR).

    The input is a nonnegative, flow-adjusted compounded wealth history that
    includes its initial baseline.  At each observed time, drawdown is
    :math:`1-W_t/\max_{u\leq t}W_u`.  CDaR integrates the worst probability
    mass :math:`1-c` of these equally weighted observed drawdowns, including
    the initial zero drawdown and any fractional boundary observation.

    This function computes its drawdown path directly.  The existing
    drawdown statistics use different missing-value and sign conventions and
    therefore are not reused here.

    Parameters
    ----------
    wealth : numpy.ndarray or pandas.Series or pandas.DataFrame
        A nonempty wealth history, shaped as time or time by portfolio.  The
        first value of each portfolio must be positive; later values may be
        zero.
    confidence : float, default 0.95
        Finite confidence level strictly between 0 and 1.

    Returns
    -------
    float or numpy.ndarray or pandas.Series
        Mean tail drawdown for each portfolio.  Any missing observation makes
        its portfolio result NaN.

    Raises
    ------
    TypeError
        If an input is not real numeric or ``confidence`` is not a real
        scalar.
    ValueError
        If an input axis is empty, wealth is negative, an observed initial
        value is not positive, values include infinity, labels are duplicated,
        or ``confidence`` is outside the open interval (0, 1).

    References
    ----------
    .. [1] Riskfolio-Lib, Risk Functions.
       https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html
    """
    level = _confidence_level(confidence)
    values = asset_array(wealth, "wealth")
    initial = values[0]
    if np.any((~np.isnan(initial)) & (initial <= 0)):
        raise ValueError("initial wealth must be positive")
    if np.any(values < 0):
        raise ValueError("wealth must be nonnegative")

    running_peak = np.maximum.accumulate(values, axis=0)
    drawdowns = 1 - values / running_peak
    result = _empirical_upper_tail_mean(drawdowns, level)
    if values.ndim == 1:
        result = float(result)
    return reduce_array_wrap(wealth, result)
