"""Portfolio profit, flow-adjusted returns, and linked returns."""

import numpy as np
import pandas as pd

from quantkit.portfolio._utils import _real_array


def _interval_values(values, name):
    """Validate a scalar or nonempty one-dimensional interval input."""
    if isinstance(values, pd.DataFrame):
        raise ValueError(f"{name} must be scalar or one-dimensional")
    if not (
        np.isscalar(values) or isinstance(values, (np.ndarray, pd.Series))
    ):
        raise TypeError(f"{name} must be a scalar, numpy array, or Series")
    if isinstance(values, pd.Series) and values.index.has_duplicates:
        raise ValueError(f"{name} has duplicate labels")

    array = _real_array(values, name)
    if array.ndim > 1 or (array.ndim == 1 and array.size == 0):
        raise ValueError(f"{name} must be scalar or one-dimensional")
    return array


def _aligned_values(reference, values, name):
    """Validate values and align matching Series labels to the reference."""
    array = _interval_values(values, name)
    if isinstance(reference, pd.Series) and isinstance(values, pd.Series):
        if (
            len(reference.index) != len(values.index)
            or not reference.index.isin(values.index).all()
        ):
            raise ValueError(f"{name} labels must match beginning_value")
        array = _real_array(values.reindex(reference.index), name)
    return array


def _flow_values(beginning_value, ending_value, external_flow):
    """Validate and align the three interval inputs."""
    beginning = _interval_values(beginning_value, "beginning_value")
    ending = _aligned_values(beginning_value, ending_value, "ending_value")
    flow = _aligned_values(beginning_value, external_flow, "external_flow")

    if beginning.ndim != ending.ndim or beginning.shape != ending.shape:
        raise ValueError("ending_value must match beginning_value shape")
    if flow.ndim != 0 and (flow.ndim != 1 or flow.shape != beginning.shape):
        raise ValueError(
            "external_flow must be scalar or match beginning_value"
        )
    return beginning, ending, flow


def _wrap_intervals(like, values):
    """Wrap interval results according to the first input."""
    result = np.asarray(values, dtype=float)
    if isinstance(like, pd.Series):
        return pd.Series(result, index=like.index, name=like.name, copy=False)
    if result.ndim == 0:
        return float(result)
    return result


def profit_loss(beginning_value, ending_value, external_flow=0.0):
    """Calculate investment profit or loss after removing external flows.

    The monetary profit or loss for each interval is
    ``ending_value - beginning_value - external_flow``. Positive flows are
    contributions and negative flows are withdrawals.

    Parameters
    ----------
    beginning_value : scalar, numpy.ndarray, or pandas.Series
        Portfolio value before the interval and before any beginning flow.
    ending_value : scalar, numpy.ndarray, or pandas.Series
        Portfolio value after the interval and after any ending flow. It must
        match the dimensionality and shape of ``beginning_value``.
    external_flow : scalar, numpy.ndarray, or pandas.Series, default 0.0
        Net contribution for each interval. A scalar broadcasts over an
        interval history; a vector must match ``beginning_value``.

    Returns
    -------
    float, numpy.ndarray, or pandas.Series
        Flow-adjusted monetary profit or loss, preserving the first Series'
        index and name.

    Notes
    -----
    This boundary-flow convention does not infer intraperiod flow timing and
    does not by itself claim compliance with the GIPS standards.

    References
    ----------
    .. [1] CFA Institute, `GIPS Standards Handbook for Firms
       <https://www.gipsstandards.org/standards/gips-standards-for-firms/gips-standards-handbook-for-firms/>`_.
    """
    beginning, ending, flow = _flow_values(
        beginning_value, ending_value, external_flow
    )
    result = ending - beginning - flow
    return _wrap_intervals(beginning_value, result)


def period_return(
    beginning_value,
    ending_value,
    external_flow=0.0,
    *,
    flow_timing="end",
):
    """Calculate a boundary-flow-adjusted return for each interval.

    For an end flow, the return is
    ``(ending_value - beginning_value - external_flow) / beginning_value``.
    For a beginning flow, its denominator is instead
    ``beginning_value + external_flow``.

    Parameters
    ----------
    beginning_value : scalar, numpy.ndarray, or pandas.Series
        Portfolio value before the interval and before any beginning flow.
    ending_value : scalar, numpy.ndarray, or pandas.Series
        Portfolio value after the interval and after any ending flow. It must
        match the dimensionality and shape of ``beginning_value``.
    external_flow : scalar, numpy.ndarray, or pandas.Series, default 0.0
        Net contribution for each interval. Positive values are contributions
        and negative values are withdrawals.
    flow_timing : {"beginning", "end"}, default "end"
        Boundary at which every supplied flow occurs.

    Returns
    -------
    float, numpy.ndarray, or pandas.Series
        Return for each interval. A missing or nonpositive denominator gives
        NaN. The first Series' index and name are preserved.

    Raises
    ------
    ValueError
        If ``flow_timing`` is invalid or input shapes or labels do not match.

    Notes
    -----
    For intraperiod flows, split the history into subperiods and supply a
    valuation at each flow boundary to obtain an exact time-weighted return.
    This explicit convention does not by itself claim GIPS compliance.

    References
    ----------
    .. [1] CFA Institute, `GIPS Standards Handbook for Firms
       <https://www.gipsstandards.org/standards/gips-standards-for-firms/gips-standards-handbook-for-firms/>`_.
    """
    if flow_timing not in ("beginning", "end"):
        raise ValueError("flow_timing must be 'beginning' or 'end'")

    beginning, ending, flow = _flow_values(
        beginning_value, ending_value, external_flow
    )
    profit = ending - beginning - flow
    denominator = beginning + flow if flow_timing == "beginning" else beginning
    result = np.full(beginning.shape, np.nan, dtype=float)
    np.divide(profit, denominator, out=result, where=denominator > 0)
    return _wrap_intervals(beginning_value, result)


def time_weighted_return(
    beginning_value,
    ending_value,
    external_flow=0.0,
    *,
    flow_timing="end",
):
    """Geometrically link flow-adjusted interval returns.

    The time-weighted return is ``product(1 + period_return) - 1`` over the
    supplied chronological interval sequence.

    Parameters
    ----------
    beginning_value : scalar, numpy.ndarray, or pandas.Series
        Portfolio value before each interval and before any beginning flow.
    ending_value : scalar, numpy.ndarray, or pandas.Series
        Portfolio value after each interval and after any ending flow. It must
        match the dimensionality and shape of ``beginning_value``.
    external_flow : scalar, numpy.ndarray, or pandas.Series, default 0.0
        Net contribution for each interval. Positive values are contributions
        and negative values are withdrawals.
    flow_timing : {"beginning", "end"}, default "end"
        Boundary at which every supplied flow occurs.

    Returns
    -------
    float
        Linked return for the complete sequence. Missing interval returns
        propagate to the result.

    Raises
    ------
    ValueError
        If an interval return is below -1, where geometric linking would
        continue after negative wealth, or if validation otherwise fails.

    Notes
    -----
    For intraperiod flows, split the history into subperiods and supply a
    valuation at each flow boundary. A return of exactly -1 is accepted and
    compounds to -1. This explicit convention does not by itself claim GIPS
    compliance.

    References
    ----------
    .. [1] CFA Institute, `GIPS Standards Handbook for Firms
       <https://www.gipsstandards.org/standards/gips-standards-for-firms/gips-standards-handbook-for-firms/>`_.
    """
    returns = period_return(
        beginning_value,
        ending_value,
        external_flow,
        flow_timing=flow_timing,
    )
    values = np.asarray(returns, dtype=float)
    if np.any(values < -1):
        raise ValueError("cannot compound an interval return below -1")
    return float(np.prod(1.0 + values) - 1.0)
