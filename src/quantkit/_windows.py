"""Shared rolling and expanding reduction engine."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

_LABEL_STATS = {
    "max_drawdown_peak",
    "max_drawdown_recovery",
    "max_drawdown_valley",
}
_BENCHMARK_STATS = {
    "alpha",
    "batting_average",
    "bear_beta",
    "beta",
    "bull_beta",
    "correlation",
    "down_capture",
    "information_ratio",
    "overall_capture",
    "r_squared",
    "tracking_error",
    "treynor_ratio",
    "up_capture",
}
_TAIL_RISK_STATS = {
    "conditional_drawdown_at_risk",
    "expected_shortfall",
}


def _numeric_frame(data):
    """Validate supported input and represent it as a float DataFrame."""
    if not isinstance(data, (np.ndarray, pd.Series, pd.DataFrame)):
        raise TypeError("data must be a numpy array, Series, or DataFrame")

    if isinstance(data, np.ndarray):
        if data.ndim not in (1, 2):
            raise ValueError("data must be one- or two-dimensional")
        if data.dtype.kind not in "iuf":
            raise TypeError("data must contain real numeric values")
        values = data.astype(float, copy=False)
        frame = pd.DataFrame(values if data.ndim == 2 else values[:, None])
    else:
        dtypes = [data.dtype] if data.ndim == 1 else list(data.dtypes)
        if any(
            not is_numeric_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_complex_dtype(dtype)
            for dtype in dtypes
        ):
            raise TypeError("data must contain real numeric values")
        if isinstance(data, pd.Series):
            values = data.to_numpy(dtype=float, na_value=np.nan)
            frame = pd.DataFrame(
                values[:, None], index=data.index, columns=[data.name]
            )
        else:
            values = data.to_numpy(dtype=float, na_value=np.nan)
            frame = pd.DataFrame(
                values, index=data.index, columns=data.columns, copy=False
            )

    if np.isinf(frame.to_numpy()).any():
        raise ValueError("data must not contain infinity")
    return frame


def _minimum_periods(min_periods):
    """Validate and normalize the observation threshold."""
    if isinstance(min_periods, (bool, np.bool_)) or not isinstance(
        min_periods, Integral
    ):
        raise TypeError("min_periods must be a nonnegative integer")
    minimum = int(min_periods)
    if minimum < 0:
        raise ValueError("min_periods must be nonnegative")
    return minimum


def _fixed_offset(window):
    """Return a positive fixed pandas offset."""
    try:
        offset = pd.tseries.frequencies.to_offset(window)
    except (TypeError, ValueError) as error:
        message = "window must be a positive integer or fixed offset"
        raise ValueError(message) from error
    try:
        nanos = offset.nanos
    except ValueError as error:
        raise ValueError("time window must have a fixed duration") from error
    if nanos <= 0:
        raise ValueError("window must be positive")
    return offset


def prepare(data, window=None, min_periods=None):
    """Validate window input and calculate each right-closed window start.

    Parameters
    ----------
    data : numpy.ndarray, pandas.Series, or pandas.DataFrame
        One- or two-dimensional real numeric observations.
    window : positive int or fixed offset, optional
        A row count, a fixed time duration, or ``None`` for expanding windows.
    min_periods : nonnegative int, optional
        Required valid observations. Defaults to one for expanding and time
        windows and to ``window`` for integer windows.

    Returns
    -------
    frame : pandas.DataFrame
        Numeric data with the original row and column labels.
    starts : numpy.ndarray
        Inclusive integer start of the window ending at each row.
    minimum : int
        Validated observation threshold.
    """
    frame = _numeric_frame(data)
    size = len(frame)

    if window is None:
        starts = np.zeros(size, dtype=int)
        default = 1
    elif isinstance(window, (bool, np.bool_)):
        raise TypeError("window must be a positive integer or fixed offset")
    elif isinstance(window, Integral):
        count = int(window)
        if count <= 0:
            raise ValueError("window must be positive")
        starts = np.maximum(0, np.arange(size) - count + 1)
        default = count
    else:
        offset = _fixed_offset(window)
        index = frame.index
        if not isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            raise TypeError(
                "time windows require a DatetimeIndex or TimedeltaIndex"
            )
        if not index.is_unique or not index.is_monotonic_increasing:
            raise ValueError("time-window index must be monotonic and unique")
        starts = np.fromiter(
            (
                index.searchsorted(value - offset, side="right")
                for value in index
            ),
            dtype=int,
            count=size,
        )
        default = 1

    minimum = _minimum_periods(default if min_periods is None else min_periods)
    return frame, starts, minimum


def wrap(data, frame):
    """Restore the input container, dimensions, axes, and Series name."""
    if isinstance(data, pd.DataFrame) and isinstance(frame, pd.DataFrame):
        result = frame.copy(deep=False)
        result.index = data.index
        result.columns = data.columns
        return result
    if isinstance(data, pd.Series) and isinstance(frame, pd.DataFrame):
        result = frame.iloc[:, 0].copy(deep=False)
        result.index = data.index
        result.name = data.name
        return result

    if isinstance(frame, pd.DataFrame):
        values = frame.to_numpy()
    else:
        values = np.asarray(frame)

    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(
            values, index=data.index, columns=data.columns, copy=False
        )
    if isinstance(data, pd.Series):
        return pd.Series(
            values[:, 0], index=data.index, name=data.name, copy=False
        )
    if data.ndim == 1:
        return values[:, 0]
    return values


def _resolve_reducer(name):
    """Resolve a public scalar reducer without creating an import cycle."""
    if not isinstance(name, str) or name.startswith("_"):
        raise ValueError(f"unknown window statistic: {name!r}")
    if name in _TAIL_RISK_STATS:
        from quantkit.portfolio import tail_risk

        module = tail_risk
        reducer = getattr(tail_risk, name)
    else:
        from quantkit import stats

        module = stats
        reducer = getattr(stats, name, None)
    if (
        reducer is None
        or not inspect.isfunction(reducer)
        or reducer.__module__ != module.__name__
    ):
        raise ValueError(f"unknown window statistic: {name!r}")
    return reducer


def _finite_scalar(value, name, *, positive=False):
    """Validate one finite real scalar parameter."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_parameters(reducer, name, parameters, benchmark):
    """Bind and validate reducer parameters before considering any window."""
    if parameters is None:
        parameters = {}
    elif not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping")
    else:
        parameters = dict(parameters)

    if "benchmark" in parameters:
        raise TypeError("pass benchmark with the benchmark argument")
    needs_benchmark = name in _BENCHMARK_STATS
    if needs_benchmark and benchmark is None:
        raise ValueError(f"{name} requires a benchmark")
    if not needs_benchmark and benchmark is not None:
        raise ValueError(f"{name} does not accept a benchmark")

    positional = [np.array([0.0, 1.0])]
    if needs_benchmark:
        positional.append(np.array([0.0, 1.0]))
    reducer_signature = inspect.signature(reducer)
    try:
        reducer_signature.bind(*positional, **parameters)
    except TypeError as error:
        raise TypeError(f"invalid parameters for {name}: {error}") from error

    if "confidence" in reducer_signature.parameters:
        confidence = parameters.get(
            "confidence", reducer_signature.parameters["confidence"].default
        )
        _finite_scalar(confidence, "confidence")
        if not 0 < float(confidence) < 1:
            raise ValueError("confidence must be in the open interval (0, 1)")
    if "ddof" in parameters:
        ddof = parameters["ddof"]
        if isinstance(ddof, (bool, np.bool_)) or not isinstance(
            ddof, Integral
        ):
            raise TypeError("ddof must be a nonnegative integer")
        if ddof < 0:
            raise ValueError("ddof must be nonnegative")
    if "order" in parameters:
        _finite_scalar(parameters["order"], "order", positive=True)
    if "periods_per_year" in parameters:
        _finite_scalar(
            parameters["periods_per_year"], "periods_per_year", positive=True
        )
    for parameter in ("excess", "mar"):
        if parameter in parameters:
            _finite_scalar(parameters[parameter], parameter)
    if "factor" in parameters and parameters["factor"] is not None:
        _finite_scalar(parameters["factor"], "factor")
    if "relative" in parameters and not isinstance(
        parameters["relative"], (bool, np.bool_)
    ):
        raise TypeError("relative must be a boolean")
    if "method" in parameters and parameters["method"] not in ("arith", "geo"):
        raise ValueError("method must be 'arith' or 'geo'")
    return parameters


def _aligned_vector(vector, data, frame, name):
    """Validate a one-dimensional companion and align it to primary rows."""
    if not isinstance(vector, (np.ndarray, pd.Series)):
        raise TypeError(f"{name} must be a one-dimensional array or Series")
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if isinstance(vector, pd.Series):
        if (
            not is_numeric_dtype(vector.dtype)
            or pd.api.types.is_bool_dtype(vector.dtype)
            or pd.api.types.is_complex_dtype(vector.dtype)
        ):
            raise TypeError(f"{name} must contain real numeric values")
        if not vector.index.is_unique:
            raise ValueError(f"{name} index must be unique")
        if isinstance(data, (pd.Series, pd.DataFrame)):
            vector = vector.reindex(frame.index)
        elif len(vector) != len(frame):
            raise ValueError(f"{name} must have the same length as data")
        values = vector.to_numpy(dtype=float, na_value=np.nan)
    else:
        if vector.dtype.kind not in "iuf":
            raise TypeError(f"{name} must contain real numeric values")
        if len(vector) != len(frame):
            raise ValueError(f"{name} must have the same length as data")
        values = vector.astype(float, copy=False)
    if np.isinf(values).any():
        raise ValueError(f"{name} must not contain infinity")
    return pd.Series(values, index=frame.index, name=name)


def _companions(data, frame, name, parameters, benchmark):
    """Prepare benchmark and optional per-period risk-free values."""
    benchmark_series = None
    if name in _BENCHMARK_STATS:
        benchmark_series = _aligned_vector(benchmark, data, frame, "benchmark")

    risk_free_series = None
    if "risk_free" in parameters:
        risk_free = parameters["risk_free"]
        if np.ndim(risk_free) == 0:
            _finite_scalar(risk_free, "risk_free")
        elif name == "sharpe_ratio":
            risk_free_series = _aligned_vector(
                risk_free, data, frame, "risk_free"
            )
        else:
            raise ValueError(f"risk_free must be scalar for {name}")
    return benchmark_series, risk_free_series


def _result_frame(frame, labels, pandas_labels):
    """Allocate an output frame with a dtype suitable for the statistic."""
    if not labels:
        return pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    if isinstance(frame.index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        result = pd.DataFrame(index=frame.index)
        for position, column in enumerate(frame.columns):
            missing = pd.array([pd.NaT] * len(frame), dtype=frame.index.dtype)
            result.insert(position, column, missing, allow_duplicates=True)
        return result
    if not pandas_labels:
        return pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    missing = pd.NaT if isinstance(frame.index, pd.PeriodIndex) else np.nan
    return pd.DataFrame(
        np.full(frame.shape, missing, dtype=object),
        index=frame.index,
        columns=frame.columns,
        dtype=object,
    )


def _validate_cdar_windows(frame, starts):
    """Validate wealth constraints before short windows can be skipped."""
    values = frame.to_numpy()
    if np.any(values < 0):
        raise ValueError("wealth must be nonnegative")
    for start in np.unique(starts):
        initial = values[start]
        if np.any((~np.isnan(initial)) & (initial <= 0)):
            raise ValueError("initial wealth must be positive")


def evaluate(
    data,
    name,
    *,
    window=None,
    min_periods=None,
    parameters=None,
    benchmark=None,
    label=False,
    terminal=False,
):
    """Evaluate a scalar statistic over trailing or expanding windows."""
    if not isinstance(label, (bool, np.bool_)):
        raise TypeError("label must be a boolean")
    if not isinstance(terminal, (bool, np.bool_)):
        raise TypeError("terminal must be a boolean")

    frame, starts, minimum = prepare(data, window, min_periods)
    reducer = _resolve_reducer(name)
    parameters = _validate_parameters(reducer, name, parameters, benchmark)
    benchmark_series, risk_free_series = _companions(
        data, frame, name, parameters, benchmark
    )
    if name == "conditional_drawdown_at_risk":
        _validate_cdar_windows(frame, starts)
    labels = label or name in _LABEL_STATS
    pandas_labels = isinstance(data, (pd.Series, pd.DataFrame))
    result = _result_frame(frame, labels, pandas_labels)

    for column_number in range(frame.shape[1]):
        for end, start in enumerate(starts):
            values = frame.iloc[start : end + 1, column_number]
            valid = values.notna()
            benchmark_slice = None
            if benchmark_series is not None:
                benchmark_slice = benchmark_series.iloc[start : end + 1]
                valid &= benchmark_slice.notna()
            risk_free_slice = None
            if risk_free_series is not None:
                risk_free_slice = risk_free_series.iloc[start : end + 1]
                valid &= risk_free_slice.notna()
            if int(valid.sum()) < max(minimum, 1):
                continue

            call_parameters = dict(parameters)
            if risk_free_slice is not None:
                call_parameters["risk_free"] = risk_free_slice.to_numpy()
            reducer_values = values
            if labels and isinstance(values.index, pd.MultiIndex):
                reducer_values = values.copy(deep=False)
                reducer_values.index = pd.Index(
                    values.index.tolist(), tupleize_cols=False
                )
            arguments = [reducer_values if labels else values.to_numpy()]
            if benchmark_slice is not None:
                arguments.append(benchmark_slice.to_numpy())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                value = reducer(*arguments, **call_parameters)
            result.iat[end, column_number] = value

    if terminal:
        result = result.mask(frame.isna())
    return wrap(data, result)
