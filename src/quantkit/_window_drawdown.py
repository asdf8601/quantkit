"""Shared expanding and rolling drawdown calculations."""

from collections import deque

import numpy as np
import pandas as pd

from quantkit._windows import prepare, wrap


def _relative_change(value, extremum):
    """Return the relative change from ``extremum`` without infinities."""
    if extremum == 0:
        return np.nan
    return (value - extremum) / extremum


def _point_path(values, starts, minimum, relative, upward):
    """Calculate each value's change from the extremum in its window."""
    rows, columns = values.shape
    result = np.full((rows, columns), np.nan)
    required = max(minimum, 1)

    for column in range(columns):
        valid = ~np.isnan(values[:, column])
        counts = np.concatenate(([0], np.cumsum(valid)))
        candidates = deque()

        for row in range(rows):
            start = starts[row]
            while candidates and candidates[0] < start:
                candidates.popleft()

            value = values[row, column]
            if not np.isnan(value):
                if upward:
                    while (
                        candidates and values[candidates[-1], column] >= value
                    ):
                        candidates.pop()
                else:
                    while (
                        candidates and values[candidates[-1], column] <= value
                    ):
                        candidates.pop()
                candidates.append(row)

            count = counts[row + 1] - counts[start]
            if np.isnan(value) or count < required:
                continue

            extremum = values[candidates[0], column]
            if relative:
                result[row, column] = _relative_change(value, extremum)
            else:
                result[row, column] = value - extremum

    return result


def _expanding_maximum(values, minimum, relative, upward):
    """Calculate maximum drawdown or drawup in linear time."""
    rows, columns = values.shape
    result = np.full((rows, columns), np.nan)
    required = max(minimum, 1)

    for column in range(columns):
        extremum = np.nan
        best = np.nan
        count = 0

        for row in range(rows):
            value = values[row, column]
            if not np.isnan(value):
                count += 1
                if np.isnan(extremum):
                    extremum = value
                elif upward:
                    extremum = min(extremum, value)
                else:
                    extremum = max(extremum, value)

                if relative:
                    candidate = _relative_change(value, extremum)
                else:
                    candidate = value - extremum

                if not np.isnan(candidate):
                    if np.isnan(best):
                        best = candidate
                    elif upward:
                        best = max(best, candidate)
                    else:
                        best = min(best, candidate)

            if count >= required:
                result[row, column] = best

    return result


def _rolling_maximum(values, starts, minimum, relative, upward):
    """Calculate the worst complete episode contained in every window."""
    rows, columns = values.shape
    result = np.full((rows, columns), np.nan)
    required = max(minimum, 1)
    valid = ~np.isnan(values)
    counts = np.vstack(
        (np.zeros(columns, dtype=int), np.cumsum(valid, axis=0))
    )

    for row in range(rows):
        start = starts[row]
        interval = values[start : row + 1]
        if upward:
            extrema = np.fmin.accumulate(interval, axis=0)
        else:
            extrema = np.fmax.accumulate(interval, axis=0)

        if relative:
            candidates = np.full_like(interval, np.nan)
            np.divide(
                interval - extrema,
                extrema,
                out=candidates,
                where=extrema != 0,
            )
        else:
            candidates = interval - extrema

        valid_candidates = ~np.isnan(candidates)
        if upward:
            best = np.max(
                np.where(valid_candidates, candidates, -np.inf), axis=0
            )
        else:
            best = np.min(
                np.where(valid_candidates, candidates, np.inf), axis=0
            )
        best[~np.any(valid_candidates, axis=0)] = np.nan

        enough = counts[row + 1] - counts[start] >= required
        result[row] = np.where(enough, best, np.nan)

    return result


def _write_out(out, values, expected_shape):
    """Validate and fill a caller-provided output buffer."""
    if not isinstance(out, (np.ndarray, pd.Series, pd.DataFrame)):
        raise TypeError("out must be a numpy or pandas output buffer")
    array = np.asarray(out)
    if array.shape != expected_shape:
        raise ValueError(
            f"out must have shape {expected_shape}, got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("out must have a floating-point dtype")
    if not array.flags.writeable:
        raise ValueError("out must be writable")
    np.copyto(array, values.reshape(expected_shape))
    return array.reshape(values.shape)


def drawdown_path(
    prices,
    *,
    window=None,
    min_periods=None,
    relative=True,
    upward=False,
    maximum=False,
    out=None,
):
    """Calculate a drawdown or drawup path over expanding or rolling windows.

    Parameters
    ----------
    prices : numpy.ndarray, pandas.Series or pandas.DataFrame
        Prices. Two-dimensional inputs are calculated column by column.
    window : int or pandas offset, optional
        Rolling window. ``None`` uses all observations through the current
        row. Integer windows include that many rows; offset windows include
        labels in ``(label - window, label]``.
    min_periods : int, optional
        Minimum number of valid prices required. The default is one for
        expanding and offset windows, and the window size for integer windows.
    relative : bool, optional
        Divide each change by its preceding extremum. The default is ``True``.
    upward : bool, optional
        Calculate rises from preceding minima instead of falls from maxima.
    maximum : bool, optional
        Return the worst complete episode in each window instead of the change
        at its final price.
    out : array-like, optional
        Floating-point buffer in which to write the result.

    Returns
    -------
    same type as prices
        A shape-preserving drawdown or drawup path. Pandas axes and Series
        names are retained.
    """
    for name, value in (
        ("relative", relative),
        ("upward", upward),
        ("maximum", maximum),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a boolean")

    frame, starts, minimum = prepare(
        prices, window=window, min_periods=min_periods
    )
    values = frame.to_numpy(dtype=float, copy=False)

    if maximum and window is None:
        result = _expanding_maximum(values, minimum, relative, upward)
    elif maximum:
        result = _rolling_maximum(values, starts, minimum, relative, upward)
    else:
        result = _point_path(values, starts, minimum, relative, upward)

    if out is not None:
        result = _write_out(out, result, prices.shape)
        if isinstance(prices, np.ndarray) and isinstance(out, np.ndarray):
            return out

    result_frame = pd.DataFrame(
        result,
        index=frame.index,
        columns=frame.columns,
        copy=False,
    )
    return wrap(prices, result_frame)
