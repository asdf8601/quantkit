"""Tests for portfolio exposure measures."""

import numpy as np
import pandas as pd
import pytest

from quantkit.portfolio import exposure


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([100.0, -40.0], (100.0, 40.0, 140.0, 60.0)),
        ([100.0, 20.0], (120.0, 0.0, 120.0, 120.0)),
        ([-100.0, -20.0], (0.0, 120.0, 120.0, -120.0)),
        ([10.0, -10.0], (10.0, 10.0, 20.0, 0.0)),
    ],
)
def test_exposure_identities_for_vectors(values, expected):
    """Long, short, gross, and net use their signed value conventions."""
    result = tuple(
        function(np.asarray(values))
        for function in (
            exposure.long_value,
            exposure.short_value,
            exposure.gross_exposure,
            exposure.net_exposure,
        )
    )
    np.testing.assert_allclose(result, expected)


def test_exposure_preserves_dataframe_time_axis():
    """DataFrame reductions return a time-indexed Series."""
    values = pd.DataFrame(
        [[100.0, -40.0], [10.0, -10.0]],
        index=pd.Index(["t1", "t2"], name="time"),
        columns=["long", "short"],
    )

    result = exposure.gross_exposure(values)

    pd.testing.assert_series_equal(
        result, pd.Series([140.0, 20.0], index=values.index)
    )


def test_gross_leverage_aligns_nav_and_rejects_bad_vector_shapes():
    """Leverage aligns time-indexed NAV and validates its denominator shape."""
    values = pd.DataFrame(
        [[100.0, -40.0], [10.0, -10.0]],
        index=["t1", "t2"],
        columns=["long", "short"],
    )
    nav = pd.Series([50.0, 100.0], index=["t2", "t1"])

    pd.testing.assert_series_equal(
        exposure.gross_leverage(values, nav),
        pd.Series([1.4, 0.4], index=values.index),
    )
    with pytest.raises(ValueError, match="time axis"):
        exposure.gross_leverage(np.ones((2, 2)), np.ones(3))


def test_gross_leverage_returns_nan_for_nonpositive_nav():
    """Nonpositive NAV cannot normalize monetary exposure."""
    result = exposure.gross_leverage(np.array([[4.0], [4.0], [4.0]]), [2.0, 0.0, -1.0])

    np.testing.assert_allclose(result, [2.0, np.nan, np.nan], equal_nan=True)


def test_exposure_scales_and_propagates_missing_values():
    """Exposure scales with position values and preserves missing observations."""
    values = np.array([[100.0, -40.0], [np.nan, 2.0]])

    np.testing.assert_allclose(
        exposure.gross_exposure(values * 3), [420.0, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(
        exposure.net_exposure(values), [60.0, np.nan], equal_nan=True
    )
