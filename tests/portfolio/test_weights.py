"""Tests for portfolio composition weights and concentration."""

import numpy as np
import pandas as pd
import pytest

from quantkit.portfolio import weights


def test_equal_gross_exposures_have_quarter_concentration_and_four_assets():
    """Four equal absolute values produce H=1/4 and N=4."""
    positions = np.array([10.0, -10.0, 10.0, -10.0])

    np.testing.assert_allclose(
        weights.gross_weights(positions), np.full(4, 0.25)
    )
    assert weights.concentration(positions) == pytest.approx(0.25)
    assert weights.effective_number_of_assets(positions) == pytest.approx(4.0)


def test_single_position_has_one_effective_asset():
    """A sole nonzero position has unit concentration and effective count."""
    positions = np.array([0.0, -25.0, 0.0])

    np.testing.assert_allclose(weights.gross_weights(positions), [0.0, 1.0, 0.0])
    assert weights.concentration(positions) == pytest.approx(1.0)
    assert weights.effective_number_of_assets(positions) == pytest.approx(1.0)


def test_short_positions_keep_sign_for_nav_and_use_magnitudes_for_gross():
    """NAV weights are signed while gross weights and H use magnitudes."""
    positions = pd.Series([100.0, -40.0], index=["long", "short"], name="value")

    pd.testing.assert_series_equal(
        weights.nav_weights(positions, 100.0),
        pd.Series([1.0, -0.4], index=positions.index, name=positions.name),
    )
    pd.testing.assert_series_equal(
        weights.gross_weights(positions),
        pd.Series([100 / 140, 40 / 140], index=positions.index, name=positions.name),
    )
    assert weights.concentration(positions) == pytest.approx(11600 / 19600)


def test_matrix_weights_use_each_row_nav_and_preserve_dataframe_metadata():
    """Each matrix row is normalized by its corresponding NAV."""
    positions = pd.DataFrame(
        [[100.0, -40.0], [20.0, -20.0]],
        index=pd.Index(["t1", "t2"], name="time"),
        columns=pd.Index(["long", "short"], name="asset"),
    )

    result = weights.nav_weights(positions, np.array([100.0, 50.0]))

    expected = pd.DataFrame(
        [[1.0, -0.4], [0.4, -0.4]],
        index=positions.index,
        columns=positions.columns,
    )
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(
        weights.gross_weights(positions),
        pd.DataFrame(
            [[100 / 140, 40 / 140], [0.5, 0.5]],
            index=positions.index,
            columns=positions.columns,
        ),
    )
    pd.testing.assert_series_equal(
        weights.concentration(positions),
        pd.Series([11600 / 19600, 0.5], index=positions.index),
    )


def test_nav_series_aligns_to_permuted_dataframe_time_index():
    """A pandas NAV series is reordered to the reference time index."""
    positions = pd.DataFrame([[4.0], [8.0]], index=["a", "b"], columns=["x"])
    nav = pd.Series([4.0, 2.0], index=["b", "a"])

    pd.testing.assert_frame_equal(
        weights.nav_weights(positions, nav),
        pd.DataFrame([[2.0], [2.0]], index=positions.index, columns=positions.columns),
    )


def test_mixed_containers_use_positional_nav_and_preserve_first_container():
    """Mixed NumPy/pandas inputs match by position and keep value metadata."""
    positions = pd.DataFrame([[4.0, -2.0], [3.0, -1.0]], index=["a", "b"])

    result = weights.nav_weights(positions, np.array([2.0, 1.0]))

    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame([[2.0, -1.0], [3.0, -1.0]], index=positions.index, columns=positions.columns),
    )


@pytest.mark.parametrize("nav", [0.0, -1.0, np.nan])
def test_nonpositive_or_missing_nav_is_undefined(nav):
    """NAV-normalized weights are undefined without positive capital."""
    result = weights.nav_weights(np.array([20.0, -10.0]), nav)

    np.testing.assert_allclose(result, [np.nan, np.nan], equal_nan=True)


def test_zero_and_missing_gross_exposure_are_undefined():
    """Zero or missing position values cannot define gross weights or H."""
    for positions in (np.zeros(2), np.array([np.nan, 1.0])):
        np.testing.assert_allclose(
            weights.gross_weights(positions), [np.nan, np.nan], equal_nan=True
        )
        assert np.isnan(weights.concentration(positions))
        assert np.isnan(weights.effective_number_of_assets(positions))


def test_weights_are_scale_invariant_and_inputs_are_not_mutated():
    """Weight and concentration ratios ignore monetary scale and do not write."""
    positions = np.array([[10.0, -5.0], [np.nan, 3.0]])
    original = positions.copy()

    np.testing.assert_allclose(
        weights.gross_weights(positions * 7),
        weights.gross_weights(positions),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        weights.nav_weights(positions * 7, np.array([70.0, 70.0])),
        weights.nav_weights(positions, np.array([10.0, 10.0])),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        weights.concentration(positions * 7),
        weights.concentration(positions),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        weights.effective_number_of_assets(positions * 7),
        weights.effective_number_of_assets(positions),
        equal_nan=True,
    )
    np.testing.assert_allclose(positions, original, equal_nan=True)
