"""Tests for portfolio covariance risk measures."""

import warnings

import numpy as np
import pandas as pd
import pytest

from quantkit.portfolio import risk


def test_diagonal_covariance_has_analytical_risk_contributions():
    weights = np.array([0.5, 0.5])
    covariance = np.diag([0.04, 0.09])
    expected_variance = 0.0325
    expected_volatility = np.sqrt(expected_variance)

    assert risk.variance(weights, covariance) == pytest.approx(
        expected_variance
    )
    assert risk.volatility(weights, covariance) == pytest.approx(
        expected_volatility
    )
    np.testing.assert_allclose(
        risk.marginal_risk_contribution(weights, covariance),
        [0.02 / expected_volatility, 0.045 / expected_volatility],
    )
    np.testing.assert_allclose(
        risk.risk_contribution(weights, covariance),
        [0.01 / expected_volatility, 0.0225 / expected_volatility],
    )


def test_risk_contributions_sum_to_volatility_and_match_derivative():
    weights = np.array([0.4, 0.6])
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    epsilon = 1e-6
    numerical = np.array(
        [
            (
                risk.volatility(weights + epsilon * basis, covariance)
                - risk.volatility(weights - epsilon * basis, covariance)
            )
            / (2 * epsilon)
            for basis in np.eye(2)
        ]
    )

    marginal = risk.marginal_risk_contribution(weights, covariance)
    contributions = risk.risk_contribution(weights, covariance)

    np.testing.assert_allclose(marginal, numerical, rtol=1e-6)
    assert np.sum(contributions) == pytest.approx(
        risk.volatility(weights, covariance)
    )


def test_signed_hedge_can_have_negative_risk_contribution():
    weights = np.array([1.0, -0.25])
    covariance = np.array([[0.04, 0.03], [0.03, 0.04]])

    contribution = risk.risk_contribution(weights, covariance)

    assert contribution[1] < 0
    assert np.sum(contribution) == pytest.approx(
        risk.volatility(weights, covariance)
    )


def test_singular_and_zero_covariances_are_supported():
    weights = np.array([[0.5, 0.5], [1.0, -1.0]])
    singular = np.array([[0.04, 0.04], [0.04, 0.04]])

    np.testing.assert_allclose(risk.variance(weights, singular), [0.04, 0.0])
    np.testing.assert_allclose(risk.volatility(weights, singular), [0.2, 0.0])
    np.testing.assert_allclose(
        risk.risk_contribution(weights, singular)[0], [0.1, 0.1]
    )
    assert np.isnan(risk.risk_contribution(weights, singular)[1]).all()
    assert risk.variance(np.array([0.5, 0.5]), np.zeros((2, 2))) == 0.0


def test_pandas_assets_align_covariance_axes_and_preserve_containers():
    weights = pd.DataFrame(
        [[0.5, 0.5], [0.2, 0.8]],
        index=pd.Index(["first", "second"], name="time"),
        columns=pd.Index(["b", "a"], name="asset"),
    )
    covariance = pd.DataFrame(
        [[0.09, 0.01], [0.01, 0.04]],
        index=["a", "b"],
        columns=["a", "b"],
    )

    expected_variance = pd.Series([0.0375, 0.0624], index=weights.index)
    pd.testing.assert_series_equal(
        risk.variance(weights, covariance), expected_variance
    )
    pd.testing.assert_frame_equal(
        risk.risk_contribution(weights, covariance),
        pd.DataFrame(
            [[0.0125, 0.025], [0.0032, 0.0592]],
            index=weights.index,
            columns=weights.columns,
        ).div(np.sqrt(expected_variance), axis=0),
    )


def test_numpy_weights_accept_reordered_dataframe_covariance_axes():
    covariance = pd.DataFrame(
        [[0.01, 0.04], [0.09, 0.01]],
        index=["b", "a"],
        columns=["a", "b"],
    )

    assert risk.variance(np.array([0.5, 0.5]), covariance) == pytest.approx(
        0.0375
    )


def test_nullable_pandas_inputs_are_supported():
    weights = pd.Series([0.5, 0.5], index=["a", "b"], dtype="Float64")
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.09]],
        index=["a", "b"],
        columns=["a", "b"],
        dtype="Float64",
    )

    result = risk.volatility(weights, covariance)

    assert isinstance(result, float)
    assert result == pytest.approx(np.sqrt(0.0325))


@pytest.mark.parametrize(
    "covariance",
    [
        np.ones((2, 3)),
        np.array([[0.04, np.nan], [np.nan, 0.09]]),
        np.array([[0.04, 0.02], [0.01, 0.09]]),
        np.diag([-1e-14, 1e-14]),
        pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["a", "a"]),
    ],
)
def test_invalid_covariances_are_rejected(covariance):
    with pytest.raises((TypeError, ValueError)):
        risk.variance(np.array([0.5, 0.5]), covariance)


def test_covariance_labels_must_match_pandas_weight_assets():
    weights = pd.Series([0.5, 0.5], index=["a", "b"])
    covariance = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]], index=["a", "c"], columns=["a", "c"]
    )

    with pytest.raises(ValueError, match="labels must match"):
        risk.variance(weights, covariance)


def test_missing_weights_propagate_through_the_complete_row():
    weights = np.array([[0.5, 0.5], [np.nan, 0.5]])
    covariance = np.diag([0.04, 0.09])

    variance = risk.variance(weights, covariance)
    contribution = risk.risk_contribution(weights, covariance)

    assert np.isnan(variance[1])
    assert np.isnan(contribution[1]).all()


def test_psd_tolerance_clips_negative_variance_without_warnings():
    covariance = np.array(
        [[-1.5e-12, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
    )
    weights = np.array([1.0, 0.0, 0.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert risk.variance(weights, covariance) == 0.0
        assert risk.volatility(weights, covariance) == 0.0
        assert np.isnan(
            risk.marginal_risk_contribution(weights, covariance)
        ).all()
        assert np.isnan(risk.risk_contribution(weights, covariance)).all()
