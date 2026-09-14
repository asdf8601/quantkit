"""Portfolio risk measures from a static covariance matrix."""

import numpy as np
import pandas as pd

from ._utils import asset_array, wrap_assets, wrap_reduction


def _covariance_array(weights, covariance):
    """Validate, label-align, and symmetrize a covariance matrix.

    Accepted asymmetry is at most ``1e-12 * max(abs(covariance))``.  The
    symmetric matrix must have no eigenvalue below
    ``-1e-12 * max(abs(eigenvalues))``.  These relative tolerances retain
    valid matrices affected only by floating-point roundoff; a zero matrix is
    valid.
    """
    if not isinstance(covariance, (np.ndarray, pd.DataFrame)):
        raise TypeError("covariance must be a numpy array or pandas DataFrame")

    if isinstance(covariance, pd.DataFrame):
        if covariance.index.has_duplicates:
            raise ValueError("covariance has duplicate index labels")
        if covariance.columns.has_duplicates:
            raise ValueError("covariance has duplicate column labels")
        if isinstance(weights, pd.Series):
            assets = weights.index
        elif isinstance(weights, pd.DataFrame):
            assets = weights.columns
        else:
            assets = None

        if assets is not None:
            if (
                len(covariance.index) != len(assets)
                or not covariance.index.isin(assets).all()
            ):
                raise ValueError("covariance index labels must match weights")
            if (
                len(covariance.columns) != len(assets)
                or not covariance.columns.isin(assets).all()
            ):
                raise ValueError("covariance column labels must match weights")
            covariance = covariance.reindex(index=assets, columns=assets)
        elif (
            len(covariance.index) != len(covariance.columns)
            or not covariance.index.isin(covariance.columns).all()
        ):
            raise ValueError("covariance index and column labels must match")
        else:
            covariance = covariance.reindex(columns=covariance.index)

        dtypes = covariance.dtypes
        if any(dtype.kind not in "iuf" for dtype in dtypes):
            raise TypeError("covariance must contain real numeric values")
        values = covariance.to_numpy(dtype=float, na_value=np.nan)
    else:
        if covariance.dtype.kind not in "iuf":
            raise TypeError("covariance must contain real numeric values")
        values = covariance.astype(float, copy=False)

    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if values.shape[0] != asset_array(weights, "weights").shape[-1]:
        raise ValueError("covariance must match the number of assets")
    if not np.isfinite(values).all():
        raise ValueError("covariance must contain finite values")

    maximum = np.max(np.abs(values))
    if np.max(np.abs(values - values.T)) > 1e-12 * maximum:
        raise ValueError("covariance must be symmetric")
    values = values / 2.0 + values.T / 2.0

    eigenvalues = np.linalg.eigvalsh(values)
    tolerance = 1e-12 * np.max(np.abs(eigenvalues))
    if np.min(eigenvalues) < -tolerance:
        raise ValueError("covariance must be positive semidefinite")
    return values


def _risk_values(weights, covariance):
    """Return validated weights, variances, volatilities, and covariance w."""
    values = asset_array(weights, "weights")
    matrix = _covariance_array(weights, covariance)
    covariance_weights = np.matmul(values, matrix)
    variances = np.sum(values * covariance_weights, axis=-1)
    variances = np.maximum(variances, 0.0)
    volatilities = np.sqrt(variances)
    return values, variances, volatilities, covariance_weights


def variance(weights, covariance):
    """Return portfolio variance ``w' C w``.

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series or pandas.DataFrame
        Asset weights as a vector or time-by-asset matrix. Short weights are
        valid, and missing weights propagate through their complete row.
    covariance : numpy.ndarray or pandas.DataFrame
        One static asset covariance matrix. A DataFrame aligns to pandas
        weight asset labels.

    Returns
    -------
    float or numpy.ndarray or pandas.Series
        Variance at the covariance matrix's input frequency. Matrix weights
        retain their time axis.

    Notes
    -----
    This function does not estimate or annualize covariance. The covariance
    matrix must be symmetric positive semidefinite; tiny accepted asymmetry is
    symmetrized before calculation. Negative variance from floating-point
    roundoff after validation returns zero.

    References
    ----------
    .. [1] https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html
    """
    _, variances, _, _ = _risk_values(weights, covariance)
    return wrap_reduction(weights, variances)


def volatility(weights, covariance):
    """Return portfolio volatility ``sqrt(w' C w)``.

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series or pandas.DataFrame
        Asset weights as a vector or time-by-asset matrix.
    covariance : numpy.ndarray or pandas.DataFrame
        One static asset covariance matrix at the desired output frequency.

    Returns
    -------
    float or numpy.ndarray or pandas.Series
        Volatility at the covariance matrix's input frequency. Matrix weights
        retain their time axis.

    Notes
    -----
    This function does not estimate or annualize covariance. Zero portfolio
    variance returns zero volatility.

    References
    ----------
    .. [1] https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html
    """
    _, _, volatilities, _ = _risk_values(weights, covariance)
    return wrap_reduction(weights, volatilities)


def marginal_risk_contribution(weights, covariance):
    """Return marginal volatility contributions ``C w / sqrt(w' C w)``.

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series or pandas.DataFrame
        Asset weights as a vector or time-by-asset matrix.
    covariance : numpy.ndarray or pandas.DataFrame
        One static asset covariance matrix at the desired output frequency.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Per-asset marginal volatility contributions. Zero-volatility rows are
        undefined and contain NaN.

    Notes
    -----
    This function does not estimate or annualize covariance. The covariance
    frequency determines the contribution frequency.

    References
    ----------
    .. [1] https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html
    """
    values, _, volatilities, covariance_weights = _risk_values(
        weights, covariance
    )
    denominator = (
        volatilities if values.ndim == 1 else volatilities[:, np.newaxis]
    )
    result = np.full(values.shape, np.nan, dtype=float)
    np.divide(
        covariance_weights,
        denominator,
        out=result,
        where=denominator > 0,
    )
    return wrap_assets(weights, result)


def risk_contribution(weights, covariance):
    """Return per-asset volatility contributions ``w * C w / sqrt(w' C w)``.

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series or pandas.DataFrame
        Asset weights as a vector or time-by-asset matrix. Signed weights may
        produce negative contributions when an asset hedges portfolio risk.
    covariance : numpy.ndarray or pandas.DataFrame
        One static asset covariance matrix at the desired output frequency.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Per-asset contributions that sum to volatility for positive-volatility
        portfolios. Zero-volatility rows contain NaN.

    Notes
    -----
    This function does not estimate or annualize covariance. The covariance
    frequency determines the contribution frequency.

    References
    ----------
    .. [1] https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html
    """
    values, _, volatilities, covariance_weights = _risk_values(
        weights, covariance
    )
    denominator = (
        volatilities if values.ndim == 1 else volatilities[:, np.newaxis]
    )
    marginal = np.full(values.shape, np.nan, dtype=float)
    np.divide(
        covariance_weights,
        denominator,
        out=marginal,
        where=denominator > 0,
    )
    return wrap_assets(weights, values * marginal)
