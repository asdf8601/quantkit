<a id="module-quantkit.portfolio.risk"></a>

<a id="quantkit-portfolio-risk"></a>

# quantkit.portfolio.risk

Portfolio risk measures from a static covariance matrix.

<!-- !! processed by numpydoc !! -->

<a id="functions"></a>

## Functions

| [`_covariance_array`](#quantkit.portfolio.risk._covariance_array)(weights, covariance)                   | Validate, label-align, and symmetrize a covariance matrix.           |
|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| [`_risk_values`](#quantkit.portfolio.risk._risk_values)(weights, covariance)                             | Return validated weights, variances, volatilities, and covariance w. |
| [`variance`](#quantkit.portfolio.risk.variance)(weights, covariance)                                     | Return portfolio variance `w' C w`.                                  |
| [`volatility`](#quantkit.portfolio.risk.volatility)(weights, covariance)                                 | Return portfolio volatility `sqrt(w' C w)`.                          |
| [`marginal_risk_contribution`](#quantkit.portfolio.risk.marginal_risk_contribution)(weights, covariance) | Return marginal volatility contributions `C w / sqrt(w' C w)`.       |
| [`risk_contribution`](#quantkit.portfolio.risk.risk_contribution)(weights, covariance)                   | Return per-asset volatility contributions `w * C w / sqrt(w' C w)`.  |

<a id="module-contents"></a>

## Module Contents

<a id="quantkit.portfolio.risk._covariance_array"></a>

### quantkit.portfolio.risk.\_covariance_array(weights, covariance)

Validate, label-align, and symmetrize a covariance matrix.

Accepted asymmetry is at most `1e-12 * max(abs(covariance))`.  The symmetric matrix must have no eigenvalue below `-1e-12 * max(abs(eigenvalues))`.  These relative tolerances retain valid matrices affected only by floating-point roundoff; a zero matrix is valid.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.risk._risk_values"></a>

### quantkit.portfolio.risk.\_risk_values(weights, covariance)

Return validated weights, variances, volatilities, and covariance w.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.risk.variance"></a>

### quantkit.portfolio.risk.variance(weights, covariance)

Return portfolio variance `w' C w`.

* **Parameters:**
  **weights** — numpy.ndarray or pandas.Series or pandas.DataFrame
  : Asset weights as a vector or time-by-asset matrix. Short weights are valid, and missing weights propagate through their complete row.

  **covariance** — numpy.ndarray or pandas.DataFrame
  : One static asset covariance matrix. A DataFrame aligns to pandas weight asset labels.
* **Returns:**
  float or numpy.ndarray or pandas.Series
  : Variance at the covariance matrix’s input frequency. Matrix weights retain their time axis.

### Notes

This function does not estimate or annualize covariance. The covariance matrix must be symmetric positive semidefinite; tiny accepted asymmetry is symmetrized before calculation. Negative variance from floating-point roundoff after validation returns zero.

### References

<a id="r57691d30d30b-1"></a>

[1]

[https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html](https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html)

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.risk.volatility"></a>

### quantkit.portfolio.risk.volatility(weights, covariance)

Return portfolio volatility `sqrt(w' C w)`.

* **Parameters:**
  **weights** — numpy.ndarray or pandas.Series or pandas.DataFrame
  : Asset weights as a vector or time-by-asset matrix.

  **covariance** — numpy.ndarray or pandas.DataFrame
  : One static asset covariance matrix at the desired output frequency.
* **Returns:**
  float or numpy.ndarray or pandas.Series
  : Volatility at the covariance matrix’s input frequency. Matrix weights retain their time axis.

### Notes

This function does not estimate or annualize covariance. Zero portfolio variance returns zero volatility.

### References

<a id="r1b2ecf604287-1"></a>

[1]

[https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html](https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html)

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.risk.marginal_risk_contribution"></a>

### quantkit.portfolio.risk.marginal_risk_contribution(weights, covariance)

Return marginal volatility contributions `C w / sqrt(w' C w)`.

* **Parameters:**
  **weights** — numpy.ndarray or pandas.Series or pandas.DataFrame
  : Asset weights as a vector or time-by-asset matrix.

  **covariance** — numpy.ndarray or pandas.DataFrame
  : One static asset covariance matrix at the desired output frequency.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Per-asset marginal volatility contributions. Zero-volatility rows are undefined and contain NaN.

### Notes

This function does not estimate or annualize covariance. The covariance frequency determines the contribution frequency.

### References

<a id="rae0fb563a47e-1"></a>

[1]

[https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html](https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html)

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio.risk.risk_contribution"></a>

### quantkit.portfolio.risk.risk_contribution(weights, covariance)

Return per-asset volatility contributions `w * C w / sqrt(w' C w)`.

* **Parameters:**
  **weights** — numpy.ndarray or pandas.Series or pandas.DataFrame
  : Asset weights as a vector or time-by-asset matrix. Signed weights may produce negative contributions when an asset hedges portfolio risk.

  **covariance** — numpy.ndarray or pandas.DataFrame
  : One static asset covariance matrix at the desired output frequency.
* **Returns:**
  numpy.ndarray or pandas.Series or pandas.DataFrame
  : Per-asset contributions that sum to volatility for positive-volatility portfolios. Zero-volatility rows contain NaN.

### Notes

This function does not estimate or annualize covariance. The covariance frequency determines the contribution frequency.

### References

<a id="r1f07f8445bda-1"></a>

[1]

[https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html](https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html)

<!-- !! processed by numpydoc !! -->
