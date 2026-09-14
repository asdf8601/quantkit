<a id="module-quantkit.portfolio._utils"></a>

<a id="quantkit-portfolio-utils"></a>

# quantkit.portfolio._utils

Validation and container helpers for portfolio calculations.

<!-- !! processed by numpydoc !! -->

<a id="functions"></a>

## Functions

| [`_validate_labels`](#quantkit.portfolio._utils._validate_labels)(obj, name)              | Reject duplicate pandas labels in an asset input.                  |
|-------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| [`_real_array`](#quantkit.portfolio._utils._real_array)(obj, name)                        | Return a finite real numeric array, allowing missing values.       |
| [`asset_array`](#quantkit.portfolio._utils.asset_array)(obj\[, name\])                    | Validate asset data and return its NumPy representation.           |
| [`_same_labels`](#quantkit.portfolio._utils._same_labels)(reference, other, axis, name)   | Require equal unique labels and return `other` in reference order. |
| [`pair_assets`](#quantkit.portfolio._utils.pair_assets)(reference, other)                 | Validate and align paired asset inputs.                            |
| [`wrap_assets`](#quantkit.portfolio._utils.wrap_assets)(like, values)                     | Wrap asset-shaped values in the container and axes of `like`.      |
| [`wrap_reduction`](#quantkit.portfolio._utils.wrap_reduction)(like, values)               | Wrap an asset-axis reduction according to its input container.     |
| [`portfolio_amount`](#quantkit.portfolio._utils.portfolio_amount)(like, amount\[, name\]) | Validate and align a scalar or time-indexed portfolio amount.      |
| [`positive_divide`](#quantkit.portfolio._utils.positive_divide)(numerator, denominator)   | Divide where the denominator is positive, returning NaN otherwise. |

<a id="module-contents"></a>

## Module Contents

<a id="quantkit.portfolio._utils._validate_labels"></a>

### quantkit.portfolio._utils.\_validate_labels(obj, name)

Reject duplicate pandas labels in an asset input.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio._utils._real_array"></a>

### quantkit.portfolio._utils.\_real_array(obj, name)

Return a finite real numeric array, allowing missing values.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio._utils.asset_array"></a>

### quantkit.portfolio._utils.asset_array(obj, name='values')

Validate asset data and return its NumPy representation.

* **Parameters:**
  **obj** — numpy.ndarray, pandas.Series, or pandas.DataFrame
  : One-dimensional assets or a time-by-asset matrix.

  **name** — str, default “values”
  : Input name used in validation errors.
* **Returns:**
  numpy.ndarray
  : The validated one- or two-dimensional asset values.
* **Raises:**
  TypeError
  : If `obj` is not a supported container or has non-real values.

  ValueError
  : If axes are empty, labels are duplicated, or values include infinity.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio._utils._same_labels"></a>

### quantkit.portfolio._utils.\_same_labels(reference, other, axis, name)

Require equal unique labels and return `other` in reference order.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio._utils.pair_assets"></a>

### quantkit.portfolio._utils.pair_assets(reference, other)

Validate and align paired asset inputs.

The second input may be an asset vector for a matrix reference, in which case it is broadcast over time. Two pandas inputs align by their asset labels and, for DataFrames, by their time labels.

* **Parameters:**
  **reference** — numpy.ndarray, pandas.Series, or pandas.DataFrame
  : The input that supplies the result shape and pandas axes.

  **other** — numpy.ndarray, pandas.Series, or pandas.DataFrame
  : Values paired with `reference`.
* **Returns:**
  tuple\[numpy.ndarray, numpy.ndarray\]
  : Reference values and aligned values.
* **Raises:**
  ValueError
  : If dimensions, shapes, or pandas labels are incompatible.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio._utils.wrap_assets"></a>

### quantkit.portfolio._utils.wrap_assets(like, values)

Wrap asset-shaped values in the container and axes of `like`.

* **Parameters:**
  **like** — numpy.ndarray, pandas.Series, or pandas.DataFrame
  : Input whose container metadata is preserved.

  **values** — array-like
  : Asset-shaped values to wrap.
* **Returns:**
  numpy.ndarray, pandas.Series, or pandas.DataFrame
  : Values in the same container type as `like`.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio._utils.wrap_reduction"></a>

### quantkit.portfolio._utils.wrap_reduction(like, values)

Wrap an asset-axis reduction according to its input container.

* **Parameters:**
  **like** — numpy.ndarray, pandas.Series, or pandas.DataFrame
  : Input whose time axis is preserved for a DataFrame.

  **values** — array-like
  : Reduction values over the final asset axis.
* **Returns:**
  float, numpy.ndarray, or pandas.Series
  : A scalar for vectors, time values for matrices, or a time-indexed Series for DataFrames.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio._utils.portfolio_amount"></a>

### quantkit.portfolio._utils.portfolio_amount(like, amount, name='amount')

Validate and align a scalar or time-indexed portfolio amount.

* **Parameters:**
  **like** — numpy.ndarray, pandas.Series, or pandas.DataFrame
  : Asset input determining whether an amount is scalar or time-indexed.

  **amount** — scalar, array-like, or pandas.Series
  : Scalar amount, or one amount per row of a matrix input.

  **name** — str, default “amount”
  : Input name used in validation errors.
* **Returns:**
  float or numpy.ndarray
  : A scalar for vector inputs; a scalar or length-time array for matrix inputs.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.portfolio._utils.positive_divide"></a>

### quantkit.portfolio._utils.positive_divide(numerator, denominator)

Divide where the denominator is positive, returning NaN otherwise.

* **Parameters:**
  **numerator** — array-like
  : Values to divide.

  **denominator** — array-like
  : Positive divisors.
* **Returns:**
  float or numpy.ndarray
  : Quotients, with NaN where the denominator is nonpositive or missing.

<!-- !! processed by numpydoc !! -->
