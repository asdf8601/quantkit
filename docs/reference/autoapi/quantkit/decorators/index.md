<a id="module-quantkit.decorators"></a>

<a id="quantkit-decorators"></a>

# quantkit.decorators

NumPy and pandas wrappers.

Helpers for adapting function inputs and wrapping results as NumPy or pandas objects. Use the decorators for transformations that preserve shape, and `reduce_array_wrap` for results that reduce one dimension.

<a id="quick-example"></a>

## Quick example

Compute one total per DataFrame column and retain the column labels:

```pycon
>>> import pandas as pd
>>> from quantkit.decorators import reduce_array_wrap
>>> positions = pd.DataFrame({"stocks": [10, 20], "bonds": [5, 15]})
>>> totals = positions.to_numpy().sum(axis=0)
>>> reduce_array_wrap(positions, totals)
stocks    30
bonds     20
dtype: int64
```

The function reference below describes input conversion, shape-preserving output wrapping and reduction wrapping.

<!-- !! processed by numpydoc !! -->

<a id="functions"></a>

## Functions

| [`numpy2pandas_args_wrapper`](#quantkit.decorators.numpy2pandas_args_wrapper)(\*pos)   | Convert selected positional NumPy inputs to pandas objects.    |
|----------------------------------------------------------------------------------------|----------------------------------------------------------------|
| [`array_output_wrapper`](#quantkit.decorators.array_output_wrapper)(pos)               | Wrap a result using a selected input's container and labels.   |
| [`reduce_array_wrap`](#quantkit.decorators.reduce_array_wrap)(obj, res)                | Wrap a reduction result using the original input's dimensions. |

<a id="module-contents"></a>

## Module Contents

<a id="quantkit.decorators.numpy2pandas_args_wrapper"></a>

### quantkit.decorators.numpy2pandas_args_wrapper(\*pos)

Convert selected positional NumPy inputs to pandas objects.

One-dimensional arrays become Series and two-dimensional arrays become DataFrames, both with default integer labels. Existing pandas inputs pass through unchanged. This decorator does not convert the return value.

* **Parameters:**
  **\*pos** — int
  : Zero-based positions of arguments to convert. Selected arguments must be passed positionally; keyword arguments are not converted.
* **Returns:**
  **decorated_func** — function
  : Decorator that converts selected inputs before calling the function.
* **Raises:**
  NotImplementedError
  : If a selected NumPy input has a dimension other than one or two.

### Examples

```pycon
>>> import numpy as np
>>> @numpy2pandas_args_wrapper(0)
... def column_totals(values):
...     return values.sum()
>>> column_totals(np.array([[1, 2], [3, 4]]))
0    4
1    6
dtype: int64
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.decorators.array_output_wrapper"></a>

### quantkit.decorators.array_output_wrapper(pos)

Wrap a result using a selected input’s container and labels.

Use this decorator for a function that returns one array-like result with its input’s shape. For pandas inputs, preserve the index, DataFrame columns and Series name. For NumPy inputs, return a NumPy array.

* **Parameters:**
  **pos** — int
  : Zero-based position of the input whose container and labels to use. This input must be passed positionally.
* **Returns:**
  **decorated_func** — function
  : Decorator that wraps the function’s result using the selected input.

### Notes

The function must return a single array-like result. Multiple return values are not supported. For dimension-reducing results, use `reduce_array_wrap` instead.

### Examples

```pycon
>>> import numpy as np
>>> import pandas as pd
>>> @array_output_wrapper(0)
... def double(values):
...     return np.asarray(values) * 2
>>> double(pd.Series([1, 2], index=["a", "b"], name="positions"))
a    2
b    4
Name: positions, dtype: int64
>>> double(np.array([1, 2]))
array([2, 4])
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.decorators.reduce_array_wrap"></a>

### quantkit.decorators.reduce_array_wrap(obj, res)

Wrap a reduction result using the original input’s dimensions.

For a DataFrame, return a Series indexed by its columns. For a two-dimensional NumPy input, return `res` as a NumPy array. For a one-dimensional input, return `res` unchanged, typically a scalar. This helper wraps an already computed result; it does not perform the reduction.

* **Parameters:**
  **obj** — numpy.ndarray or pandas.Series or pandas.DataFrame
  : Original one- or two-dimensional input, used to determine the output container and labels.

  **res** — array-like or scalar
  : Computed reduction result. For a DataFrame, supply one value per column in the original column order.
* **Returns:**
  **out** — pandas.Series or numpy.ndarray or scalar
  : Wrapped result for two-dimensional inputs, or `res` unchanged for one-dimensional inputs.
* **Raises:**
  NotImplementedError
  : If `obj` has a dimension other than one or two.

### Examples

```pycon
>>> import numpy as np
>>> import pandas as pd
>>> values = np.array([[1, 2], [3, 4]])
>>> reduce_array_wrap(values, values.sum(axis=0))
array([4, 6])
>>> frame = pd.DataFrame(values, columns=["a", "b"])
>>> reduce_array_wrap(frame, values.sum(axis=0))
a    4
b    6
dtype: int64
>>> series = pd.Series([1, 2, 3])
>>> int(reduce_array_wrap(series, series.sum()))
6
```

<!-- !! processed by numpydoc !! -->
