<a id="module-quantkit.utils"></a>

<a id="quantkit-utils"></a>

# quantkit.utils

Module of utilities.

<!-- !! processed by numpydoc !! -->

<a id="functions"></a>

## Functions

| [`first_valid_index`](#quantkit.utils.first_valid_index)(array)   | Find the first no null value and returns its index.                |
|-------------------------------------------------------------------|--------------------------------------------------------------------|
| [`last_valid_index`](#quantkit.utils.last_valid_index)(array)     | Find the last no null value and returns its index.                 |
| [`iloc`](#quantkit.utils.iloc)(obj, idx)                          | Access to idx in obj.                                              |
| [`array_wrap`](#quantkit.utils.array_wrap)(like, values)          | Wrap `values` in the same container type as `like`.                |
| [`align`](#quantkit.utils.align)(returns, benchmark)              | Align `returns` with a 1D `benchmark` and return both as ndarrays. |

<a id="module-contents"></a>

## Module Contents

<a id="quantkit.utils.first_valid_index"></a>

### quantkit.utils.first_valid_index(array)

Find the first no null value and returns its index.

* **Parameters:**
  **array** — array-like
  : Numpy or pandas (or array-like) object.
* **Returns:**
  **index** — int
  : The first position valid.
* **Raises:**
  NotImplementedError
  : If array has higher dimension than 2.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.utils.last_valid_index"></a>

### quantkit.utils.last_valid_index(array)

Find the last no null value and returns its index.

* **Parameters:**
  **array** — array-like
  : Numpy or pandas (or array-like) object.
* **Returns:**
  **index** — int
  : The last position valid.
* **Raises:**
  NotImplementedError
  : If array has higher dimension than 2.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.utils.iloc"></a>

### quantkit.utils.iloc(obj, idx)

Access to idx in obj.

This is a wrapper to access in the same way to pandas and numpy objects.

* **Parameters:**
  **obj** — pandas or numpy
  : Object to access.

  **idx** — indexer
  : Indexer allowed by obj.
* **Returns:**
  **obj_sliced** — pandas or numpy
* **Raises:**
  NotImplementedError
  : If obj is not a pandas or numpy object handled by this function.

<!-- !! processed by numpydoc !! -->

<a id="quantkit.utils.array_wrap"></a>

### quantkit.utils.array_wrap(like, values)

Wrap `values` in the same container type as `like`.

Public replacement for `like.__array_wrap__(values)`, removed from pandas 2.0. Pandas objects are rebuilt with the axes and name of `like` without copying `values`; numpy objects keep numpy’s own protocol.

* **Parameters:**
  **like** — pandas.Series, pandas.DataFrame or numpy.ndarray
  : Object whose type and axes are preserved.

  **values** — array-like
  : Data to wrap. Must match the shape of `like`.
* **Returns:**
  **out** — same type as `like`

<!-- !! processed by numpydoc !! -->

<a id="quantkit.utils.align"></a>

### quantkit.utils.align(returns, benchmark)

Align `returns` with a 1D `benchmark` and return both as ndarrays.

Pandas objects are inner-joined on their index, so only the labels known on both sides survive. As soon as one side is a numpy array there is no index to join on and both inputs must have the same length.

* **Parameters:**
  **returns** — pandas.Series, pandas.DataFrame or numpy.ndarray
  : 1D or 2D returns.

  **benchmark** — pandas.Series or numpy.ndarray
  : 1D benchmark returns.
* **Returns:**
  **arr_returns** — numpy.ndarray
  : Aligned returns, with the same number of dimensions as `returns`.

  **arr_benchmark** — numpy.ndarray
  : Aligned 1D benchmark.
* **Raises:**
  ValueError
  : If `benchmark` is not 1D, or if either input is a numpy array and the lengths differ.

### Examples

```pycon
>>> returns = pd.Series([1.0, 2.0, 3.0], index=[1, 2, 3])
>>> benchmark = pd.Series([10.0, 20.0, 30.0], index=[0, 1, 2])
>>> align(returns, benchmark)
(array([1., 2.]), array([20., 30.]))
```

<!-- !! processed by numpydoc !! -->
