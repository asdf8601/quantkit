<a id="module-quantkit.decorators"></a>

<a id="quantkit-decorators"></a>

# quantkit.decorators

Decorators module.

f(x) -> y

x : pandas.DataFrame, pandas.Series, np.array (, list)

f(x): x -> array_protocol -> x_arr -> do stuff -> out : the same type as x

Ej1 x : np.array\[n,m\] y : np.array\[m\]

Ej2 x : pandas.DataFrame y : pandas.Series

Ej3 x : np.array\[m\] y : float

f(x): x -> array_protocol -> x_arr -> do stuff -> out : the same type as x

intput to {pandas, numpy} -> output to {pandas, numpy}:

- array_wrap_reduce  (collapse one dimension)
- array_wrap_transform  (non-collapsing)
- array_wrap_increase  (increse one dimension)

<!-- !! processed by numpydoc !! -->

<a id="functions"></a>

## Functions

| [`_np2pd`](#quantkit.decorators._np2pd)(np_obj)                                      |                                                                         |
|--------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| [`numpy2pandas_args_wrapper`](#quantkit.decorators.numpy2pandas_args_wrapper)(\*pos) | Convert numpy to pandas output objects according to an input parameter. |
| [`array_output_wrapper`](#quantkit.decorators.array_output_wrapper)(pos)             | Preserves output according to the input of the decorated functions.     |
| [`reduce_array_wrap`](#quantkit.decorators.reduce_array_wrap)(obj, res)              | Array wrap for reducing functions.                                      |

<a id="module-contents"></a>

## Module Contents

<a id="quantkit.decorators._np2pd"></a>

### quantkit.decorators.\_np2pd(np_obj)

<a id="quantkit.decorators.numpy2pandas_args_wrapper"></a>

### quantkit.decorators.numpy2pandas_args_wrapper(\*pos)

Convert numpy to pandas output objects according to an input parameter.

This decorator receive positions of the parameters in the decorated function to pick the numpy object and convert it to pandas preserving the orignal dimension.

* **Parameters:**
  **\*pos** — int
  : Parameter position in the decorated function which will be converted to pandas.
* **Returns:**
  **decorated_func** — function

<!-- !! processed by numpydoc !! -->

<a id="quantkit.decorators.array_output_wrapper"></a>

### quantkit.decorators.array_output_wrapper(pos)

Preserves output according to the input of the decorated functions.

Decorator which converts the decorated function’s output in the same object as one of the input parameters. This allow us preserve the same object type in the output.

* **Parameters:**
  **pos** — int
  : Parameter position of the decorated function to preserve object type from.
* **Returns:**
  **decorated_func** — function

<!-- !! processed by numpydoc !! -->

<a id="quantkit.decorators.reduce_array_wrap"></a>

### quantkit.decorators.reduce_array_wrap(obj, res)

Array wrap for reducing functions.

Allow easly wrap the result of a reducing function in the proper dimension of the original array-like object.

* **Parameters:**
  **obj** — pandas or numpy
  : Object to access.

  **res** — array-like or number
  : Indexer allowed by obj.
* **Returns:**
  **out** — array-like or number
* **Raises:**
  NotImplementedError
  : If `obj` has higher dimension than 2 or less than 1.

### Examples

```pycon
>>> import numpy as np
>>> import pandas as pd
```

Numpy objects

```pycon
>>> obj = np.array([[0], [1]])
>>> obj_reduced = np.array([22])
>>> reduce_array_wrap(obj, obj_reduced)
```

Pandas objects

```pycon
>>> obj = pd.DataFrame(np.array([[0], [1]]))
>>> obj_reduced = np.array([22])
>>> reduce_array_wrap(obj, obj_reduced)
```

More realistic example

```pycon
>>> obj = pd.DataFrame(np.array([[0], [1]]))
>>> obj_reduced = obj.sum()  # apply a reduction function
>>> reduce_array_wrap(obj, obj_reduced)
```

<!-- !! processed by numpydoc !! -->
