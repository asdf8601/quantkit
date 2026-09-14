<a id="module-quantkit.core"></a>

<a id="quantkit-core"></a>

# quantkit.core

Core module contains common functions to perform finance analysis.

<!-- !! processed by numpydoc !! -->

<a id="functions"></a>

## Functions

| [`returns`](#quantkit.core.returns)(prices\[, period, out\])               | Arithmetic Returns.                      |
|----------------------------------------------------------------------------|------------------------------------------|
| [`cum_returns`](#quantkit.core.cum_returns)(returns\[, first_price, out\]) | Cummulative arithmetic returns.          |
| [`rebase`](#quantkit.core.rebase)(prices\[, base, out\])                   | Rebase prices to start in the same base. |

<a id="module-contents"></a>

## Module Contents

<a id="quantkit.core.returns"></a>

### quantkit.core.returns(prices, period=1, out=None)

Arithmetic Returns.

Calculate the returns of a price series.

* **Parameters:**
  **prices** — array-like
  : Series on which the returns are to be calculated.

  **period** — int
  : Distance between prices to perform the returns.

  **out** — array-like, optional
  : Alternative output array in which to place the result. It must have the same shape and buffer length as the expected output but the type will be cast if necessary.
* **Returns:**
  **returns** — array-like
  : Returns series.

### Examples

```pycon
>>> import numpy as np
>>> returns(np.array([1, 2, 3]))
array([nan , 1. , 0.5 ])
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.core.cum_returns"></a>

### quantkit.core.cum_returns(returns, first_price=None, out=None)

Cummulative arithmetic returns.

* **Parameters:**
  **returns** — array-like
  : Return series which will be used to perform the calculation.

  **first_price** — float, optional
  : Passing a first_price you can control the first element of the cummulative returns series performed by this function.

  **out** — array-like, optional
  : Alternative output array in which to place the result. It must have the same shape and buffer length as the expected output but the type will be cast if necessary.
* **Returns:**
  **cumreturns** — array-like
  : Cummulative returns series.

### Examples

```pycon
>>> import numpy as np
>>> cum_returns(np.array([np.nan, 1, 0.5]))
array([0. , 1. , 2. ])
```

Also you can specify the first price:

```pycon
>>> cum_returns(returns(np.array([1, 2, 3])), first_price=1)
array([1. , 2. , 3. ])
```

<!-- !! processed by numpydoc !! -->

<a id="quantkit.core.rebase"></a>

### quantkit.core.rebase(prices, base=100, out=None)

Rebase prices to start in the same base.

* **Parameters:**
  **prices** — array-like
  : Series on which the returns are to be calculated.

  **base** — float
  : prices start point.

  **out** — array-like, optional
  : Alternative output array in which to place the result. It must have the same shape and buffer length as the expected output but the type will be cast if necessary.
* **Returns:**
  **price_rebased** — array-like
  : Price series rebased.

<!-- !! processed by numpydoc !! -->
