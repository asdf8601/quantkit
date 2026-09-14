<a id="module-quantkit.expanding"></a>

<a id="quantkit-expanding"></a>

# quantkit.expanding

Financial statistics applied using an expanding window basis approach.

<!-- !! processed by numpydoc !! -->

<a id="functions"></a>

## Functions

| [`drawdown`](#quantkit.expanding.drawdown)(prices\[, relative, out\])   | Drawdown of a given prices series.   |
|-------------------------------------------------------------------------|--------------------------------------|
| [`drawup`](#quantkit.expanding.drawup)(prices\[, relative, out\])       | Drawup of a given prices series.     |

<a id="module-contents"></a>

## Module Contents

<a id="quantkit.expanding.drawdown"></a>

### quantkit.expanding.drawdown(prices, relative=True, out=None)

Drawdown of a given prices series.

The drawdown of a price process $S$ at time $t$ is defined as the drop of the asset prices from its running maximum up to time $t$

$$
D_{t} = \max_{u \in [0, t]}(S_{u}) - S_{t}
$$

This means that, by definition, the drawdown should be a strictly positive series. However, the results are often presented as a negative series for better understanding, and that is the way we chose to implement here.

$$
D_{t} = S_{t} - \max_{u \in [0, t]}(S_{u})
$$

* **Parameters:**
  **prices** — array-like
  : Series on which the drawdown is to be calculated.

  **relative** — bool, optional, default: False
  : Passing True makes the drawdown series relative (in parts per unit).

  **out** — array-like, optional, default: None
  : Alternative output array in which to place the result. It must have the same shape and buffer length as the expected output but the type will be cast if necessary.
* **Returns:**
  **out** — array-like
  : Drawdown series.

### References

<a id="r3fdda46605e1-1"></a>

[1]

Jan Vecer - Maximum Drawdown and Directional Trading [http://www.stat.columbia.edu/~vecer/maxdrawdown3.pdf](http://www.stat.columbia.edu/~vecer/maxdrawdown3.pdf)

<a id="r3fdda46605e1-2"></a>

[2]

Enrico Schumann - Computing Drawdown Statistics [http://comisef.wikidot.com/tutorial:drawdowns](http://comisef.wikidot.com/tutorial:drawdowns)

<!-- !! processed by numpydoc !! -->

<a id="quantkit.expanding.drawup"></a>

### quantkit.expanding.drawup(prices, relative=True, out=None)

Drawup of a given prices series.

The drawup of a price process $S$ at time $t$ is defined as the rise of the asset prices from its running minimum up to time $t$, the mirror image of the drawdown

$$
U_{t} = S_{t} - \min_{u \in [0, t]}(S_{u})
$$

By definition the drawup is a positive series. Passing `relative=True` divides it by the running minimum, giving the rise in parts per unit. Where the running minimum is zero the relative drawup is NaN.

* **Parameters:**
  **prices** — array-like
  : Series on which the drawup is to be calculated.

  **relative** — bool, optional, default: True
  : Passing True makes the drawup series relative (in parts per unit).

  **out** — array-like, optional, default: None
  : Alternative output array in which to place the result. It must have the same shape and buffer length as the expected output but the type will be cast if necessary.
* **Returns:**
  **out** — array-like
  : Drawup series.

### References

<a id="rc0f2f1f2690d-1"></a>

[1]

Jan Vecer - Maximum Drawdown and Directional Trading, Risk 19(12), 2006. Maximum drawdown and maximum drawup are studied as a pair. [http://www.stat.columbia.edu/~vecer/maxdrawdown3.pdf](http://www.stat.columbia.edu/~vecer/maxdrawdown3.pdf)

### Examples

```pycon
>>> import numpy as np
>>> drawup(np.array([4, 2, 3, 6]), relative=False)
0    0.0
1    0.0
2    1.0
3    4.0
dtype: float64
```

```pycon
>>> drawup(np.array([4, 2, 3, 6]), relative=True)
0    0.0
1    0.0
2    0.5
3    2.0
dtype: float64
```

<!-- !! processed by numpydoc !! -->
