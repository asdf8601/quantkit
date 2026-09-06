"""Every reducer of :mod:`quantkit.stats` returns NaN on empty input.

Empty here means "no observation to reduce": a 2-dimensional object with
zero rows, or an empty 1-dimensional one. The library convention is that an
undefined result is NaN, so none of these may raise or warn.
"""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

from quantkit import stats


def _public_reducers():
    """Every public function defined in ``quantkit.stats``, by name."""
    return [
        (name, func)
        for name, func in inspect.getmembers(stats, inspect.isfunction)
        if not name.startswith("_") and func.__module__ == stats.__name__
    ]


REDUCERS = _public_reducers()
REDUCER_IDS = [name for name, _ in REDUCERS]
REDUCER_FUNCS = [func for _, func in REDUCERS]

# built fresh on every call so no reducer can see another one's input
CASES = [
    ("numpy-2d", lambda: np.empty((0, 2))),
    ("pandas-2d", lambda: pd.DataFrame(columns=["a", "b"], dtype=float)),
    ("numpy-1d", lambda: np.array([])),
]
CASE_IDS = [case for case, _ in CASES]


def _call(func, data):
    """Call ``func`` on ``data``, filling in its other required arguments."""
    params = inspect.signature(func).parameters
    kwargs = {}

    if "benchmark" in params:
        kwargs["benchmark"] = np.array([])
    if "risk_free" in params:
        if params["risk_free"].default is inspect.Parameter.empty:
            kwargs["risk_free"] = 0.0

    with warnings.catch_warnings():
        # an empty input is a documented NaN, never a warning
        warnings.simplefilter("error")
        return func(data, **kwargs)


@pytest.mark.parametrize("case, make_input", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("func", REDUCER_FUNCS, ids=REDUCER_IDS)
def test_reducer_returns_nan_on_empty_input(func, case, make_input):
    out = _call(func, make_input())

    if case == "numpy-2d":
        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)  # one value per column
    elif case == "pandas-2d":
        assert isinstance(out, pd.Series)
        assert list(out.index) == ["a", "b"]  # indexed by the columns
    else:
        assert np.ndim(out) == 0  # a single number, not a container

    assert np.isnan(np.asarray(out, float)).all()


def test_every_public_reducer_is_covered():
    assert len(REDUCERS) == len(set(REDUCER_IDS))
    assert "sharpe_ratio" in REDUCER_IDS
    assert not any(name.startswith("_") for name in REDUCER_IDS)
