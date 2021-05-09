import quantkit as qnt
import numpy as np
import pandas as pd


def test_returns_numpy():
    p = np.array([1, 2, 3])

    obtained = qnt.core.returns(p, 1)
    expected = np.array([np.nan, 1., 0.5])

    np.testing.assert_almost_equal(obtained, expected)


def test_returns_pandas():
    p = pd.DataFrame(np.array([1, 2, 3]))

    obtained = qnt.core.returns(p, 1)
    expected = pd.DataFrame(np.array([np.nan, 1., 0.5]))

    pd.testing.assert_frame_equal(obtained, expected)


def test_cum_returns_numpy():
    p = np.array([np.nan, 1, 0.5])

    obtained = qnt.core.cum_returns(p, 1)
    expected = np.array([1., 2, 3])

    np.testing.assert_almost_equal(obtained, expected)


def test_rebase_numpy():
    p = np.array([1., 2, 3])

    obtained = qnt.core.rebase(p)
    expected = np.array([100., 200, 300])

    np.testing.assert_almost_equal(obtained, expected)


def test_rebase_numpy2d():
    x = np.array([1., 2, 3]).reshape(-1, 1)
    p = np.concatenate([x, x], axis=1)

    obtained = qnt.core.rebase(p)
    expected = np.array([[100., 200, 300], [100., 200, 300]]).T

    np.testing.assert_almost_equal(obtained, expected)
