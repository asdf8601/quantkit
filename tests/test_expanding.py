import quantkit as qnt
import numpy as np
import pandas as pd


def test_drawdown_numpy():
    p = np.array([7, 3, 1, 5])
    
    expected = np.array([ 0, -4, -6, -2])
    obtained = qnt.expanding.drawdown(p, relative=False)
    
    np.testing.assert_almost_equal(obtained, expected)


def test_drawdown_pandas():
    p = pd.DataFrame(np.array([7, 3, 1, 5]))
    
    expected = pd.DataFrame(np.array([ 0, -4, -6, -2]), dtype=np.float64)
    obtained = qnt.expanding.drawdown(p, relative=False)
    
    pd.testing.assert_frame_equal(obtained, expected)


def test_drawdown_nan_numpy():
    p = np.array([7, np.nan, 1, 5])

    expected = np.array([ 0., np.nan, -6., -2.], dtype=np.float64)
    obtained = qnt.expanding.drawdown(p, relative=False)

    np.testing.assert_almost_equal(obtained, expected)


def test_drawdown_nan_pandas():
    p = pd.DataFrame(np.array([7, np.nan, 1, 5]))

    expected = pd.DataFrame(np.array([ 0, np.nan, -6, -2]), dtype=np.float64)
    obtained = qnt.expanding.drawdown(p, relative=False)
    
    pd.testing.assert_frame_equal(obtained, expected)


def test_drawdown_numpy_relative():
    p = np.array([7, 3, 1, 5])
    
    expected = np.array([ 0., -0.57142857, -0.85714286, -0.28571429])
    obtained = qnt.expanding.drawdown(p, relative=True)

    np.testing.assert_almost_equal(obtained, expected)


def test_drawdown_pandas_relative():
    p = pd.DataFrame(np.array([7, 3, 1, 5]))
    
    expected = pd.DataFrame(np.array([ 0., -0.57142857, -0.85714286, -0.28571429]))
    obtained = qnt.expanding.drawdown(p, relative=True)

    pd.testing.assert_frame_equal(obtained, expected)