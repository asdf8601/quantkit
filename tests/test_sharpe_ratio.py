"""Sharpe ratio: mean excess return per unit of its standard deviation.

The reduction is column-wise, like every other reducer of
:mod:`quantkit.stats`, and the deviation is the population one (``ddof=0``).
Each test doubles as the specification: the expected value is worked out by
hand in a comment next to the data.
"""

import warnings

import numpy as np
import pandas as pd

from quantkit import stats
from quantkit.conventions import BYEAR

# Three observations, two columns. The ratio does not change if a column is
# scaled, so the numbers below are the tenth of [0, 0, 3] and [0, 2, 4].
#
#   col a : 0.0, 0.0, 0.3 -> mean 0.1, deviations -0.1, -0.1, 0.2
#           var = (0.01 + 0.01 + 0.04) / 3 = 0.02, sd = 0.1414214
#           sharpe = 0.1 / 0.1414214 = 0.7071068  (= 1 / sqrt(2))
#   col b : 0.0, 0.2, 0.4 -> mean 0.2, deviations -0.2, 0.0, 0.2
#           var = (0.04 + 0.00 + 0.04) / 3 = 0.0266667, sd = 0.1632993
#           sharpe = 0.2 / 0.1632993 = 1.2247449  (= sqrt(3 / 2))
RETS_2D = np.array(
    [
        [0.0, 0.0],
        [0.0, 0.2],
        [0.3, 0.4],
    ]
)
EXPECTED_2D = [1 / np.sqrt(2), np.sqrt(1.5)]

COLUMNS = ["a", "b"]
INDEX = ["mon", "tue", "wed"]


def test_sharpe_ratio_of_a_1d_series_is_a_single_number():
    # [0, 2] minus [0, 0]: mean 1, population sd 1 -> exactly 1
    obtained = stats.sharpe_ratio(np.array([0, 2]), np.array([0, 0]), factor=1)
    assert np.ndim(obtained) == 0
    np.testing.assert_almost_equal(obtained, 1.0)


def test_sharpe_ratio_reduces_a_2d_array_to_one_value_per_column():
    obtained = stats.sharpe_ratio(RETS_2D, 0.0, factor=1)

    assert obtained.shape == (2,)  # one per column, not one per row
    np.testing.assert_almost_equal(obtained, EXPECTED_2D)


def test_sharpe_ratio_of_a_column_equals_the_ratio_of_that_column_alone():
    obtained = stats.sharpe_ratio(RETS_2D, 0.0, factor=1)

    for position, column in enumerate(RETS_2D.T):
        alone = stats.sharpe_ratio(column, 0.0, factor=1)
        np.testing.assert_almost_equal(obtained[position], alone)


def test_sharpe_ratio_of_a_dataframe_is_a_series_indexed_by_the_columns():
    rets = pd.DataFrame(RETS_2D, index=INDEX, columns=COLUMNS)

    obtained = stats.sharpe_ratio(rets, 0.0, factor=1)
    expected = pd.Series(EXPECTED_2D, index=COLUMNS)

    pd.testing.assert_series_equal(obtained, expected)


def test_sharpe_ratio_subtracts_a_scalar_risk_free_from_every_period():
    # excess a : -0.05, -0.05, 0.25 -> mean 0.05, sd unchanged 0.1414214
    #            0.05 / 0.1414214 = 0.3535534
    # excess b : -0.05, 0.15, 0.35 -> mean 0.15, sd unchanged 0.1632993
    #            0.15 / 0.1632993 = 0.9185587
    obtained = stats.sharpe_ratio(RETS_2D, 0.05, factor=1)
    np.testing.assert_almost_equal(obtained, [0.3535534, 0.9185587])


def test_sharpe_ratio_subtracts_a_per_period_risk_free_row_by_row():
    # rf 0.0, 0.1, 0.1 is subtracted row by row, not column by column
    #   excess a : 0.0, -0.1, 0.2 -> mean 0.0333333
    #              var = (0.0011111 + 0.0177778 + 0.0277778) / 3 = 0.0155556
    #              sd = 0.1247219 -> 0.0333333 / 0.1247219 = 0.2672612
    #   excess b : 0.0, 0.1, 0.3 -> mean 0.1333333, same deviations, same sd
    #              0.1333333 / 0.1247219 = 1.0690450
    risk_free = np.array([0.0, 0.1, 0.1])

    obtained = stats.sharpe_ratio(RETS_2D, risk_free, factor=1)
    np.testing.assert_almost_equal(obtained, [0.2672612, 1.0690450])


def test_sharpe_ratio_of_a_1d_series_takes_a_per_period_risk_free():
    obtained = stats.sharpe_ratio(
        RETS_2D[:, 0], np.array([0.0, 0.1, 0.1]), factor=1
    )
    np.testing.assert_almost_equal(obtained, 0.2672612)


def test_sharpe_ratio_aligns_a_series_risk_free_by_row_index():
    rets = pd.DataFrame(RETS_2D, index=INDEX, columns=COLUMNS)
    # same rates as the row-by-row test, in another order: the alignment is
    # by row label, so the shuffling must not change the result
    risk_free = pd.Series([0.1, 0.0, 0.1], index=["tue", "mon", "wed"])

    obtained = stats.sharpe_ratio(rets, risk_free, factor=1)
    expected = pd.Series([0.2672612, 1.0690450], index=COLUMNS)

    pd.testing.assert_series_equal(obtained, expected)


def test_sharpe_ratio_of_a_constant_excess_return_is_nan_not_inf():
    # every excess return is 0.1: the deviation is zero and the ratio is
    # undefined, which is NaN by convention
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        obtained = stats.sharpe_ratio(np.array([0.1, 0.1, 0.1]), 0.0)

    assert np.isnan(obtained)


def test_sharpe_ratio_ignores_a_nan_only_in_its_own_column():
    # column a keeps 0.0, 0.0, 0.3 and column b keeps 0.0, 0.2, 0.4, so both
    # columns give the values of RETS_2D
    rets = np.array(
        [
            [0.0, 0.0],
            [np.nan, 0.2],
            [0.0, 0.4],
            [0.3, np.nan],
        ]
    )

    obtained = stats.sharpe_ratio(rets, 0.0, factor=1)
    np.testing.assert_almost_equal(obtained, EXPECTED_2D)


def test_sharpe_ratio_of_an_all_nan_column_is_nan_without_warning():
    rets = np.column_stack([np.full(3, np.nan), RETS_2D[:, 1]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        obtained = stats.sharpe_ratio(rets, 0.0, factor=1)

    assert np.isnan(obtained[0])
    np.testing.assert_almost_equal(obtained[1], EXPECTED_2D[1])


def test_sharpe_ratio_factor_multiplies_the_result():
    obtained = stats.sharpe_ratio(RETS_2D, 0.0, factor=2)
    np.testing.assert_almost_equal(obtained, np.multiply(EXPECTED_2D, 2))


def test_sharpe_ratio_annualizes_with_the_square_root_of_a_year_by_default():
    obtained = stats.sharpe_ratio(RETS_2D, 0.0)
    expected = np.multiply(EXPECTED_2D, np.sqrt(BYEAR))
    np.testing.assert_almost_equal(obtained, expected)
