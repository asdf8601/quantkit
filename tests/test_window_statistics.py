"""Public contracts for rolling and expanding statistics.

The scalar reducers are deliberately used only as the oracle for the
corresponding prefix or trailing interval.  The tests below also keep a few
hand-worked cases for window boundaries and path-dependent statistics, where
calling the same reducer would hide an indexing error.
"""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

import quantkit as qnt


WINDOW = 3
INDEX = pd.date_range("2024-01-02", periods=4, freq="B")
RETURNS = pd.Series([0.10, -0.20, 0.05, -0.10], index=INDEX, name="fund")
BENCHMARK = pd.Series([0.05, -0.10, 0.03, -0.06], index=INDEX, name="market")
PRICES = pd.Series([100.0, 110.0, 88.0, 92.4], index=INDEX, name="fund")

# The module exposes exactly the scalar public statistics as window functions.
PRICE_STATS = {
    "total_returns",
    "drawdown",
    "max_drawdown",
    "max_drawup",
    "max_drawdown_peak",
    "max_drawdown_valley",
    "max_drawdown_recovery",
    "max_drawdown_duration",
    "max_drawdown_recovery_duration",
    "longest_drawdown_duration",
    "average_drawdown",
}
PAIRWISE_STATS = {
    "alpha",
    "batting_average",
    "bear_beta",
    "beta",
    "bull_beta",
    "correlation",
    "down_capture",
    "information_ratio",
    "overall_capture",
    "r_squared",
    "tracking_error",
    "treynor_ratio",
    "up_capture",
}
LABEL_STATS = {
    "max_drawdown_peak",
    "max_drawdown_valley",
    "max_drawdown_recovery",
}
STAT_NAMES = {
    name
    for name, function in inspect.getmembers(qnt.stats, inspect.isfunction)
    if not name.startswith("_") and function.__module__ == qnt.stats.__name__
}


def _input_for(name):
    return PRICES if name in PRICE_STATS else RETURNS


def _scalar(name, values, benchmark=BENCHMARK):
    function = getattr(qnt.stats, name)
    # A one-observation volatility has the scalar API's documented NaN value,
    # although NumPy warns while computing its sample deviation.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if name in PAIRWISE_STATS:
            return function(values, benchmark)
        if name == "sharpe_ratio":
            return function(values, 0.0, factor=1.0)
        return function(values)


def _assert_values_equal(obtained, expected):
    """Compare values while accepting scalar reducers' NaN convention."""
    if pd.isna(expected):
        assert pd.isna(obtained)
    elif isinstance(expected, pd.Timestamp):
        assert obtained == expected
    else:
        assert obtained == pytest.approx(expected, nan_ok=True)


@pytest.mark.parametrize("name", sorted(STAT_NAMES))
def test_expanding_matches_the_scalar_statistic_on_each_prefix(name):
    """Every expanding value uses data available at that timestamp only."""
    data = _input_for(name)
    function = getattr(qnt.expanding, name)
    kwargs = {"min_periods": 1}
    if name == "sharpe_ratio":
        obtained = function(data, 0.0, factor=1.0, **kwargs)
    elif name in PAIRWISE_STATS:
        obtained = function(data, BENCHMARK, **kwargs)
    else:
        obtained = function(data, **kwargs)

    assert isinstance(obtained, pd.Series)
    pd.testing.assert_index_equal(obtained.index, data.index)
    assert obtained.name == data.name

    for stop, label in enumerate(data.index, start=1):
        expected = _scalar(name, data.iloc[:stop])
        _assert_values_equal(obtained.loc[label], expected)


@pytest.mark.parametrize("name", sorted(STAT_NAMES))
def test_rolling_matches_the_scalar_statistic_on_each_trailing_window(name):
    """A count window is the current row and its preceding ``WINDOW - 1`` rows."""
    data = _input_for(name)
    function = getattr(qnt.rolling, name)
    kwargs = {"window": WINDOW, "min_periods": WINDOW}
    if name == "sharpe_ratio":
        obtained = function(data, 0.0, factor=1.0, **kwargs)
    elif name in PAIRWISE_STATS:
        obtained = function(data, BENCHMARK, **kwargs)
    else:
        obtained = function(data, **kwargs)

    assert isinstance(obtained, pd.Series)
    pd.testing.assert_index_equal(obtained.index, data.index)
    assert obtained.name == data.name
    assert obtained.iloc[: WINDOW - 1].isna().all()

    for stop in range(WINDOW, len(data) + 1):
        expected = _scalar(name, data.iloc[stop - WINDOW : stop])
        _assert_values_equal(obtained.iloc[stop - 1], expected)


def test_window_names_are_available_from_the_public_package():
    assert qnt.rolling is not None
    for name in STAT_NAMES | {"drawup", "expected_shortfall", "conditional_drawdown_at_risk"}:
        assert callable(getattr(qnt.rolling, name))
        assert callable(getattr(qnt.expanding, name))


def test_rolling_uses_the_right_closed_time_interval_without_future_data():
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.Series([100.0, 110.0, 88.0, 92.4], index=index)

    obtained = qnt.rolling.total_returns(
        prices, window="2D", min_periods=1
    )

    # (t - 2 days, t]: at 3 Jan this is 2--3 Jan, not 1--3 Jan.
    expected = pd.Series([np.nan, 0.10, -0.20, 0.05], index=index)
    pd.testing.assert_series_equal(obtained, expected)


def test_pairwise_windows_keep_primary_index_and_align_a_reordered_benchmark():
    index = pd.date_range("2024-02-01", periods=4, freq="D")
    returns = pd.Series([0.02, 0.08, -0.04, 0.03], index=index, name="fund")
    benchmark = pd.Series(
        [0.01, -0.02, 0.04, 0.30],
        index=[index[3], index[2], index[1], index[-1] + pd.Timedelta(days=1)],
        name="market",
    )

    obtained = qnt.rolling.beta(
        returns, benchmark, window=3, min_periods=1
    )

    assert obtained.name == returns.name
    pd.testing.assert_index_equal(obtained.index, returns.index)
    for stop in range(1, len(returns) + 1):
        expected = qnt.stats.beta(returns.iloc[max(0, stop - 3) : stop], benchmark)
        _assert_values_equal(obtained.iloc[stop - 1], expected)


def test_pairwise_windows_use_only_complete_pairs_in_each_interval():
    returns = pd.Series([0.01, np.nan, 0.03, 0.04], index=INDEX)
    benchmark = pd.Series([0.00, 0.01, np.nan, 0.02], index=INDEX)

    obtained = qnt.rolling.tracking_error(
        returns, benchmark, window=4, min_periods=1
    )

    # Only (0.01, 0.00) and (0.04, 0.02) are complete pairs.
    assert obtained.iloc[-1] == pytest.approx(np.std([0.01, 0.02], ddof=1))
    assert obtained.iloc[-1] == pytest.approx(
        qnt.stats.tracking_error(returns, benchmark)
    )


def test_alpha_and_beta_keep_their_linear_economic_relationship():
    benchmark = pd.Series([-0.02, 0.01, 0.03, -0.01], index=INDEX)
    returns = 2.0 * benchmark + 0.005

    rolling_beta = qnt.rolling.beta(
        returns, benchmark, window=3, min_periods=3
    )
    rolling_alpha = qnt.rolling.alpha(
        returns, benchmark, window=3, min_periods=3, factor=1.0
    )
    expanding_beta = qnt.expanding.beta(returns, benchmark)
    expanding_alpha = qnt.expanding.alpha(returns, benchmark, factor=1.0)

    assert rolling_beta.iloc[-1] == pytest.approx(2.0)
    assert rolling_alpha.iloc[-1] == pytest.approx(0.005)
    assert expanding_beta.iloc[-1] == pytest.approx(2.0)
    assert expanding_alpha.iloc[-1] == pytest.approx(0.005)


def test_cash_like_zero_returns_have_no_dispersion_or_sharpe_ratio():
    returns = pd.Series([0.0, 0.0, 0.0, 0.0], index=INDEX)

    volatility = qnt.rolling.volatility(returns, window=3, min_periods=3)
    sharpe = qnt.expanding.sharpe_ratio(returns, 0.0, factor=1.0)

    assert volatility.iloc[-1] == 0.0
    assert sharpe.isna().all()


def test_rolling_drawdown_and_drawup_restart_their_path_at_the_window_boundary():
    index = pd.date_range("2024-03-01", periods=4, freq="D")
    prices = pd.Series([100.0, 50.0, 60.0, 70.0], index=index)

    drawdown = qnt.rolling.drawdown(prices, window=3, min_periods=3)
    drawup = qnt.rolling.drawup(prices, window=3, min_periods=3)

    # The old high of 100 is outside the final [50, 60, 70] window.
    assert drawdown.iloc[-1] == 0.0
    # The current value is 20 above that same window's low of 50.
    assert drawup.iloc[-1] == pytest.approx(0.4)


@pytest.mark.parametrize("function_name", ["drawdown", "drawup"])
def test_expanding_path_statistics_preserve_the_current_nan_mask(function_name):
    prices = pd.Series([10.0, np.nan, 8.0, 12.0], index=INDEX)
    obtained = getattr(qnt.expanding, function_name)(prices, min_periods=1)

    assert pd.isna(obtained.iloc[1])
    assert not pd.isna(obtained.iloc[0])
    assert not pd.isna(obtained.iloc[2])
    assert not pd.isna(obtained.iloc[3])


@pytest.mark.parametrize("function_name", ["drawdown", "drawup"])
def test_rolling_path_statistics_preserve_the_current_nan_mask(function_name):
    prices = pd.Series([10.0, np.nan, 8.0, 12.0], index=INDEX)
    obtained = getattr(qnt.rolling, function_name)(
        prices, window=3, min_periods=1
    )

    assert pd.isna(obtained.iloc[1])
    assert not pd.isna(obtained.iloc[0])
    assert not pd.isna(obtained.iloc[2])
    assert not pd.isna(obtained.iloc[3])


def test_rolling_drawdown_labels_use_original_datetime_labels_and_numpy_positions():
    prices = np.array([10.0, 10.0, 6.0, 10.0, 9.0])
    dates = pd.date_range("2024-04-01", periods=len(prices), freq="D")
    series = pd.Series(prices, index=dates)

    for name, expected in {
        "max_drawdown_peak": (dates[1], 1.0),
        "max_drawdown_valley": (dates[2], 2.0),
        "max_drawdown_recovery": (dates[3], 3.0),
    }.items():
        pandas_result = getattr(qnt.rolling, name)(
            series, window=4, min_periods=4
        )
        numpy_result = getattr(qnt.rolling, name)(
            prices, window=4, min_periods=4
        )
        assert pandas_result.iloc[-1] == expected[0]
        assert numpy_result[-1] == expected[1]


def test_rolling_drawdown_labels_do_not_lose_large_integer_index_precision():
    labels = pd.Index([2**53 + value for value in range(1, 6)])
    prices = pd.Series([10.0, 10.0, 6.0, 10.0, 9.0], index=labels)

    obtained = qnt.rolling.max_drawdown_peak(
        prices, window=4, min_periods=4
    )

    assert pd.isna(obtained.iloc[0])
    assert obtained.iloc[-1] == labels[1]


@pytest.mark.parametrize(
    ("name", "label", "position"),
    [
        ("max_drawdown_peak", 1, 1.0),
        ("max_drawdown_valley", 2, 2.0),
        ("max_drawdown_recovery", 3, 3.0),
    ],
)
def test_rolling_drawdown_labels_keep_two_dimensional_axes_and_dtype(
    name, label, position
):
    values = np.array(
        [[10.0, 5.0], [10.0, 5.0], [6.0, 2.5], [10.0, 5.0], [9.0, 4.5]]
    )
    dates = pd.date_range("2024-04-01", periods=len(values), freq="D")
    frame = pd.DataFrame(values, index=dates, columns=["a", "b"])

    pandas_result = getattr(qnt.rolling, name)(
        frame, window=4, min_periods=4
    )
    numpy_result = getattr(qnt.rolling, name)(
        values, window=4, min_periods=4
    )

    assert isinstance(pandas_result, pd.DataFrame)
    pd.testing.assert_index_equal(pandas_result.index, frame.index)
    pd.testing.assert_index_equal(pandas_result.columns, frame.columns)
    assert pandas_result.iloc[-1].tolist() == [dates[label], dates[label]]
    np.testing.assert_equal(numpy_result[-1], [position, position])


def test_calmar_counts_a_loss_on_the_first_return_from_initial_capital():
    returns = pd.Series([-0.5, 3.0], index=INDEX[:2])

    rolling = qnt.rolling.calmar_ratio(
        returns, window=2, min_periods=2, periods_per_year=2
    )
    expanding = qnt.expanding.calmar_ratio(returns, periods_per_year=2)

    # Annualized return is 1.0 and initial capital to 0.5 is a 50% drawdown.
    assert rolling.iloc[-1] == pytest.approx(2.0)
    assert expanding.iloc[-1] == pytest.approx(2.0)


def test_expected_shortfall_propagates_a_nan_in_its_interval():
    returns = pd.Series([-0.10, np.nan, 0.20], index=INDEX[:3])

    rolling = qnt.rolling.expected_shortfall(
        returns, window=3, min_periods=3, confidence=0.5
    )
    expanding = qnt.expanding.expected_shortfall(returns, confidence=0.5)

    assert pd.isna(rolling.iloc[-1])
    assert pd.isna(expanding.iloc[-1])


def test_conditional_drawdown_at_risk_averages_the_drawdowns_at_the_boundary():
    prices = pd.Series([100.0, 80.0, 90.0, 70.0], index=INDEX)

    rolling = qnt.rolling.conditional_drawdown_at_risk(
        prices, window=4, min_periods=4, confidence=0.5
    )
    expanding = qnt.expanding.conditional_drawdown_at_risk(
        prices, confidence=0.5
    )

    # Drawdowns are 0, -0.2, -0.1, -0.3; the worst half loses 0.25.
    assert rolling.iloc[-1] == pytest.approx(0.25)
    assert expanding.iloc[-1] == pytest.approx(0.25)
