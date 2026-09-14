"""Execute the optional examples against their local market snapshot."""

import importlib.util
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("marimo", reason="Install the examples dependency group")
pytest.importorskip("plotly", reason="Install the examples dependency group")

NOTEBOOKS = Path(__file__).resolve().parents[1] / "examples" / "notebooks"


@pytest.fixture
def run_notebook(monkeypatch):
    monkeypatch.syspath_prepend(str(NOTEBOOKS))

    def no_network(*args, **kwargs):
        raise AssertionError("Notebooks must run using only the local data")

    monkeypatch.setattr(socket.socket, "connect", no_network)

    def run(name, overrides=None):
        spec = importlib.util.spec_from_file_location(
            name, NOTEBOOKS / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        outputs, definitions = module.app.run(defs=overrides)
        assert outputs
        return definitions

    return run


@pytest.mark.parametrize("selection", [None, ("GLD", [2022, 2022])])
def test_quickstart_compounds_observed_prices(run_notebook, selection):
    overrides = None
    if selection is not None:
        overrides = {
            "ticker": SimpleNamespace(value=selection[0]),
            "years": SimpleNamespace(value=selection[1]),
        }
    definitions = run_notebook("01_quickstart", overrides)
    prices = definitions["prices"]
    returns = definitions["asset_returns"]
    wealth = definitions["wealth"]
    rebased = definitions["rebased_prices"]
    assert rebased.index.equals(prices.index)
    np.testing.assert_allclose(rebased, wealth)
    np.testing.assert_allclose(rebased, 100 * prices / prices.iloc[0])
    assert np.isfinite(returns).all()
    assert wealth.iloc[0] == pytest.approx(100)
    assert wealth.iloc[-1] / wealth.iloc[0] == pytest.approx(
        prices.iloc[-1] / prices.iloc[0]
    )
    assert (1 + returns).prod() == pytest.approx(
        prices.iloc[-1] / prices.iloc[0]
    )


def test_advanced_runs_offline(run_notebook):
    definitions = run_notebook("02_advanced")
    assert np.isfinite(definitions["asset_returns"].to_numpy()).all()
    drawdowns = definitions["window_drawdowns"]
    expanding = drawdowns["Expanding maximum drawdown"]
    assert (expanding.diff().dropna() <= 0).all()
    assert expanding.iloc[-1] == pytest.approx(
        definitions["results"]["max_drawdown"]
    )
    rolling = drawdowns.dropna()
    assert (
        rolling["Rolling maximum drawdown"]
        <= rolling["Rolling drawdown"] + 1e-14
    ).all()
    assert not definitions["window_ratios"].dropna().empty


def test_advanced_identical_benchmark_and_short_window(run_notebook):
    definitions = run_notebook(
        "02_advanced",
        {
            "asset_control": SimpleNamespace(value="TLT"),
            "benchmark_control": SimpleNamespace(value="TLT"),
            "years_control": SimpleNamespace(value=[2022, 2022]),
            "window_control": SimpleNamespace(value=252),
            "confidence_control": SimpleNamespace(value=0.99),
        },
    )
    results = definitions["results"]
    assert results["beta"] == pytest.approx(1)
    assert results["correlation"] == pytest.approx(1)
    assert results["tracking_error"] == pytest.approx(0)
    assert np.isnan(results["information_ratio"])
    assert definitions["rolling_volatility"].isna().all()


def test_portfolio_valuation_and_risk_reconcile(run_notebook):
    definitions = run_notebook("03_portfolio")
    weights = definitions["portfolio_weights"]
    covariance = definitions["covariance"].loc[weights.index, weights.index]
    volatility = definitions["portfolio_volatility"]
    assert definitions["nav"] > 0
    assert definitions["nav"] == pytest.approx(
        definitions["position_values"].sum() + 10_000
    )
    assert definitions["risk_contributions"].sum() == pytest.approx(volatility)
    assert volatility**2 == pytest.approx(
        weights.to_numpy() @ covariance.to_numpy() @ weights.to_numpy()
    )
    returns = definitions["buy_hold_returns"]
    wealth = definitions["buy_hold_wealth"]
    assert (1 + returns).prod() == pytest.approx(
        wealth.iloc[-1] / wealth.iloc[0]
    )
    np.testing.assert_allclose(
        wealth, definitions["buy_hold_sleeves"].sum(axis=1)
    )
    example = definitions["flow_example"]
    assert example["twr"] == pytest.approx((1 + returns.iloc[:2]).prod() - 1)
    np.testing.assert_allclose(
        definitions["cash_example_nav"],
        [10000, 15000, 15000, 15900, 16900, 14900],
    )
    np.testing.assert_allclose(
        definitions["cash_example_returns"], [0, 0, 0.06, 0, 0]
    )
    assert definitions["cash_example_pnl"].sum() == pytest.approx(900)
    assert definitions["cash_example_twr"] == pytest.approx(0.06)


def test_portfolio_cash_only_has_zero_returns(run_notebook):
    definitions = run_notebook(
        "03_portfolio",
        {
            "controls": SimpleNamespace(
                value={
                    "Period": [2022, 2022],
                    "Date": "2022-12-30",
                    "SPY": 0,
                    "QQQ": 0,
                    "TLT": 0,
                    "GLD": 0,
                }
            )
        },
    )
    assert definitions["nav"] == pytest.approx(10_000)
    assert definitions["portfolio_volatility"] == pytest.approx(0)
    assert definitions["risk_contributions"].isna().all()
    np.testing.assert_allclose(definitions["buy_hold_returns"], 0)
    np.testing.assert_allclose(definitions["rebalanced_returns"], 0)
    assert definitions["flow_example"]["twr"] == pytest.approx(0)
