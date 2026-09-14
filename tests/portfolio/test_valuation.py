"""Tests for portfolio valuation calculations."""

import numpy as np
import pandas as pd
import pytest

from quantkit.portfolio import valuation


def test_position_values_aligns_labels_and_broadcasts_prices():
    """Position valuation retains quantity labels after price alignment."""
    quantities = pd.DataFrame(
        [[2.0, -1.0], [3.0, -2.0]],
        index=["t1", "t2"],
        columns=["stock", "hedge"],
    )
    prices = pd.Series([20.0, 10.0], index=["hedge", "stock"])

    result = valuation.position_values(quantities, prices)

    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame(
            [[20.0, -20.0], [30.0, -40.0]],
            index=quantities.index,
            columns=quantities.columns,
        ),
    )


def test_position_values_rejects_negative_prices_and_preserves_nan():
    """Prices cannot be negative, while unknown values remain unknown."""
    with pytest.raises(ValueError, match="nonnegative"):
        valuation.position_values(np.array([1.0]), np.array([-1.0]))

    result = valuation.position_values(
        np.array([0.0, 2.0]), np.array([np.nan, 3.0])
    )
    np.testing.assert_allclose(result, [np.nan, 6.0], equal_nan=True)


def test_gross_and_net_asset_value_match_balance_sheet_fixture():
    """Long, short, cash, and liability values follow the v1 formulas."""
    values = np.array([100.0, -40.0])

    assert valuation.gross_asset_value(values, cash=40.0) == 140.0
    assert valuation.net_asset_value(values, cash=40.0) == 100.0
    assert valuation.gross_asset_value(np.array([100.0]), cash=-20.0) == 100.0
    assert valuation.net_asset_value(
        np.array([100.0]), cash=-20.0, liabilities=5.0
    ) == 75.0


def test_portfolio_values_preserve_time_labels_and_missing_values():
    """Time amounts align by labels and missing positions propagate."""
    values = pd.DataFrame(
        [[100.0, -40.0], [np.nan, 10.0]],
        index=["t1", "t2"],
        columns=["long", "short"],
    )
    cash = pd.Series([40.0, 20.0], index=["t2", "t1"])

    gross = valuation.gross_asset_value(values, cash)
    net = valuation.net_asset_value(values, cash)

    pd.testing.assert_series_equal(
        gross, pd.Series([120.0, np.nan], index=values.index)
    )
    pd.testing.assert_series_equal(
        net, pd.Series([80.0, np.nan], index=values.index)
    )


def test_net_asset_value_rejects_negative_additional_liabilities():
    """Additional liabilities cannot be used to add assets."""
    with pytest.raises(ValueError, match="nonnegative"):
        valuation.net_asset_value(np.array([10.0]), liabilities=-1.0)


def test_nav_per_share_aligns_series_and_keeps_name():
    """Share Series align to NAV labels while NAV metadata is retained."""
    nav = pd.Series([100.0, -20.0, np.nan], index=["t1", "t2", "t3"], name="NAV")
    shares = pd.Series([10.0, 20.0, 5.0], index=["t2", "t1", "t3"])

    result = valuation.nav_per_share(nav, shares)

    pd.testing.assert_series_equal(
        result,
        pd.Series([5.0, -2.0, np.nan], index=nav.index, name="NAV"),
    )


@pytest.mark.parametrize("shares", [0.0, -1.0, np.inf])
def test_nav_per_share_rejects_invalid_shares(shares):
    """Nonpositive and infinite share counts are invalid."""
    with pytest.raises(ValueError):
        valuation.nav_per_share(100.0, shares)


def test_nav_per_share_propagates_missing_shares():
    """Missing share observations remain missing in the per-share value."""
    result = valuation.nav_per_share(
        np.array([100.0, 200.0]), np.array([10.0, np.nan])
    )

    np.testing.assert_allclose(result, [10.0, np.nan], equal_nan=True)


def test_nav_per_share_rejects_unmatched_series_labels_and_shapes():
    """NAV and shares need matching labels or positional shape."""
    nav = pd.Series([100.0, 200.0], index=["t1", "t2"])

    with pytest.raises(ValueError, match="labels"):
        valuation.nav_per_share(nav, pd.Series([10.0, 20.0], index=["t1", "t3"]))
    with pytest.raises(ValueError, match="shape"):
        valuation.nav_per_share(np.array([100.0, 200.0]), np.array([10.0]))
