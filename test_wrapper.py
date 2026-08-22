"""Comprehensive Verification Test Suite for wrapper/parser.py and wrapper/router.py.
"""

import io
import os
import sys
import numpy as np
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from wrapper.parser import PortfolioParser
from wrapper.router import AllocationRouter


def test_parser_basic_and_ticker_standardization():
    print("\n--- TEST 1: PortfolioParser Basic & Ticker Standardization ---")
    parser = PortfolioParser()

    # 1. Ticker standardization checks
    assert PortfolioParser.standardize_ticker("RELIANCE") == "RELIANCE.NS"
    assert PortfolioParser.standardize_ticker("tcs") == "TCS.NS"
    assert PortfolioParser.standardize_ticker("INFY.NS") == "INFY.NS"
    assert PortfolioParser.standardize_ticker("HDFCBANK.BO") == "HDFCBANK.BO"
    assert PortfolioParser.standardize_ticker("^NSEI") == "^NSEI"
    assert PortfolioParser.standardize_ticker("CASH") == "CASH"
    print("[PASS] Ticker standardization tests passed.")

    # 2. Parse CSV data with missing Current_Price, using latest_prices lookup
    csv_data = """Ticker,Quantity,Buy_Price
RELIANCE,10,2500.0
TCS,5,3500.0
INFY,20,1600.0
"""
    latest_prices = {
        "RELIANCE.NS": 3000.0,  # in profit
        "TCS.NS": 3200.0,       # in loss
        "INFY.NS": 1600.0,      # breakeven
    }

    result = parser.parse(io.StringIO(csv_data), latest_prices=latest_prices)
    df = result["holdings_df"]
    summary = result["summary"]

    assert list(df["Ticker"]) == ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    
    # Check calculated Current_Values
    # RELIANCE: 10 * 3000 = 30,000; cost = 25,000; PnL = +5,000; PnL_Pct = 20%
    rel_row = df[df["Ticker"] == "RELIANCE.NS"].iloc[0]
    assert rel_row["Current_Value"] == 30000.0
    assert rel_row["PnL"] == 5000.0
    assert abs(rel_row["PnL_Pct"] - 20.0) < 1e-4
    assert rel_row["Position_Status"] == "GREEN"

    # TCS: 5 * 3200 = 16,000; cost = 17,500; PnL = -1,500; PnL_Pct = -8.57%
    tcs_row = df[df["Ticker"] == "TCS.NS"].iloc[0]
    assert tcs_row["Current_Value"] == 16000.0
    assert tcs_row["PnL"] == -1500.0
    assert tcs_row["Position_Status"] == "RED"

    # INFY: 20 * 1600 = 32,000; cost = 32,000; PnL = 0; PnL_Pct = 0%
    infy_row = df[df["Ticker"] == "INFY.NS"].iloc[0]
    assert infy_row["Current_Value"] == 32000.0
    assert infy_row["PnL"] == 0.0
    assert infy_row["Position_Status"] == "GREEN"

    total_market_val = 30000.0 + 16000.0 + 32000.0  # 78,000
    assert result["total_market_value"] == total_market_val
    assert summary["green_positions_count"] == 2
    assert summary["red_positions_count"] == 1

    # Check weights sum to 1.0
    assert abs(df["Current_Weight"].sum() - 1.0) < 1e-4
    print(f"[PASS] Basic portfolio parse & PnL metrics calculation passed (Total Value: Rs. {total_market_val:,.2f}).")


def test_parser_with_current_price_column_and_excel():
    print("\n--- TEST 2: PortfolioParser Direct Current_Price & Excel Buffer ---")
    parser = PortfolioParser()

    df_raw = pd.DataFrame([
        {"Ticker": "HDFCBANK", "Quantity": 50, "Buy_Price": 1400.0, "Current_Price": 1650.0},
        {"Ticker": "ICICIBANK", "Quantity": 40, "Buy_Price": 1100.0, "Current_Price": 1050.0},
        {"Ticker": "GOLDBEES", "Quantity": 1000, "Buy_Price": 55.0, "Current_Price": 65.0},
    ])

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df_raw.to_excel(writer, index=False)
    excel_buffer.seek(0)

    result = parser.parse(excel_buffer)
    df = result["holdings_df"]

    assert len(df) == 3
    assert "HDFCBANK.NS" in list(df["Ticker"])
    assert "GOLDBEES.NS" in list(df["Ticker"])
    assert df[df["Ticker"] == "GOLDBEES.NS"]["Asset_Class"].iloc[0] == "Gold"
    print("[PASS] Excel parsing and asset class classification passed.")


def test_router_sip_first_mode():
    print("\n--- TEST 3: AllocationRouter 'SIP_FIRST' Mode ---")
    router = AllocationRouter()

    # Create a portfolio:
    # Asset A: 50k value (50% weight)
    # Asset B: 50k value (50% weight)
    # Target: Asset A: 30%, Asset B: 70%
    # Cash Budget: 20,000 INR
    # Total Post Value: 120,000 INR
    # Target A = 36k, Target B = 84k
    # Asset A is overweight (50k > 36k) -> Shortfall = 0
    # Asset B is underweight (50k < 84k) -> Shortfall = 34k
    # In SIP_FIRST: ALL cash budget should go to Asset B to close gap, 0 sells on Asset A.

    portfolio = pd.DataFrame([
        {"Ticker": "ASSET_A.NS", "Current_Quantity": 50, "Current_Price": 1000.0, "Current_Value": 50000.0},
        {"Ticker": "ASSET_B.NS", "Current_Quantity": 50, "Current_Price": 1000.0, "Current_Value": 50000.0},
    ])

    targets = pd.Series({"ASSET_A.NS": 0.30, "ASSET_B.NS": 0.70})
    prices = pd.Series({"ASSET_A.NS": 1000.0, "ASSET_B.NS": 1000.0})
    cash_budget = 20000.0

    plan_df = router.generate_allocation_plan(
        current_portfolio=portfolio,
        target_weights=targets,
        cash_budget=cash_budget,
        latest_prices=prices,
        mode="SIP_FIRST",
    )

    print("SIP_FIRST Plan:")
    print(plan_df[["Ticker", "Current_Quantity", "Current_Weight", "Target_Weight", "Action", "Trade_Units", "Trade_Value_INR", "Post_Trade_Quantity", "Post_Trade_Weight"]])

    # Verify: ZERO SELL orders
    assert (plan_df["Action"] == "SELL").sum() == 0

    # Verify: ASSET_A is HOLD, ASSET_B is BUY 20 units
    row_a = plan_df[plan_df["Ticker"] == "ASSET_A.NS"].iloc[0]
    row_b = plan_df[plan_df["Ticker"] == "ASSET_B.NS"].iloc[0]

    assert row_a["Action"] == "HOLD"
    assert row_a["Trade_Units"] == 0
    assert row_b["Action"] == "BUY"
    assert row_b["Trade_Units"] == 20
    assert row_b["Trade_Value_INR"] == 20000.0
    assert row_b["Post_Trade_Quantity"] == 70

    # Verify residual cash
    assert plan_df.attrs["residual_cash"] == 0.0
    print("[PASS] SIP_FIRST mode passed perfectly (zero sells, targeted cash allocation).")


def test_router_tactical_mode_drift_exceeded():
    print("\n--- TEST 4: AllocationRouter 'TACTICAL' Mode (Drift > Threshold) ---")
    router = AllocationRouter()

    # Portfolio:
    # Asset A: 80,000 (80% weight) -> Target: 40% (Target Value = 40k) -> Drift = +40% (> 5%)
    # Asset B: 20,000 (20% weight) -> Target: 60% (Target Value = 60k) -> Drift = -40%
    # Cash Budget: 0 INR
    # Price A = 1000, Price B = 500
    # Tactical rebalance should SELL 40 units of A (40k) and BUY 80 units of B (40k).

    portfolio = pd.DataFrame([
        {"Ticker": "STOCK_A.NS", "Current_Quantity": 80, "Current_Price": 1000.0, "Current_Value": 80000.0},
        {"Ticker": "STOCK_B.NS", "Current_Quantity": 40, "Current_Price": 500.0, "Current_Value": 20000.0},
    ])

    targets = pd.Series({"STOCK_A.NS": 0.40, "STOCK_B.NS": 0.60})
    prices = pd.Series({"STOCK_A.NS": 1000.0, "STOCK_B.NS": 500.0})

    plan_df = router.generate_allocation_plan(
        current_portfolio=portfolio,
        target_weights=targets,
        cash_budget=0.0,
        latest_prices=prices,
        mode="TACTICAL",
        drift_threshold=0.05,
    )

    print("TACTICAL Plan (Drift > 5%):")
    print(plan_df[["Ticker", "Current_Quantity", "Current_Weight", "Target_Weight", "Action", "Trade_Units", "Trade_Value_INR", "Post_Trade_Quantity", "Post_Trade_Weight"]])

    row_a = plan_df[plan_df["Ticker"] == "STOCK_A.NS"].iloc[0]
    row_b = plan_df[plan_df["Ticker"] == "STOCK_B.NS"].iloc[0]

    assert row_a["Action"] == "SELL"
    assert row_a["Trade_Units"] == 40
    assert row_a["Post_Trade_Quantity"] == 40
    assert abs(row_a["Post_Trade_Weight"] - 0.40) < 1e-4

    assert row_b["Action"] == "BUY"
    assert row_b["Trade_Units"] == 80
    assert row_b["Post_Trade_Quantity"] == 120
    assert abs(row_b["Post_Trade_Weight"] - 0.60) < 1e-4

    assert plan_df.attrs["mode_executed"] == "TACTICAL"
    assert plan_df.attrs["residual_cash"] == 0.0
    print("[PASS] TACTICAL mode with drift > threshold passed perfectly.")


def test_router_tactical_mode_drift_within_threshold():
    print("\n--- TEST 5: AllocationRouter 'TACTICAL' Mode (Drift <= Threshold -> Auto SIP_FIRST) ---")
    router = AllocationRouter()

    # Portfolio:
    # Asset A: 52,000 (52%) -> Target: 50% -> Drift = +2% (< 5%)
    # Asset B: 48,000 (48%) -> Target: 50% -> Drift = -2%
    # Cash Budget: 10,000 INR
    # In TACTICAL mode with drift <= 5%, should automatically route via SIP_FIRST without selling Asset A!

    portfolio = pd.DataFrame([
        {"Ticker": "STOCK_A.NS", "Current_Quantity": 52, "Current_Price": 1000.0, "Current_Value": 52000.0},
        {"Ticker": "STOCK_B.NS", "Current_Quantity": 48, "Current_Price": 1000.0, "Current_Value": 48000.0},
    ])

    targets = pd.Series({"STOCK_A.NS": 0.50, "STOCK_B.NS": 0.50})
    prices = pd.Series({"STOCK_A.NS": 1000.0, "STOCK_B.NS": 1000.0})

    plan_df = router.generate_allocation_plan(
        current_portfolio=portfolio,
        target_weights=targets,
        cash_budget=10000.0,
        latest_prices=prices,
        mode="TACTICAL",
        drift_threshold=0.05,
    )

    print("TACTICAL Plan (Drift <= 5% -> Auto SIP_FIRST):")
    print(plan_df[["Ticker", "Current_Weight", "Target_Weight", "Action", "Trade_Units", "Trade_Value_INR"]])

    assert plan_df.attrs["mode_executed"] == "SIP_FIRST"
    assert (plan_df["Action"] == "SELL").sum() == 0  # No sells triggered
    print("[PASS] Automatic switch from TACTICAL to SIP_FIRST on small drift passed.")


def test_router_integer_flooring_and_min_trade_filter():
    print("\n--- TEST 6: Integer Share Flooring & Min Trade Filter ---")
    router = AllocationRouter()

    # Small cash budget of 1500 INR
    # Price = 800 INR -> 1500 / 800 = 1.875 -> Floored to 1 share (800 INR)
    # Trade value 800 < min_trade_inr (1000) -> Filtered to HOLD (0 units, residual cash = 1500)
    portfolio = pd.DataFrame([
        {"Ticker": "EXPENSIVE.NS", "Current_Quantity": 10, "Current_Price": 800.0, "Current_Value": 8000.0},
    ])
    targets = pd.Series({"EXPENSIVE.NS": 1.0})
    prices = pd.Series({"EXPENSIVE.NS": 800.0})

    plan_df = router.generate_allocation_plan(
        current_portfolio=portfolio,
        target_weights=targets,
        cash_budget=1500.0,
        latest_prices=prices,
        mode="SIP_FIRST",
        min_trade_inr=1000.0,
    )

    row = plan_df.iloc[0]
    assert row["Action"] == "HOLD"
    assert row["Trade_Units"] == 0
    assert plan_df.attrs["residual_cash"] == 1500.0
    print("[PASS] Micro-trade filtering and residual cash tracking verified.")


def test_backward_compatibility():
    print("\n--- TEST 7: Backward Compatibility with Existing Methods ---")
    parser = PortfolioParser()
    sample_res = parser._fallback_sample_portfolio()
    assert "total_market_value" in sample_res
    assert "holdings_df" in sample_res

    router = AllocationRouter()
    target_weights = {"^NSEI": 0.60, "GOLDBEES.NS": 0.20, "LIQUIDBEES.NS": 0.20}
    
    sip_res = router.route_sip_inflow(sample_res["holdings_df"], target_weights, sip_amount=50000.0)
    assert "sip_recommendations" in sip_res
    assert sip_res["mode"] == "SIP Cash Inflow"

    tac_res = router.route_tactical_rebalance(sample_res["holdings_df"], target_weights)
    assert "trade_recommendations" in tac_res
    assert tac_res["mode"] == "Tactical Rebalance"
    print("[PASS] Backward compatibility wrapper methods verified.")


if __name__ == "__main__":
    test_parser_basic_and_ticker_standardization()
    test_parser_with_current_price_column_and_excel()
    test_router_sip_first_mode()
    test_router_tactical_mode_drift_exceeded()
    test_router_tactical_mode_drift_within_threshold()
    test_router_integer_flooring_and_min_trade_filter()
    test_backward_compatibility()
    print("\n==================================================")
    print("  ALL 7 TESTS PASSED ACCURATELY AND COMPLETELY!   ")
    print("==================================================")
