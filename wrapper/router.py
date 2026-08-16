import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AllocationRouter:
    """
    Capital Allocation Router.
    Routes capital between Tactical Rebalancing (reallocating existing capital)
    and SIP Routing (deploying new cash inflows to drift towards target weights).
    """

    def route_tactical_rebalance(
        self,
        current_holdings: pd.DataFrame,
        target_weights: Dict[str, float],
        min_trade_amount: float = 1000.0
    ) -> Dict[str, Any]:
        """
        Generates buy/sell trade instructions to align portfolio with target weights.
        
        Parameters:
            current_holdings: DataFrame with columns ['Ticker', 'Asset_Class', 'Market_Value', 'Current_Price']
            target_weights: Dict of target weight fractions per asset class or ticker
            min_trade_amount: Minimum trade value threshold in INR
            
        Returns:
            Dict containing trade action recommendations, target allocation values,
            and estimated transaction cost.
        """
        total_market_value = float(current_holdings["Market_Value"].sum())
        
        trade_actions = []

        # Map target weights by Asset_Class or Ticker
        for idx, row in current_holdings.iterrows():
            ticker = row["Ticker"]
            asset_class = row["Asset_Class"]
            curr_val = float(row["Market_Value"])
            curr_price = float(row["Current_Price"]) if float(row["Current_Price"]) > 0 else 1.0

            # Find matching target weight
            target_w = target_weights.get(ticker, target_weights.get(asset_class, 0.0))
            
            # If target weight given at Asset_Class level, distribute among constituents
            ac_total_val = current_holdings[current_holdings["Asset_Class"] == asset_class]["Market_Value"].sum()
            if ticker not in target_weights and asset_class in target_weights and ac_total_val > 0:
                const_share = curr_val / ac_total_val
                target_w = target_weights[asset_class] * const_share

            target_val = total_market_value * target_w
            diff_val = target_val - curr_val

            if abs(diff_val) >= min_trade_amount:
                action = "BUY" if diff_val > 0 else "SELL"
                trade_units = int(abs(diff_val) / curr_price)
                actual_trade_val = trade_units * curr_price
                
                trade_actions.append({
                    "Ticker": ticker,
                    "Asset_Class": asset_class,
                    "Action": action,
                    "Current_Weight_%": round((curr_val / total_market_value) * 100, 2),
                    "Target_Weight_%": round(target_w * 100, 2),
                    "Trade_Amount_INR": round(actual_trade_val, 2),
                    "Trade_Units": trade_units,
                    "Current_Price_INR": round(curr_price, 2)
                })

        df_trades = pd.DataFrame(trade_actions)

        return {
            "mode": "Tactical Rebalance",
            "total_portfolio_value": total_market_value,
            "trade_recommendations": df_trades,
            "trade_count": len(df_trades)
        }

    def route_sip_inflow(
        self,
        current_holdings: pd.DataFrame,
        target_weights: Dict[str, float],
        sip_amount: float
    ) -> Dict[str, Any]:
        """
        Routes fresh SIP cash inflow across underweight assets without selling existing holdings.
        
        Parameters:
            current_holdings: DataFrame with columns ['Ticker', 'Asset_Class', 'Market_Value', 'Current_Price']
            target_weights: Dict of target weight fractions
            sip_amount: Cash inflow amount in INR (e.g. 50,000 INR)
            
        Returns:
            Dict with SIP allocation recommendations per asset.
        """
        total_curr_val = float(current_holdings["Market_Value"].sum())
        new_total_val = total_curr_val + sip_amount

        df = current_holdings.copy()
        
        # Calculate target value after SIP
        df["Target_Weight"] = df.apply(lambda r: target_weights.get(r["Ticker"], target_weights.get(r["Asset_Class"], 0.25)), axis=1)
        df["Target_Value_New"] = new_total_val * df["Target_Weight"]
        df["Shortfall_INR"] = np.maximum(0.0, df["Target_Value_New"] - df["Market_Value"])

        total_shortfall = df["Shortfall_INR"].sum()

        if total_shortfall > 0:
            df["SIP_Alloc_INR"] = (df["Shortfall_INR"] / total_shortfall) * sip_amount
        else:
            df["SIP_Alloc_INR"] = df["Target_Weight"] * sip_amount

        sip_actions = []
        for idx, row in df.iterrows():
            alloc_inr = float(row["SIP_Alloc_INR"])
            curr_price = float(row["Current_Price"]) if float(row["Current_Price"]) > 0 else 1.0
            units = int(alloc_inr / curr_price)
            actual_inr = units * curr_price

            if actual_inr > 0:
                sip_actions.append({
                    "Ticker": row["Ticker"],
                    "Asset_Class": row["Asset_Class"],
                    "SIP_Allocation_INR": round(actual_inr, 2),
                    "Recommended_Units": units,
                    "Current_Price_INR": round(curr_price, 2),
                    "Post_SIP_Weight_%": round(((row["Market_Value"] + actual_inr) / new_total_val) * 100, 2)
                })

        df_sip = pd.DataFrame(sip_actions)

        return {
            "mode": "SIP Cash Inflow",
            "sip_amount": sip_amount,
            "new_portfolio_value": new_total_val,
            "sip_recommendations": df_sip
        }
