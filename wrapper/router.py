"""Capital Allocation Router Module for Indian Equity Portfolios.

This module provides the `AllocationRouter` class, implementing capital deployment
and rebalancing strategies. It supports two primary operational modes:
1. 'SIP_FIRST': Directs fresh cash inflows proportionally to under-allocated assets
   to close weight gaps without triggering any asset sales.
2. 'TACTICAL': Evaluates portfolio drift against a threshold (default 5%). If drift
   exceeds threshold, generates optimal SELL and BUY orders to achieve target weights;
   otherwise automatically routes via 'SIP_FIRST'.

All trade quantities are floored to integer share counts, micro-trades under minimum
INR thresholds are filtered out, residual cash is tracked, and post-trade weights
are computed.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from wrapper.parser import PortfolioParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AllocationRouter:
    """Intelligent Portfolio Allocation and Rebalancing Router.

    Calculates execution plans for cash inflows (SIP) and tactical portfolio
    rebalancing with integer share flooring, minimum trade filtering, and
    drift threshold switching.
    """

    def generate_allocation_plan(
        self,
        current_portfolio: pd.DataFrame,
        target_weights: Union[pd.Series, Dict[str, float]],
        cash_budget: float = 0.0,
        latest_prices: Optional[Union[pd.Series, Dict[str, float]]] = None,
        mode: str = "SIP_FIRST",
        drift_threshold: float = 0.05,
        min_trade_inr: float = 1000.0,
        asset_tolerance: float = 0.025,
    ) -> pd.DataFrame:
        """Generates an actionable trade allocation plan for portfolio holdings.

        Args:
            current_portfolio: DataFrame containing portfolio holdings. Expected columns
                include 'Ticker', 'Quantity', and 'Current_Price' (or 'Buy_Price').
            target_weights: Target allocation fractions keyed by Ticker or Asset_Class
                (e.g., {'^NSEI': 0.60, 'GOLDBEES.NS': 0.20, 'LIQUIDBEES.NS': 0.20} or Series).
            cash_budget: Fresh cash capital to deploy in INR (default 0.0).
            latest_prices: Optional Series or Dict of latest market prices per ticker.
            mode: Routing mode, either 'SIP_FIRST' (cash inflow without sells) or
                'TACTICAL' (full rebalance with buy/sell if drift exceeds threshold).
            drift_threshold: Maximum allowable drift fraction before triggering tactical
                sells in 'TACTICAL' mode (default 0.05 = 5%).
            min_trade_inr: Minimum order execution value threshold in INR (default 1000.0).
            asset_tolerance: Per-asset weight deadband (default 0.025 = 2.5%). Assets
                whose individual |Current_Weight − Target_Weight| falls within this
                tolerance are held steady (HOLD, 0 units) to reduce micro-trade churn.
                In SIP_FIRST mode, fresh capital is only routed to assets whose weight
                deficit exceeds asset_tolerance / 2.

        Returns:
            pd.DataFrame with comprehensive trade allocation details per asset:
                - 'Ticker': Standardized ticker symbol.
                - 'Asset_Class': Inferred or provided asset class.
                - 'Current_Price': Execution price in INR.
                - 'Current_Quantity': Pre-trade share count.
                - 'Current_Value': Pre-trade market value in INR.
                - 'Current_Weight': Pre-trade portfolio weight (fraction).
                - 'Target_Weight': Desired target weight (fraction).
                - 'Target_Value': Target market value in INR.
                - 'Drift': Current weight minus target weight.
                - 'Action': Recommended action ('BUY', 'SELL', 'HOLD').
                - 'Trade_Units': Integer share quantity to transact.
                - 'Trade_Value_INR': Value of executed trade in INR.
                - 'Post_Trade_Quantity': Post-trade share count.
                - 'Post_Trade_Value': Expected post-trade market value in INR.
                - 'Post_Trade_Weight': Expected post-trade portfolio weight (fraction).
        """
        # 1. Normalize and prepare price lookup
        price_lookup = self._build_price_lookup(latest_prices)

        # 2. Normalize and align portfolio holdings and target weights
        norm_portfolio, norm_targets = self._align_portfolio_and_targets(
            current_portfolio=current_portfolio,
            target_weights=target_weights,
            price_lookup=price_lookup,
        )

        total_portfolio_val = norm_portfolio["Current_Value"].sum()
        total_post_val = total_portfolio_val + max(0.0, cash_budget)

        # Compute current weights and drift
        if total_portfolio_val > 0:
            norm_portfolio["Current_Weight"] = norm_portfolio["Current_Value"] / total_portfolio_val
        else:
            norm_portfolio["Current_Weight"] = 0.0

        norm_portfolio["Target_Weight"] = norm_portfolio["Ticker"].map(norm_targets).fillna(0.0)
        norm_portfolio["Drift"] = norm_portfolio["Current_Weight"] - norm_portfolio["Target_Weight"]
        norm_portfolio["Abs_Drift"] = norm_portfolio["Drift"].abs()
        norm_portfolio["Target_Value"] = total_post_val * norm_portfolio["Target_Weight"]

        # 3. Determine Execution Mode
        requested_mode = mode.strip().upper() if isinstance(mode, str) else str(mode).strip().upper()
        max_drift = float(norm_portfolio["Abs_Drift"].max()) if not norm_portfolio.empty else 0.0

        executed_mode = requested_mode
        if requested_mode == "TACTICAL":
            if max_drift <= drift_threshold:
                logger.info(
                    f"Max portfolio drift ({max_drift:.2%}) is within drift threshold "
                    f"({drift_threshold:.2%}). Routing automatically via 'SIP_FIRST'."
                )
                executed_mode = "SIP_FIRST"
            else:
                logger.info(
                    f"Max portfolio drift ({max_drift:.2%}) exceeds threshold "
                    f"({drift_threshold:.2%}). Executing 'TACTICAL' rebalance."
                )

        # 4. Execute Allocation Strategy
        if executed_mode == "TACTICAL":
            plan_df = self._execute_tactical_allocation(
                df=norm_portfolio,
                cash_budget=cash_budget,
                total_post_val=total_post_val,
                min_trade_inr=min_trade_inr,
                asset_tolerance=asset_tolerance,
            )
        else:
            plan_df = self._execute_sip_first_allocation(
                df=norm_portfolio,
                cash_budget=cash_budget,
                total_post_val=total_post_val,
                min_trade_inr=min_trade_inr,
                asset_tolerance=asset_tolerance,
            )

        # 5. Compute Post-Trade Quantities, Values, and Weights
        plan_df = self._compute_post_trade_metrics(
            plan_df=plan_df,
            cash_budget=cash_budget,
            total_post_val=total_post_val,
        )

        # Attach execution metadata to DataFrame attributes
        residual_cash = plan_df.attrs.get("residual_cash", 0.0)
        plan_df.attrs = {
            "mode_requested": requested_mode,
            "mode_executed": executed_mode,
            "cash_budget": cash_budget,
            "residual_cash": float(residual_cash),
            "total_portfolio_value_pre": total_portfolio_val,
            "total_portfolio_value_post": total_post_val,
            "max_drift": max_drift,
            "drift_threshold": drift_threshold,
            "min_trade_inr": min_trade_inr,
            "asset_tolerance": asset_tolerance,
            "buy_orders_count": (plan_df["Action"] == "BUY").sum(),
            "sell_orders_count": (plan_df["Action"] == "SELL").sum(),
            "hold_count": (plan_df["Action"] == "HOLD").sum(),
        }

        return plan_df

    def _execute_sip_first_allocation(
        self,
        df: pd.DataFrame,
        cash_budget: float,
        total_post_val: float,
        min_trade_inr: float,
        asset_tolerance: float = 0.025,
    ) -> pd.DataFrame:
        """Executes 'SIP_FIRST' cash deployment logic without selling any holdings.

        Allocates fresh cash budget proportionally across under-allocated assets.
        """
        res_df = df.copy()
        cash_avail = max(0.0, cash_budget)

        # Calculate shortfall (dollar gap) for under-allocated assets
        # Shortfall_i = max(0, Target_Value_i - Current_Value_i)
        res_df["Shortfall"] = np.maximum(0.0, (res_df["Target_Value"] - res_df["Current_Value"]).to_numpy()).tolist()
        total_shortfall = float(res_df["Shortfall"].sum())

        if total_shortfall > 0:
            res_df["Allocated_Cash"] = cash_avail * (res_df["Shortfall"] / total_shortfall)
        else:
            # If no shortfall, distribute cash according to target weights
            res_df["Allocated_Cash"] = cash_avail * res_df["Target_Weight"]

        actions: List[str] = []
        trade_units: List[int] = []
        trade_values: List[float] = []

        sip_deadband = asset_tolerance / 2.0

        for _, row in res_df.iterrows():
            # Asset-level deadband: skip near-target assets to avoid dusting
            weight_deficit = float(row["Target_Weight"]) - float(row["Current_Weight"])
            if weight_deficit <= sip_deadband:
                actions.append("HOLD")
                trade_units.append(0)
                trade_values.append(0.0)
                continue

            alloc_cash = float(row["Allocated_Cash"])
            price = float(row["Current_Price"]) if float(row["Current_Price"]) > 0 else 1.0

            # Floor buy units to integer share counts
            raw_units = int(np.floor(alloc_cash / price))
            trade_val = raw_units * price

            # Strictly ZERO SELL orders in SIP_FIRST mode
            if raw_units > 0 and trade_val >= min_trade_inr:
                actions.append("BUY")
                trade_units.append(raw_units)
                trade_values.append(round(trade_val, 2))
            else:
                actions.append("HOLD")
                trade_units.append(0)
                trade_values.append(0.0)

        res_df["Action"] = actions
        res_df["Trade_Units"] = trade_units
        res_df["Trade_Value_INR"] = trade_values

        return res_df

    def _execute_tactical_allocation(
        self,
        df: pd.DataFrame,
        cash_budget: float,
        total_post_val: float,
        min_trade_inr: float,
        asset_tolerance: float = 0.025,
    ) -> pd.DataFrame:
        """Executes 'TACTICAL' rebalancing generating SELL and BUY orders.

        Overweight assets generate SELL orders freeing up liquidity. The combined
        proceeds and fresh cash budget are deployed into underweight assets.
        """
        res_df = df.copy()
        cash_avail = max(0.0, cash_budget)

        # Net dollar adjustment needed per asset
        res_df["Delta_Value"] = res_df["Target_Value"] - res_df["Current_Value"]

        actions: List[str] = ["HOLD"] * len(res_df)
        trade_units: List[int] = [0] * len(res_df)
        trade_values: List[float] = [0.0] * len(res_df)

        # Step 1: Process SELL orders for overweight assets (Delta_Value < 0)
        cash_from_sells = 0.0
        for idx_val, row in res_df.iterrows():
            i = int(idx_val)  # type: ignore[arg-type]

            # Asset-level deadband: skip assets within tolerance
            individual_drift = abs(float(row["Current_Weight"]) - float(row["Target_Weight"]))
            if individual_drift <= asset_tolerance:
                continue  # leave as HOLD with 0 units

            delta_val = float(row["Delta_Value"])
            price = float(row["Current_Price"]) if float(row["Current_Price"]) > 0 else 1.0
            curr_qty = int(row["Current_Quantity"])

            if delta_val < 0:
                sell_amount = abs(delta_val)
                raw_sell_units = int(np.floor(sell_amount / price))
                # Cannot sell more shares than held
                actual_sell_units = min(curr_qty, raw_sell_units)
                sell_val = actual_sell_units * price

                if actual_sell_units > 0 and sell_val >= min_trade_inr:
                    actions[i] = "SELL"
                    trade_units[i] = actual_sell_units
                    trade_values[i] = round(sell_val, 2)
                    cash_from_sells += sell_val

        # Step 2: Combine fresh cash budget and proceeds from sells
        total_buy_cash_pool = cash_avail + cash_from_sells

        # Step 3: Process BUY orders for underweight assets (Delta_Value > 0)
        # Only consider assets that breach the deadband tolerance
        underweight_mask = res_df.apply(
            lambda r: float(r["Delta_Value"]) > 0
            and abs(float(r["Current_Weight"]) - float(r["Target_Weight"])) > asset_tolerance,
            axis=1,
        )
        total_buy_shortfall = float(res_df.loc[underweight_mask, "Delta_Value"].sum())

        for idx_val, row in res_df.iterrows():
            i = int(idx_val)  # type: ignore[arg-type]

            # Asset-level deadband: skip assets within tolerance
            individual_drift = abs(float(row["Current_Weight"]) - float(row["Target_Weight"]))
            if individual_drift <= asset_tolerance:
                continue  # already HOLD from Step 1 pass

            delta_val = float(row["Delta_Value"])
            price = float(row["Current_Price"]) if float(row["Current_Price"]) > 0 else 1.0

            if delta_val > 0:
                if total_buy_shortfall > 0:
                    allocated_buy_cash = total_buy_cash_pool * (delta_val / total_buy_shortfall)
                else:
                    allocated_buy_cash = total_buy_cash_pool * float(row["Target_Weight"])

                raw_buy_units = int(np.floor(allocated_buy_cash / price))
                buy_val = raw_buy_units * price

                if raw_buy_units > 0 and buy_val >= min_trade_inr:
                    actions[i] = "BUY"
                    trade_units[i] = raw_buy_units
                    trade_values[i] = round(buy_val, 2)

        res_df["Action"] = actions
        res_df["Trade_Units"] = trade_units
        res_df["Trade_Value_INR"] = trade_values

        return res_df

    def _compute_post_trade_metrics(
        self,
        plan_df: pd.DataFrame,
        cash_budget: float,
        total_post_val: float,
    ) -> pd.DataFrame:
        """Calculates post-trade quantities, values, weights, and residual cash."""
        post_quantities: List[int] = []
        post_values: List[float] = []

        total_cash_spent = 0.0
        total_cash_from_sells = 0.0

        for _, row in plan_df.iterrows():
            action = row["Action"]
            curr_qty = int(row["Current_Quantity"])
            units = int(row["Trade_Units"])
            price = float(row["Current_Price"])
            trade_val = float(row["Trade_Value_INR"])

            if action == "BUY":
                new_qty = curr_qty + units
                new_val = new_qty * price
                total_cash_spent += trade_val
            elif action == "SELL":
                new_qty = max(0, curr_qty - units)
                new_val = new_qty * price
                total_cash_from_sells += trade_val
            else:
                new_qty = curr_qty
                new_val = float(row["Current_Value"])

            post_quantities.append(new_qty)
            post_values.append(round(new_val, 2))

        plan_df["Post_Trade_Quantity"] = post_quantities
        plan_df["Post_Trade_Value"] = post_values

        # Residual cash tracking
        total_cash_in = max(0.0, cash_budget) + total_cash_from_sells
        residual_cash = max(0.0, total_cash_in - total_cash_spent)

        # Expected post-trade weights relative to full portfolio including residual cash
        if total_post_val > 0:
            plan_df["Post_Trade_Weight"] = plan_df["Post_Trade_Value"] / total_post_val
        else:
            plan_df["Post_Trade_Weight"] = 0.0

        # Store residual cash in attrs
        plan_df.attrs["residual_cash"] = round(residual_cash, 2)

        # Select and format final output columns
        output_cols = [
            "Ticker",
            "Asset_Class",
            "Current_Price",
            "Current_Quantity",
            "Current_Value",
            "Current_Weight",
            "Target_Weight",
            "Target_Value",
            "Drift",
            "Action",
            "Trade_Units",
            "Trade_Value_INR",
            "Post_Trade_Quantity",
            "Post_Trade_Value",
            "Post_Trade_Weight",
        ]
        available_cols = [c for c in output_cols if c in plan_df.columns]
        return plan_df[available_cols]

    def _build_price_lookup(
        self,
        latest_prices: Optional[Union[pd.Series, Dict[str, float]]]
    ) -> Dict[str, float]:
        """Builds a standardized price lookup dictionary."""
        lookup: Dict[str, float] = {}
        if latest_prices is None:
            return lookup

        if isinstance(latest_prices, pd.Series):
            raw_dict = latest_prices.dropna().to_dict()
        elif isinstance(latest_prices, dict):
            raw_dict = {k: v for k, v in latest_prices.items() if v is not None}
        else:
            return lookup

        for k, v in raw_dict.items():
            val = float(v)
            lookup[str(k)] = val
            lookup[PortfolioParser.standardize_ticker(str(k))] = val
            raw_sym = str(k).replace(".NS", "").replace(".BO", "").strip().upper()
            lookup[raw_sym] = val

        return lookup

    def _align_portfolio_and_targets(
        self,
        current_portfolio: pd.DataFrame,
        target_weights: Union[pd.Series, Dict[str, float]],
        price_lookup: Dict[str, float],
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Aligns holdings and target weights into a unified asset universe."""
        df_raw = current_portfolio.copy() if current_portfolio is not None else pd.DataFrame()

        # Build clean canonical DataFrame with unambiguous columns
        col_lower_map = {c.strip().lower().replace(" ", "_") if isinstance(c, str) else str(c).strip().lower().replace(" ", "_"): c for c in df_raw.columns}

        # 1. Ticker column
        ticker_col = next((col_lower_map[a] for a in ["ticker", "symbol", "asset", "code", "stock", "instrument"] if a in col_lower_map), None)
        # 2. Quantity column
        qty_col = next((col_lower_map[a] for a in ["current_quantity", "quantity", "qty", "units", "shares", "holding_qty"] if a in col_lower_map), None)
        # 3. Price column (prefer current market price, fallback to buy/avg price)
        price_col = next((col_lower_map[a] for a in ["current_price", "ltp", "price", "market_price", "cmp", "close", "last_price"] if a in col_lower_map), None)
        if price_col is None:
            price_col = next((col_lower_map[a] for a in ["buy_price", "avg_price", "cost_price", "average_price"] if a in col_lower_map), None)
        # 4. Asset Class column
        ac_col = next((col_lower_map[a] for a in ["asset_class", "type", "category", "asset_type", "sector"] if a in col_lower_map), None)

        clean_records = []
        if not df_raw.empty and ticker_col is not None:
            for _, r in df_raw.iterrows():
                t = PortfolioParser.standardize_ticker(r[ticker_col])
                raw_q = pd.to_numeric(r[qty_col], errors="coerce") if qty_col is not None and not pd.isna(r.get(qty_col)) else float("nan")
                q = int(raw_q) if not pd.isna(raw_q) else 0
                raw_p = pd.to_numeric(r[price_col], errors="coerce") if price_col is not None and not pd.isna(r.get(price_col)) else float("nan")
                p = float(raw_p) if not pd.isna(raw_p) else 0.0
                if p <= 0 and t in price_lookup:
                    p = price_lookup[t]
                ac = str(r[ac_col]) if ac_col is not None and not pd.isna(r.get(ac_col)) else PortfolioParser.infer_asset_class(t)
                clean_records.append({
                    "Ticker": t,
                    "Current_Quantity": q,
                    "Current_Price": p,
                    "Asset_Class": ac,
                    "Current_Value": round(q * p, 2),
                })
        df = pd.DataFrame(clean_records) if clean_records else pd.DataFrame(columns=["Ticker", "Current_Quantity", "Current_Price", "Asset_Class", "Current_Value"])

        # Process target weights
        targets_dict: Dict[str, float] = {}
        if isinstance(target_weights, pd.Series):
            targets_dict = target_weights.to_dict()
        elif isinstance(target_weights, dict):
            targets_dict = dict(target_weights)

        # Check if targets are by Asset_Class or Ticker
        # Known asset classes
        known_classes = {"EQUITY", "GOLD", "DEBT", "CASH"}
        is_asset_class_level = any(str(k).upper() in known_classes for k in targets_dict.keys())

        standardized_targets: Dict[str, float] = {}
        if is_asset_class_level and not df.empty:
            # Distribute asset class weight across constituents
            for ac_name, weight in targets_dict.items():
                ac_str = str(ac_name).capitalize()
                matching_rows = df[df["Asset_Class"].str.lower() == ac_str.lower()]
                if not matching_rows.empty:
                    ac_total_val = float((matching_rows["Current_Quantity"] * matching_rows["Current_Price"]).sum())
                    for _, r in matching_rows.iterrows():
                        t = str(r["Ticker"])
                        curr_v = float(r["Current_Quantity"]) * float(r["Current_Price"])
                        share = (curr_v / ac_total_val) if ac_total_val > 0 else (1.0 / len(matching_rows))
                        standardized_targets[t] = float(weight) * share
                else:
                    # Representative ticker for missing asset class
                    default_rep = {"Gold": "GOLDBEES.NS", "Debt": "LIQUIDBEES.NS", "Equity": "^NSEI", "Cash": "CASH"}.get(ac_str, f"{ac_str}.NS")
                    standardized_targets[default_rep] = float(weight)
        else:
            for k, v in targets_dict.items():
                standardized_targets[PortfolioParser.standardize_ticker(str(k))] = float(v)

        # Normalize target weights to sum to 1.0
        total_target_w = sum(standardized_targets.values())
        if total_target_w > 0:
            standardized_targets = {k: v / total_target_w for k, v in standardized_targets.items()}

        # Build full universe (union of portfolio tickers and target tickers)
        universe_tickers = list(dict.fromkeys(list(df["Ticker"]) + list(standardized_targets.keys())))

        aligned_rows = []
        for ticker in universe_tickers:
            match = df[df["Ticker"] == ticker]
            if not match.empty:
                r = match.iloc[0]
                qty = int(r["Current_Quantity"])
                price = float(r["Current_Price"])
                if price <= 0 and ticker in price_lookup:
                    price = price_lookup[ticker]
                ac = str(r.get("Asset_Class", PortfolioParser.infer_asset_class(ticker)))
            else:
                qty = 0
                price = price_lookup.get(ticker, 100.0)
                ac = PortfolioParser.infer_asset_class(ticker)

            curr_val = round(qty * price, 2)
            aligned_rows.append({
                "Ticker": ticker,
                "Asset_Class": ac,
                "Current_Quantity": qty,
                "Current_Price": price,
                "Current_Value": curr_val,
            })

        aligned_df = pd.DataFrame(aligned_rows).reset_index(drop=True)
        return aligned_df, standardized_targets

    # -------------------------------------------------------------------------
    # Backward-compatible convenience methods
    # -------------------------------------------------------------------------

    def route_tactical_rebalance(
        self,
        current_holdings: pd.DataFrame,
        target_weights: Union[Dict[str, float], pd.Series],
        min_trade_amount: float = 1000.0,
        cash_budget: float = 0.0,
        latest_prices: Optional[Union[pd.Series, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """Convenience method for tactical rebalancing execution."""
        plan_df = self.generate_allocation_plan(
            current_portfolio=current_holdings,
            target_weights=target_weights,
            cash_budget=cash_budget,
            latest_prices=latest_prices,
            mode="TACTICAL",
            drift_threshold=0.0,  # Force tactical rebalance when explicitly called
            min_trade_inr=min_trade_amount,
        )

        trade_recs = plan_df[plan_df["Action"].isin(["BUY", "SELL"])].copy()
        if not trade_recs.empty:
            trade_recs = trade_recs.rename(
                columns={
                    "Current_Weight": "Current_Weight_%",
                    "Target_Weight": "Target_Weight_%",
                    "Trade_Value_INR": "Trade_Amount_INR",
                    "Current_Price": "Current_Price_INR",
                }
            )
            trade_recs["Current_Weight_%"] = (trade_recs["Current_Weight_%"] * 100).round(2)
            trade_recs["Target_Weight_%"] = (trade_recs["Target_Weight_%"] * 100).round(2)

        return {
            "mode": "Tactical Rebalance",
            "total_portfolio_value": plan_df.attrs.get("total_portfolio_value_pre", 0.0),
            "trade_recommendations": trade_recs,
            "trade_count": len(trade_recs),
            "residual_cash": plan_df.attrs.get("residual_cash", 0.0),
            "allocation_plan": plan_df,
        }

    def route_sip_inflow(
        self,
        current_holdings: pd.DataFrame,
        target_weights: Union[Dict[str, float], pd.Series],
        sip_amount: float,
        latest_prices: Optional[Union[pd.Series, Dict[str, float]]] = None,
        min_trade_amount: float = 1000.0,
    ) -> Dict[str, Any]:
        """Convenience method for fresh SIP cash deployment."""
        plan_df = self.generate_allocation_plan(
            current_portfolio=current_holdings,
            target_weights=target_weights,
            cash_budget=sip_amount,
            latest_prices=latest_prices,
            mode="SIP_FIRST",
            min_trade_inr=min_trade_amount,
        )

        sip_recs = plan_df[plan_df["Action"] == "BUY"].copy()
        if not sip_recs.empty:
            sip_recs = sip_recs.rename(
                columns={
                    "Trade_Units": "Recommended_Units",
                    "Trade_Value_INR": "SIP_Allocation_INR",
                    "Current_Price": "Current_Price_INR",
                    "Post_Trade_Weight": "Post_SIP_Weight_%",
                }
            )
            sip_recs["Post_SIP_Weight_%"] = (sip_recs["Post_SIP_Weight_%"] * 100).round(2)

        return {
            "mode": "SIP Cash Inflow",
            "sip_amount": sip_amount,
            "new_portfolio_value": plan_df.attrs.get("total_portfolio_value_post", 0.0),
            "sip_recommendations": sip_recs,
            "residual_cash": plan_df.attrs.get("residual_cash", 0.0),
            "allocation_plan": plan_df,
        }
