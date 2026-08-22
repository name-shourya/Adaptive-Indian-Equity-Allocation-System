"""Portfolio Parser Module for Indian Equities.

This module provides the `PortfolioParser` class for ingesting, standardizing,
validating, and analyzing user portfolio holdings from CSV or Excel files.
It computes critical financial metrics including position market values in INR,
unrealized P&L and percentages, portfolio weights, and position health indicators
('GREEN' / 'RED').
"""

import io
import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Standard column names required in the output DataFrame
REQUIRED_COLUMNS: List[str] = [
    "Ticker",
    "Quantity",
    "Buy_Price",
    "Current_Price",
]

# Aliases mapping various CSV/Excel column headers to standard names
COLUMN_ALIASES: Dict[str, str] = {
    # Ticker / Symbol
    "ticker": "Ticker",
    "symbol": "Ticker",
    "asset": "Ticker",
    "code": "Ticker",
    "stock": "Ticker",
    "instrument": "Ticker",
    "name": "Ticker",
    "security": "Ticker",
    # Quantity / Units
    "quantity": "Quantity",
    "qty": "Quantity",
    "units": "Quantity",
    "shares": "Quantity",
    "volume": "Quantity",
    "holding_qty": "Quantity",
    "holding_units": "Quantity",
    # Buy Price / Avg Price
    "buy_price": "Buy_Price",
    "avg_price": "Buy_Price",
    "cost_price": "Buy_Price",
    "average_price": "Buy_Price",
    "purchase_price": "Buy_Price",
    "buy_rate": "Buy_Price",
    "cost": "Buy_Price",
    "entry_price": "Buy_Price",
    # Current Price / Market Price
    "current_price": "Current_Price",
    "ltp": "Current_Price",
    "price": "Current_Price",
    "market_price": "Current_Price",
    "cmp": "Current_Price",
    "close": "Current_Price",
    "close_price": "Current_Price",
    "last_price": "Current_Price",
    # Asset Class (Optional)
    "asset_class": "Asset_Class",
    "type": "Asset_Class",
    "category": "Asset_Class",
    "asset_type": "Asset_Class",
    "sector": "Asset_Class",
}


class PortfolioParser:
    """Parses and validates CSV and Excel user portfolio state files.

    Standardizes Indian ticker symbols, resolves current market prices from
    direct columns or external lookup tables, calculates unrealized P&L,
    portfolio weight distributions, and flags position status ('GREEN' / 'RED').
    """

    @staticmethod
    def standardize_ticker(ticker: Any) -> str:
        """Standardizes Indian stock ticker symbols by auto-appending '.NS' if not present.

        Args:
            ticker: Raw ticker symbol (e.g., 'RELIANCE', 'TCS.NS', '^NSEI', 'GOLDBEES').

        Returns:
            Standardized uppercase ticker string with '.NS' suffix for Indian equities,
            while preserving indices (starting with '^') and existing suffixes ('.NS', '.BO').
        """
        if ticker is None or pd.isna(ticker):
            return "UNKNOWN.NS"

        ticker_str = str(ticker).strip().upper()
        if not ticker_str:
            return "UNKNOWN.NS"

        # Preserve indices (e.g., ^NSEI, ^BSESN) and cash representations
        if ticker_str.startswith("^") or ticker_str in {"CASH", "LIQUID", "INR"}:
            return ticker_str

        # Preserve existing valid exchange suffixes
        if ticker_str.endswith(".NS") or ticker_str.endswith(".BO"):
            return ticker_str

        # Auto-append .NS for Indian equity symbols
        return f"{ticker_str}.NS"

    @staticmethod
    def infer_asset_class(ticker: str) -> str:
        """Infers asset class (Equity, Gold, Debt, Cash) based on ticker pattern.

        Args:
            ticker: Standardized ticker symbol.

        Returns:
            Asset class string ('Equity', 'Gold', 'Debt', or 'Cash').
        """
        t_upper = ticker.upper()
        if "GOLD" in t_upper:
            return "Gold"
        if any(x in t_upper for x in ["LIQUID", "DEBT", "GILT", "BOND", "TREASURY"]):
            return "Debt"
        if t_upper in {"CASH", "INR", "USD"}:
            return "Cash"
        return "Equity"

    def parse(
        self,
        file_path_or_buffer: Union[str, os.PathLike, io.BytesIO, io.StringIO, Any],
        latest_prices: Optional[Union[Dict[str, float], pd.Series]] = None,
    ) -> Dict[str, Any]:
        """Parses portfolio holdings from a file path or file-like buffer.

        Args:
            file_path_or_buffer: File path string or file-like object/buffer (e.g. Streamlit UploadedFile).
            latest_prices: Optional dictionary or Series of latest market prices keyed by ticker.

        Returns:
            Dict containing:
                - 'holdings_df': Cleaned pd.DataFrame with standardized columns.
                - 'summary': Dictionary of total portfolio metrics.
                - 'total_market_value': Total portfolio market value in INR.
                - 'total_cost': Total investment cost in INR.
                - 'total_pnl': Total unrealized P&L in INR.
                - 'total_pnl_pct': Total unrealized return percentage.
                - 'asset_weights': Dict of asset weights.
                - 'asset_class_df': Summary DataFrame grouped by asset class.
                - 'success': Boolean success status.
        """
        try:
            df = self._read_file_to_dataframe(file_path_or_buffer)
            return self.parse_dataframe(df, latest_prices=latest_prices)
        except Exception as e:
            logger.error(f"Failed to parse portfolio file: {e}", exc_info=True)
            return self._fallback_sample_portfolio(latest_prices=latest_prices)

    def parse_portfolio_file(
        self,
        file_path_or_buffer: Union[str, os.PathLike, io.BytesIO, io.StringIO, Any],
        latest_prices: Optional[Union[Dict[str, float], pd.Series]] = None,
    ) -> Dict[str, Any]:
        """Convenience alias for `parse` to maintain API consistency."""
        return self.parse(file_path_or_buffer, latest_prices=latest_prices)

    def parse_dataframe(
        self,
        df: pd.DataFrame,
        latest_prices: Optional[Union[Dict[str, float], pd.Series]] = None,
    ) -> Dict[str, Any]:
        """Validates and processes a raw portfolio DataFrame into standardized metrics.

        Args:
            df: Raw input DataFrame containing portfolio columns.
            latest_prices: Optional dictionary or Series of latest market prices keyed by ticker.

        Returns:
            Dict containing cleaned DataFrame, total portfolio metrics summary,
            and asset class weight breakdowns.
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to parse_dataframe. Synthesizing empty structure.")
            df = pd.DataFrame(columns=["Ticker", "Quantity", "Buy_Price", "Current_Price"])

        # Copy to prevent mutation
        df = df.copy()

        # Standardize column headers using alias map
        renamed_cols = {}
        for col in df.columns:
            clean_col = col.strip().lower().replace(" ", "_").replace("-", "_") if isinstance(col, str) else str(col).strip().lower().replace(" ", "_").replace("-", "_")
            if clean_col in COLUMN_ALIASES:
                renamed_cols[col] = COLUMN_ALIASES[clean_col]

        df = df.rename(columns=renamed_cols)

        # Ensure required columns exist
        if "Ticker" not in df.columns:
            logger.warning("Column 'Ticker' missing in portfolio. Synthesizing generic tickers.")
            df["Ticker"] = [f"ASSET_{i+1}" for i in range(len(df))]

        if "Quantity" not in df.columns:
            logger.warning("Column 'Quantity' missing in portfolio. Defaulting to 0.")
            df["Quantity"] = 0.0

        if "Buy_Price" not in df.columns:
            logger.warning("Column 'Buy_Price' missing in portfolio. Defaulting to 0.0.")
            df["Buy_Price"] = 0.0

        if "Current_Price" not in df.columns:
            df["Current_Price"] = np.nan

        # Standardize ticker symbols
        df["Ticker"] = df["Ticker"].apply(self.standardize_ticker)

        # Standardize numeric columns
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0).astype(float)
        df["Buy_Price"] = pd.to_numeric(df["Buy_Price"], errors="coerce").fillna(0.0).astype(float)
        df["Current_Price"] = pd.to_numeric(df["Current_Price"], errors="coerce")

        # Resolve missing Current_Price using latest_prices if provided
        price_lookup: Dict[str, float] = {}
        if latest_prices is not None:
            if isinstance(latest_prices, pd.Series):
                price_lookup = latest_prices.dropna().to_dict()
            elif isinstance(latest_prices, dict):
                price_lookup = {k: v for k, v in latest_prices.items() if v is not None}

            # Normalize keys in lookup dictionary for flexible matching
            norm_lookup: Dict[str, float] = {}
            for k, v in price_lookup.items():
                norm_lookup[k] = float(v)
                norm_lookup[self.standardize_ticker(k)] = float(v)
                raw_k = str(k).replace(".NS", "").replace(".BO", "").strip().upper()
                norm_lookup[raw_k] = float(v)
            price_lookup = norm_lookup

        # Apply price lookup for NaN or non-positive Current_Price
        for idx in df.index:
            curr_p = df.at[idx, "Current_Price"]
            if pd.isna(curr_p) or float(curr_p) <= 0:
                ticker = df.at[idx, "Ticker"]
                if ticker in price_lookup:
                    df.at[idx, "Current_Price"] = price_lookup[ticker]
                else:
                    # Fallback to Buy_Price if Current_Price not provided
                    df.at[idx, "Current_Price"] = df.at[idx, "Buy_Price"]

        # Final cleanup for Current_Price
        df["Current_Price"] = df["Current_Price"].fillna(df["Buy_Price"]).fillna(0.0).astype(float)

        # Infer Asset_Class if missing
        if "Asset_Class" not in df.columns:
            df["Asset_Class"] = df["Ticker"].apply(self.infer_asset_class)
        else:
            df["Asset_Class"] = df["Asset_Class"].fillna(df["Ticker"].apply(self.infer_asset_class)).astype(str)

        # Financial Computations
        # 1. Investment Cost & Current Market Value in INR
        df["Investment_Cost"] = df["Quantity"] * df["Buy_Price"]
        df["Current_Value"] = df["Quantity"] * df["Current_Price"]

        # 2. Unrealized Profit / Loss
        df["PnL"] = df["Current_Value"] - df["Investment_Cost"]
        df["PnL_Pct"] = np.where(
            (df["Investment_Cost"] > 0).to_numpy(),
            ((df["PnL"] / df["Investment_Cost"]) * 100.0).to_numpy(),
            0.0
        )

        # 3. Portfolio Totals
        total_market_value = float(df["Current_Value"].sum())
        total_cost = float(df["Investment_Cost"].sum())
        total_pnl = total_market_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100.0) if total_cost > 0 else 0.0

        # 4. Current Weights
        if total_market_value > 0:
            df["Current_Weight"] = df["Current_Value"] / total_market_value
            df["Current_Weight_Pct"] = df["Current_Weight"] * 100.0
        else:
            df["Current_Weight"] = 0.0
            df["Current_Weight_Pct"] = 0.0

        # 5. Position Status ('GREEN' vs 'RED')
        df["Position_Status"] = np.where((df["PnL"] >= 0.0).to_numpy(), "GREEN", "RED")

        # Standard column ordering
        standard_cols = [
            "Ticker",
            "Asset_Class",
            "Quantity",
            "Buy_Price",
            "Current_Price",
            "Investment_Cost",
            "Current_Value",
            "PnL",
            "PnL_Pct",
            "Current_Weight",
            "Current_Weight_Pct",
            "Position_Status",
        ]
        # Include any extra original columns at the end
        extra_cols = [c for c in df.columns if c not in standard_cols]
        df = df[standard_cols + extra_cols]

        # Asset Class Aggregations
        asset_class_df = df.groupby("Asset_Class").agg(
            Market_Value=("Current_Value", "sum"),
            Investment_Cost=("Investment_Cost", "sum"),
            PnL=("PnL", "sum"),
        ).reset_index()

        if total_market_value > 0:
            asset_class_df["Weight_Pct"] = (asset_class_df["Market_Value"] / total_market_value) * 100.0
            asset_class_df["Weight"] = asset_class_df["Market_Value"] / total_market_value
        else:
            asset_class_df["Weight_Pct"] = 0.0
            asset_class_df["Weight"] = 0.0

        asset_weights = dict(zip(asset_class_df["Asset_Class"], asset_class_df["Weight"]))
        ticker_weights = dict(zip(df["Ticker"], df["Current_Weight"]))

        summary: Dict[str, Any] = {
            "total_portfolio_value": total_market_value,
            "total_investment_cost": total_cost,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "total_positions": len(df),
            "green_positions_count": (df["Position_Status"] == "GREEN").sum(),
            "red_positions_count": (df["Position_Status"] == "RED").sum(),
        }

        return {
            "success": True,
            "holdings_df": df,
            "summary": summary,
            "total_market_value": total_market_value,
            "total_cost": total_cost,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "asset_class_df": asset_class_df,
            "asset_weights": asset_weights,
            "ticker_weights": ticker_weights,
        }

    def _read_file_to_dataframe(
        self,
        file_path_or_buffer: Union[str, os.PathLike, io.BytesIO, io.StringIO, bytes, Any]
    ) -> pd.DataFrame:
        """Reads CSV or Excel file / buffer into a pandas DataFrame with format detection."""
        if isinstance(file_path_or_buffer, (str, os.PathLike)):
            path_str = str(file_path_or_buffer)
            if path_str.lower().endswith((".xlsx", ".xls", ".xlsm")):
                return pd.read_excel(path_str)
            try:
                return pd.read_csv(path_str, encoding="utf-8")
            except UnicodeDecodeError:
                return pd.read_csv(path_str, encoding="latin1")

        if isinstance(file_path_or_buffer, io.StringIO):
            return pd.read_csv(file_path_or_buffer)

        # For BytesIO, UploadedFile, or raw bytes
        filename = getattr(file_path_or_buffer, "name", "").lower()
        if filename.endswith((".xlsx", ".xls", ".xlsm")):
            return pd.read_excel(file_path_or_buffer)

        if isinstance(file_path_or_buffer, (io.BytesIO, bytes)) or hasattr(file_path_or_buffer, "read"):
            # Check magic bytes for ZIP/XLSX (PK\x03\x04) or OLE/XLS (\xd0\xcf\x11\xe0)
            if hasattr(file_path_or_buffer, "seek") and hasattr(file_path_or_buffer, "read"):
                file_path_or_buffer.seek(0)
                magic = file_path_or_buffer.read(4)
                file_path_or_buffer.seek(0)
                if magic == b"PK\x03\x04" or magic == b"\xd0\xcf\x11\xe0":
                    return pd.read_excel(file_path_or_buffer)

            try:
                if hasattr(file_path_or_buffer, "seek"):
                    file_path_or_buffer.seek(0)
                return pd.read_csv(file_path_or_buffer)  # type: ignore[arg-type]
            except Exception:
                if hasattr(file_path_or_buffer, "seek"):
                    file_path_or_buffer.seek(0)
                return pd.read_excel(file_path_or_buffer)

        return pd.read_csv(file_path_or_buffer)

    def create_sample_portfolio_csv(
        self,
        output_path: str = "data/raw/sample_portfolio.csv"
    ) -> str:
        """Generates a standard sample Indian user portfolio CSV file.

        Args:
            output_path: Target filesystem path for the sample CSV.

        Returns:
            The absolute path of the created sample portfolio CSV.
        """
        sample_data = pd.DataFrame([
            {"Ticker": "^NSEI", "Quantity": 15, "Buy_Price": 22000.0, "Current_Price": 24500.0, "Asset_Class": "Equity"},
            {"Ticker": "RELIANCE", "Quantity": 50, "Buy_Price": 2800.0, "Current_Price": 3050.0, "Asset_Class": "Equity"},
            {"Ticker": "TCS", "Quantity": 30, "Buy_Price": 3800.0, "Current_Price": 4200.0, "Asset_Class": "Equity"},
            {"Ticker": "GOLDBEES", "Quantity": 1500, "Buy_Price": 58.0, "Current_Price": 66.5, "Asset_Class": "Gold"},
            {"Ticker": "LIQUIDBEES", "Quantity": 150, "Buy_Price": 1000.0, "Current_Price": 1000.0, "Asset_Class": "Debt"},
        ])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sample_data.to_csv(output_path, index=False)
        logger.info(f"Sample portfolio CSV created at {output_path}")
        return output_path

    def _fallback_sample_portfolio(
        self,
        latest_prices: Optional[Union[Dict[str, float], pd.Series]] = None
    ) -> Dict[str, Any]:
        """Provides default sample portfolio structure when no file is uploaded or on error."""
        sample_path = self.create_sample_portfolio_csv()
        df = pd.read_csv(sample_path)
        return self.parse_dataframe(df, latest_prices=latest_prices)
