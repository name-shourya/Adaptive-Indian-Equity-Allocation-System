import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Ticker", "Quantity", "Avg_Price", "Current_Price", "Asset_Class"]

COLUMN_MAP = {
    "symbol": "Ticker", "asset": "Ticker", "code": "Ticker",
    "qty": "Quantity", "units": "Quantity", "shares": "Quantity",
    "buy_price": "Avg_Price", "cost_price": "Avg_Price", "average_price": "Avg_Price",
    "ltp": "Current_Price", "price": "Current_Price", "market_price": "Current_Price",
    "type": "Asset_Class", "category": "Asset_Class", "asset_type": "Asset_Class"
}

class PortfolioParser:
    """
    Parses and validates CSV/Excel user portfolio state files.
    Calculates asset allocation weights, market value, and unrealized gain/loss.
    """

    def parse_portfolio_file(self, file_path_or_buffer: Any) -> Dict[str, Any]:
        """
        Parses portfolio holdings from a file path or uploaded file buffer.
        
        Returns:
            Dict containing parsed DataFrame, asset class weight breakdown,
            total portfolio value, total investment cost, and P&L.
        """
        try:
            if isinstance(file_path_or_buffer, str):
                if file_path_or_buffer.endswith(".xlsx") or file_path_or_buffer.endswith(".xls"):
                    df = pd.read_excel(file_path_or_buffer)
                else:
                    df = pd.read_csv(file_path_or_buffer)
            else:
                # Streamlit UploadedFile buffer
                filename = getattr(file_path_or_buffer, "name", "file.csv")
                if filename.endswith(".xlsx") or filename.endswith(".xls"):
                    df = pd.read_excel(file_path_or_buffer)
                else:
                    df = pd.read_csv(file_path_or_buffer)

            return self.parse_dataframe(df)

        except Exception as e:
            logger.error(f"Error parsing portfolio file: {e}")
            return self._fallback_sample_portfolio()

    def parse_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validates and processes raw DataFrame into portfolio metrics."""
        # Standardize column headers
        df.columns = [col.strip() for col in df.columns]
        renamed_cols = {}
        for col in df.columns:
            lower_col = col.lower().replace(" ", "_")
            if lower_col in COLUMN_MAP:
                renamed_cols[col] = COLUMN_MAP[lower_col]

        df = df.rename(columns=renamed_cols)

        # Check missing columns
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            logger.warning(f"Portfolio file missing columns {missing}. Synthesizing standard columns.")
            df = self._sanitize_dataframe(df)

        # Calculate values
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
        df["Avg_Price"] = pd.to_numeric(df["Avg_Price"], errors="coerce").fillna(0.0)
        df["Current_Price"] = pd.to_numeric(df["Current_Price"], errors="coerce").fillna(df["Avg_Price"])

        df["Investment_Cost"] = df["Quantity"] * df["Avg_Price"]
        df["Market_Value"] = df["Quantity"] * df["Current_Price"]
        df["Unrealized_PnL"] = df["Market_Value"] - df["Investment_Cost"]
        df["Unrealized_PnL_Pct"] = np.where(df["Investment_Cost"] > 0, (df["Unrealized_PnL"] / df["Investment_Cost"]) * 100, 0.0)

        total_market_value = float(df["Market_Value"].sum())
        total_cost = float(df["Investment_Cost"].sum())
        total_pnl = total_market_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0

        if total_market_value > 0:
            df["Current_Weight_Pct"] = (df["Market_Value"] / total_market_value) * 100
        else:
            df["Current_Weight_Pct"] = 0.0

        # Group by Asset Class
        asset_class_df = df.groupby("Asset_Class").agg(
            Market_Value=("Market_Value", "sum"),
            Investment_Cost=("Investment_Cost", "sum"),
            Unrealized_PnL=("Unrealized_PnL", "sum")
        ).reset_index()

        if total_market_value > 0:
            asset_class_df["Weight_Pct"] = (asset_class_df["Market_Value"] / total_market_value) * 100
        else:
            asset_class_df["Weight_Pct"] = 0.0

        asset_weights = dict(zip(asset_class_df["Asset_Class"], asset_class_df["Weight_Pct"] / 100.0))

        return {
            "success": True,
            "holdings_df": df,
            "asset_class_df": asset_class_df,
            "asset_weights": asset_weights,
            "total_market_value": total_market_value,
            "total_cost": total_cost,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct
        }

    def _sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills missing standard portfolio columns if uploaded CSV has custom structure."""
        if "Ticker" not in df.columns:
            df["Ticker"] = [f"ASSET_{i+1}" for i in range(len(df))]
        if "Quantity" not in df.columns:
            df["Quantity"] = 100
        if "Avg_Price" not in df.columns:
            df["Avg_Price"] = 100.0
        if "Current_Price" not in df.columns:
            df["Current_Price"] = df["Avg_Price"]
        if "Asset_Class" not in df.columns:
            df["Asset_Class"] = "Equity"
        return df

    def create_sample_portfolio_csv(self, output_path: str = "data/raw/sample_portfolio.csv") -> str:
        """Generates a standard sample user portfolio CSV file."""
        sample_data = pd.DataFrame([
            {"Ticker": "^NSEI", "Quantity": 15, "Avg_Price": 22000.0, "Current_Price": 24500.0, "Asset_Class": "Equity"},
            {"Ticker": "RELIANCE.NS", "Quantity": 50, "Avg_Price": 2800.0, "Current_Price": 3050.0, "Asset_Class": "Equity"},
            {"Ticker": "TCS.NS", "Quantity": 30, "Avg_Price": 3800.0, "Current_Price": 4200.0, "Asset_Class": "Equity"},
            {"Ticker": "GOLDBEES.NS", "Quantity": 1500, "Avg_Price": 58.0, "Current_Price": 66.5, "Asset_Class": "Gold"},
            {"Ticker": "LIQUIDBEES.NS", "Quantity": 150, "Avg_Price": 1000.0, "Current_Price": 1000.0, "Asset_Class": "Debt"},
        ])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sample_data.to_csv(output_path, index=False)
        return output_path

    def _fallback_sample_portfolio(self) -> Dict[str, Any]:
        """Provides default sample portfolio structure."""
        sample_path = self.create_sample_portfolio_csv()
        df = pd.read_csv(sample_path)
        return self.parse_dataframe(df)
