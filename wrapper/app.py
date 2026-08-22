"""RiskAdjusted Tactical Rebalancer (RATR) — Streamlit Dashboard.

Orchestrates the backend quantitative pipeline:
  MarketDataFeed  →  VolatilityModeler  →  SharpeOptimizer
  PortfolioParser →  AllocationRouter   →  Trade Execution Plan
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Ensure the project root is importable regardless of working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.ingestion import MarketDataFeed, MarketDataIngestor
from core.optimization import SharpeOptimizer
from core.volatility import VolatilityModeler
from wrapper.parser import PortfolioParser
from wrapper.router import AllocationRouter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock Nifty 50 universe (10 highly-liquid constituents for fast dev runs)
# ---------------------------------------------------------------------------
MOCK_NIFTY_UNIVERSE: List[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "BHARTIARTL.NS",
    "ITC.NS",
    "LTIM.NS",
    "KOTAKBANK.NS",
    "LT.NS",
]


# ═══════════════════════════════════════════════════════════════════════════
# Bloomberg Terminal CSS Injection
# ═══════════════════════════════════════════════════════════════════════════
def inject_terminal_css() -> None:
    """Inject Bloomberg Terminal dark-theme CSS overrides via st.markdown."""
    st.markdown(
        """
        <style>
        /* ── Global Terminal Override ───────────────────────────────── */
        * {
            font-family: 'Courier New', Courier, monospace !important;
            border-radius: 0px !important;
        }

        /* ── Restore Material Icon Ligatures ────────────────────────── */
        .material-symbols-rounded,
        .material-icons,
        span[class*="material-symbols"],
        span[class*="material-icons"],
        [data-testid="stSidebar"] .material-symbols-rounded,
        [data-testid="stHeader"] .material-symbols-rounded {
            font-family: 'Material Symbols Rounded', 'Material Icons' !important;
            font-style: normal;
            -webkit-font-feature-settings: 'liga';
            font-feature-settings: 'liga';
        }

        .stApp {
            background-color: #000000 !important;
            color: #FF9900 !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #000000 !important;
        }
        section[data-testid="stSidebar"] * {
            color: #FF9900 !important;
        }
        h1, h2, h3, h4, h5, h6, p, label, div {
            color: #FF9900 !important;
        }

        /* ── Sidebar Labels ────────────────────────────────────────── */
        section[data-testid="stSidebar"] label {
            font-size: 0.78rem !important;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            color: #FF9900 !important;
            margin-bottom: 0.15rem !important;
        }
        section[data-testid="stSidebar"] .stMarkdown h3 {
            border-bottom: 1px solid #333333;
            padding-bottom: 0.3rem;
            margin-bottom: 0.8rem !important;
        }

        /* ── Input Borders ─────────────────────────────────────────── */
        .stTextInput > div > div,
        .stSelectbox > div > div,
        .stNumberInput > div > div,
        div[data-testid="stDataFrame"],
        div[data-testid="stMetric"] {
            border: 1px solid #333333 !important;
            background-color: #000000 !important;
            border-radius: 0px !important;
        }

        /* ── File Uploader — Full Fix ──────────────────────────────── */
        .stFileUploader {
            margin-bottom: 0.8rem !important;
        }
        .stFileUploader > div,
        .stFileUploader > div > div,
        .stFileUploader > section,
        .stFileUploader label,
        section[data-testid="stSidebar"] .stFileUploader,
        section[data-testid="stSidebar"] .stFileUploader > div,
        section[data-testid="stSidebar"] .stFileUploader > div > div {
            border: none !important;
            background-color: #000000 !important;
            background: #000000 !important;
            padding: 0 !important;
        }
        [data-testid="stFileUploadDropzone"],
        section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] {
            border: 1px dashed #555555 !important;
            border-radius: 0px !important;
            background-color: #000000 !important;
            background: #000000 !important;
            padding: 0.6rem 0.8rem !important;
            min-height: 0 !important;
            overflow: visible !important;
        }
        [data-testid="stFileUploadDropzone"] *,
        section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] * {
            background-color: #000000 !important;
            background: #000000 !important;
        }
        /* Hide the broken Material Icon "upload" text inside dropzone */
        [data-testid="stFileUploadDropzone"] [data-testid="stIconMaterial"],
        [data-testid="stFileUploadDropzone"] span.material-symbols-rounded,
        [data-testid="stFileUploadDropzone"] span[class*="material-symbols"],
        [data-testid="stFileUploadDropzone"] span[class*="material-icons"] {
            display: none !important;
        }
        [data-testid="stFileUploadDropzone"] span,
        [data-testid="stFileUploadDropzone"] small,
        [data-testid="stFileUploadDropzone"] p,
        [data-testid="stFileUploadDropzone"] div {
            color: #888888 !important;
            font-size: 0.75rem !important;
        }
        [data-testid="stFileUploadDropzone"] button,
        section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] button {
            background-color: #000000 !important;
            background: #000000 !important;
            color: #FF9900 !important;
            border: 1px solid #FF9900 !important;
            font-size: 0.75rem !important;
            padding: 0.15rem 0.5rem !important;
        }

        /* ── Download Button (sidebar) ─────────────────────────────── */
        section[data-testid="stSidebar"] .stDownloadButton > button {
            background-color: #000000 !important;
            color: #FF9900 !important;
            border: 1px solid #FF9900 !important;
            font-size: 0.75rem !important;
            padding: 0.3rem 0.6rem !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            width: 100% !important;
        }
        section[data-testid="stSidebar"] .stDownloadButton > button:hover {
            background-color: #FF9900 !important;
            color: #000000 !important;
        }

        /* ── Sidebar Dividers ──────────────────────────────────────── */
        section[data-testid="stSidebar"] hr {
            border-color: #333333 !important;
            margin: 0.6rem 0 !important;
        }

        /* ── Metric elements ───────────────────────────────────────── */
        div[data-testid="stMetric"] {
            padding: 0.6rem 0.8rem;
        }
        div[data-testid="stMetric"] label {
            color: #888888 !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #FF9900 !important;
        }

        /* ── Tabs ──────────────────────────────────────────────────── */
        button[data-baseweb="tab"] {
            color: #888888 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #FF9900 !important;
            border-bottom: 2px solid #FF9900 !important;
        }

        /* ── Buttons — Pure White / Black ──────────────────────────── */
        .stButton > button {
            background-color: #FFFFFF !important;
            color: #000000 !important;
            border: 1px solid #FFFFFF !important;
            border-radius: 0px !important;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .stButton > button:hover,
        .stButton > button:focus,
        .stButton > button:active,
        .stButton > button:visited {
            background-color: #FFFFFF !important;
            color: #000000 !important;
            border: 1px solid #FFFFFF !important;
            outline: none !important;
            box-shadow: none !important;
        }

        /* ── Sidebar Toggle Button (Advanced Settings) ─────────────── */
        section[data-testid="stSidebar"] .stButton > button {
            background-color: #000000 !important;
            color: #FF9900 !important;
            border: 1px solid #333333 !important;
            font-weight: 600;
            font-size: 0.78rem !important;
            letter-spacing: 0.5px;
            padding: 0.35rem 0.6rem !important;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            border-color: #FF9900 !important;
        }

        /* ── Slider ────────────────────────────────────────────────── */
        section[data-testid="stSidebar"] .stSlider label {
            color: #FF9900 !important;
        }
        section[data-testid="stSidebar"] .stSlider div[data-testid="stTickBarMin"],
        section[data-testid="stSidebar"] .stSlider div[data-testid="stTickBarMax"] {
            color: #888888 !important;
        }

        /* ── Alert boxes ───────────────────────────────────────────── */
        div[data-testid="stAlert"] {
            background-color: #0a0a0a !important;
            border: 1px solid #333333 !important;
            color: #FF9900 !important;
            border-radius: 0px !important;
        }

        /* ── Expander (clean terminal look) ─────────────────────────── */
        details[data-testid="stExpander"] {
            border: 1px solid #333333 !important;
            background-color: #000000 !important;
            border-radius: 0px !important;
        }
        details[data-testid="stExpander"] summary {
            color: #FF9900 !important;
            font-size: 0.78rem !important;
            padding: 0.4rem 0.6rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Page / Theme Configuration
# ═══════════════════════════════════════════════════════════════════════════
def setup_page() -> None:
    """Configure Streamlit page metadata and inject terminal CSS."""
    st.set_page_config(
        page_title="RATR // BLOOMBERG TERMINAL",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load external CSS file
    css_file = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_file):
        with open(css_file, encoding="utf-8") as f:
            st.markdown(f"<style>\n{f.read()}\n</style>", unsafe_allow_html=True)

    # Inject inline terminal CSS overrides
    inject_terminal_css()



# ═══════════════════════════════════════════════════════════════════════════
# Helper — colour-coded trade-plan table (Bloomberg text-colour scheme)
# ═══════════════════════════════════════════════════════════════════════════
def style_trade_plan(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Return a Styler with Bloomberg terminal text colours for actions.

    BUY  → Neon Green (#00FF00)
    SELL → Pure Red   (#FF0000)
    HOLD → Gray       (#888888)
    Background: Black  (#000000), Default text: Amber (#FF9900)
    """

    def _row_colour(row: pd.Series) -> List[str]:
        action = str(row.get("Action", "")).upper()
        base = "background-color: #000000;"
        if action == "BUY":
            return [
                f"{base} color: #00FF00; font-weight: 600;"
            ] * len(row)
        if action == "SELL":
            return [
                f"{base} color: #FF0000; font-weight: 600;"
            ] * len(row)
        return [f"{base} color: #888888;"] * len(row)

    fmt = {}
    for col, spec in [
        ("Current_Price", "₹{:,.2f}"),
        ("Current_Value", "₹{:,.2f}"),
        ("Target_Value", "₹{:,.2f}"),
        ("Trade_Value_INR", "₹{:,.2f}"),
        ("Post_Trade_Value", "₹{:,.2f}"),
        ("Current_Weight", "{:.2%}"),
        ("Target_Weight", "{:.2%}"),
        ("Drift", "{:+.2%}"),
        ("Post_Trade_Weight", "{:.2%}"),
    ]:
        if col in df.columns:
            fmt[col] = spec

    return (
        df.style
        .apply(_row_colour, axis=1)
        .format(fmt)
        .set_properties(**{
            "background-color": "#000000",
            "color": "#FF9900",
            "font-family": "'Courier New', Courier, monospace",
        })
    )


# ═══════════════════════════════════════════════════════════════════════════
# Helper — weight-comparison bar chart (Bloomberg terminal palette)
# ═══════════════════════════════════════════════════════════════════════════
def render_weight_comparison(plan_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: Current vs Target vs Post-Trade weights."""
    chart = plan_df.copy()
    chart["Current %"] = chart["Current_Weight"] * 100.0
    chart["Target %"] = chart["Target_Weight"] * 100.0
    chart["Post-Trade %"] = chart["Post_Trade_Weight"] * 100.0

    fig = go.Figure()
    fig.add_trace(go.Bar(x=chart["Ticker"], y=chart["Current %"],
                         name="CURRENT", marker_color="#00FFFF"))
    fig.add_trace(go.Bar(x=chart["Ticker"], y=chart["Target %"],
                         name="TARGET [MAX SHARPE]", marker_color="#FF9900"))
    fig.add_trace(go.Bar(x=chart["Ticker"], y=chart["Post-Trade %"],
                         name="POST-TRADE", marker_color="#00FF00"))

    fig.update_layout(
        title="ALLOCATION: CURRENT / TARGET / POST-TRADE (%)",
        barmode="group",
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="#FF9900",
                  family="'Courier New', Courier, monospace"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11, color="#FF9900")),
        xaxis=dict(gridcolor="#222222", title="TICKER",
                   tickfont=dict(color="#FF9900")),
        yaxis=dict(gridcolor="#222222", title="WEIGHT (%)",
                   tickfont=dict(color="#FF9900")),
        margin=dict(l=40, r=20, t=60, b=40),
        height=420,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Helper — metric card HTML snippet
# ═══════════════════════════════════════════════════════════════════════════
def _card(label: str, value: str, badge_text: str, badge_cls: str) -> str:
    return (
        '<div class="metric-card">\n'
        f'    <div class="metric-label">{label}</div>\n'
        f'    <div class="metric-value">{value}</div>\n'
        f'    <span class="badge {badge_cls}">{badge_text}</span>\n'
        '</div>'
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    setup_page()

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="main-title">RISKADJUSTED TACTICAL REBALANCER [RATR]</div>'
        '<div class="sub-title">SYSTEM: GJR-GARCH ALLOCATION ROUTER // '
        "STATUS: ONLINE</div>",
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════
    # Sidebar — user inputs
    # ══════════════════════════════════════════════════════════════════════
    st.sidebar.markdown("### RATR CONFIGURATION")

    # 1. Portfolio file uploader
    uploaded_file = st.sidebar.file_uploader(
        "UPLOAD PORTFOLIO (CSV / EXCEL)",
        type=["csv", "xlsx", "xls"],
        help="Columns: Ticker, Quantity, Buy_Price, [Current_Price]. "
             "If omitted, an empty portfolio (initial cash deployment) is assumed.",
    )

    # 2. Sample template — direct download button (no expander)
    sample_parser = PortfolioParser()
    sample_csv_path = sample_parser.create_sample_portfolio_csv()
    with open(sample_csv_path, "r", encoding="utf-8") as fh:
        csv_bytes = fh.read().encode("utf-8")
    st.sidebar.download_button(
        label="DOWNLOAD SAMPLE TEMPLATE",
        data=csv_bytes,
        file_name="sample_portfolio.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.sidebar.markdown("---")

    # 3. Cash budget
    cash_budget = st.sidebar.number_input(
        "Fresh Cash Budget (INR)",
        min_value=0.0,
        value=50_000.0,
        step=5_000.0,
        format="%.2f",
        help="Additional capital available for deployment or SIP inflows.",
    )

    # 4. Rebalance mode
    rebalance_mode: str = st.sidebar.selectbox(
        "Rebalance Mode",
        options=["SIP_FIRST", "TACTICAL"],
        index=0,
        help=(
            "SIP_FIRST — deploys cash into under-allocated assets (zero sells). "
            "TACTICAL — buys & sells if drift exceeds threshold."
        ),
    )

    st.sidebar.markdown("---")

    # 5. Advanced settings — toggle button (not expander)
    if "show_advanced" not in st.session_state:
        st.session_state["show_advanced"] = False

    if st.sidebar.button(
        "ADVANCED RISK & MODEL SETTINGS",
        use_container_width=True,
    ):
        st.session_state["show_advanced"] = not st.session_state["show_advanced"]

    if st.session_state["show_advanced"]:
        drift_threshold_pct = st.sidebar.slider(
            "Tactical Drift Threshold (%)", 1.0, 15.0, 5.0, 0.5,
            help="Max acceptable drift before triggering tactical sell orders.",
        )
        drift_threshold = drift_threshold_pct / 100.0

        min_trade_inr = st.sidebar.number_input(
            "Min Trade Threshold (INR)", min_value=100.0,
            value=1000.0, step=250.0,
            help="Trades below this rupee amount are suppressed as HOLD.",
        )

        max_asset_cap = (
            st.sidebar.slider("Max Single-Asset Cap (%)", 10.0, 40.0, 25.0, 1.0) / 100.0
        )

        asset_tolerance_pct = st.sidebar.slider(
            "Asset Tolerance Band (%)", 0.0, 10.0, 2.5, 0.5,
            help=(
                "Per-asset weight deadband. Assets whose individual drift "
                "falls within this band are held steady (HOLD) to reduce "
                "micro-trade churn."
            ),
        )
        asset_tolerance = asset_tolerance_pct / 100.0

        lookback_period: str = st.sidebar.selectbox(
            "Historical Lookback", ["3y", "2y", "1y", "5y"], index=0,
            help="Window used to fit GJR-GARCH and estimate covariance.",
        )
    else:
        drift_threshold = 0.05
        min_trade_inr = 1000.0
        max_asset_cap = 0.25
        asset_tolerance = 0.025
        lookback_period = "3y"

    st.sidebar.markdown("---")

    # 6. Run button
    run_btn = st.sidebar.button(
        "EXECUTE OPTIMIZATION",
        type="primary",
        use_container_width=True,
    )

    # ══════════════════════════════════════════════════════════════════════
    # Execution pipeline (triggered on button click)
    # ══════════════════════════════════════════════════════════════════════
    if run_btn:
        try:
            # ── Step 1/4  Market Data Ingestion ───────────────────────────
            with st.spinner("STEP 1/4 — INGESTING OHLCV MARKET DATA ..."):
                feed = MarketDataFeed()
                df_prices, df_log_returns, _ = feed.fetch_ohlcv(
                    tickers=MOCK_NIFTY_UNIVERSE,
                    period=lookback_period,
                    interval="1d",
                )
                if df_prices.empty or df_log_returns.empty:
                    raise ValueError(
                        "Market data feed returned empty price / return series."
                    )
                latest_prices_series: pd.Series = df_prices.iloc[-1]

            # ── Step 2/4  GJR-GARCH Volatility Forecasts ─────────────────
            with st.spinner("STEP 2/4 — FITTING GJR-GARCH VOLATILITY MODELS ..."):
                vol_modeler = VolatilityModeler(dist="StudentsT")
                garch_vol_forecasts: Dict[str, float] = {}

                for ticker in MOCK_NIFTY_UNIVERSE:
                    if ticker in df_log_returns.columns:
                        series = df_log_returns[ticker].dropna()
                        try:
                            vol_forecast = vol_modeler.fit_predict_next_day(series)
                            if np.isnan(vol_forecast) or vol_forecast <= 0:
                                vol_forecast = float(series.std() * np.sqrt(252))
                        except Exception as garch_err:
                            logger.warning(
                                "GJR-GARCH failed for %s: %s — using sample σ.",
                                ticker, garch_err,
                            )
                            vol_forecast = float(series.std() * np.sqrt(252))
                        garch_vol_forecasts[ticker] = vol_forecast
                    else:
                        garch_vol_forecasts[ticker] = 0.20  # sensible default

            # ── Step 3/4  Max-Sharpe Optimisation ─────────────────────────
            with st.spinner("STEP 3/4 — SOLVING FOR MAX SHARPE TARGET WEIGHTS ..."):
                clean_returns = df_log_returns[MOCK_NIFTY_UNIVERSE].dropna()
                cov_matrix: pd.DataFrame = clean_returns.cov() * 252.0
                expected_returns: pd.Series = clean_returns.mean() * 252.0

                optimizer = SharpeOptimizer(
                    risk_free_rate=0.065, max_weight=max_asset_cap,
                )
                target_weights: pd.Series = optimizer.optimize(
                    expected_returns=expected_returns,
                    cov_matrix=cov_matrix,
                )

            # ── Step 4/4  Portfolio Parsing → Allocation Routing ──────────
            with st.spinner("STEP 4/4 — ROUTING ALLOCATION PLAN ..."):
                parser = PortfolioParser()

                if uploaded_file is not None:
                    parsed = parser.parse(
                        uploaded_file, latest_prices=latest_prices_series,
                    )
                    current_portfolio_df: pd.DataFrame = parsed["holdings_df"]
                    existing_val: float = parsed["total_market_value"]
                    portfolio_summary: Dict[str, Any] = parsed["summary"]
                else:
                    current_portfolio_df = pd.DataFrame(
                        columns=["Ticker", "Quantity", "Buy_Price", "Current_Price"],
                    )
                    existing_val = 0.0
                    portfolio_summary = {
                        "total_portfolio_value": 0.0,
                        "total_investment_cost": 0.0,
                        "total_pnl": 0.0,
                        "total_pnl_pct": 0.0,
                    }

                router = AllocationRouter()
                plan_df: pd.DataFrame = router.generate_allocation_plan(
                    current_portfolio=current_portfolio_df,
                    target_weights=target_weights,
                    cash_budget=cash_budget,
                    latest_prices=latest_prices_series,
                    mode=rebalance_mode,
                    drift_threshold=drift_threshold,
                    min_trade_inr=min_trade_inr,
                    asset_tolerance=asset_tolerance,
                )

            # ── Persist results in session state ──────────────────────────
            st.session_state.update(
                optimized=True,
                plan_df=plan_df,
                existing_val=existing_val,
                cash_budget=cash_budget,
                target_weights=target_weights,
                garch_vol_forecasts=garch_vol_forecasts,
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
                rebalance_mode=rebalance_mode,
                portfolio_summary=portfolio_summary,
                current_portfolio_df=current_portfolio_df,
            )

        except Exception as exc:
            logger.error("Optimization run failed: %s", exc, exc_info=True)
            st.error(f"FATAL: REBALANCER EXECUTION FAILED — {exc}")
            return

    # ══════════════════════════════════════════════════════════════════════
    # UI — Results Display
    # ══════════════════════════════════════════════════════════════════════
    if st.session_state.get("optimized", False):
        plan_df: pd.DataFrame = st.session_state["plan_df"]
        existing_val: float = st.session_state["existing_val"]
        budget: float = st.session_state["cash_budget"]
        total_capital: float = existing_val + budget
        residual_cash: float = plan_df.attrs.get("residual_cash", 0.0)
        mode_exec: str = plan_df.attrs.get("mode_executed", rebalance_mode)
        max_drift: float = plan_df.attrs.get("max_drift", 0.0)
        buy_orders: int = plan_df.attrs.get("buy_orders_count", 0)
        sell_orders: int = plan_df.attrs.get("sell_orders_count", 0)

        # ── Executive metric cards ────────────────────────────────────────
        st.markdown("### EXECUTIVE PORTFOLIO OVERVIEW")
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.markdown(
                _card("CURRENT VALUE", f"INR {existing_val:,.2f}",
                      "PRE-REBALANCE", "badge-blue"),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                _card("CASH BUDGET", f"INR {budget:,.2f}",
                      "FRESH INFLOW", "badge-purple"),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                _card("TOTAL CAPITAL", f"INR {total_capital:,.2f}",
                      "POST DEPLOYMENT", "badge-green"),
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                _card("RESIDUAL CASH", f"INR {residual_cash:,.2f}",
                      "UNALLOCATED FLOAT", "badge-blue"),
                unsafe_allow_html=True,
            )
        with c5:
            badge_cls = "badge-green" if mode_exec == "SIP_FIRST" else "badge-purple"
            st.markdown(
                _card("ACTIVE ROUTE", mode_exec,
                      f"MAX DRIFT: {max_drift:.1%}", badge_cls),
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tabbed analytical layout ──────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs([
            "TRADE EXECUTION PLAN",
            "ALLOCATION WEIGHT COMPARISON",
            "GJR-GARCH & RISK MATRIX",
            "HOLDINGS & P&L",
        ])

        # ▸ TAB 1 — Trade Execution Plan ──────────────────────────────────
        with tab1:
            st.subheader("ACTIONABLE TRADE EXECUTION INSTRUCTIONS")
            st.markdown(
                f"GENERATED **{buy_orders} BUY** AND **{sell_orders} SELL** "
                f"ORDERS // INTEGER-SHARE FLOORING // MICRO-TRADE FILTER "
                f"< INR {min_trade_inr:,.0f}"
            )

            st.dataframe(
                style_trade_plan(plan_df),
                use_container_width=True,
                height=380,
            )

            # Order summaries side-by-side
            trades_only = plan_df[plan_df["Action"].isin(["BUY", "SELL"])].copy()
            if not trades_only.empty:
                left, right = st.columns(2)
                with left:
                    st.markdown("#### BUY ORDERS")
                    buys = trades_only[trades_only["Action"] == "BUY"][
                        ["Ticker", "Trade_Units", "Current_Price", "Trade_Value_INR"]
                    ]
                    if buys.empty:
                        st.info("NO BUY ORDERS GENERATED.")
                    else:
                        st.dataframe(
                            buys.style
                            .format({
                                "Current_Price": "₹{:,.2f}",
                                "Trade_Value_INR": "₹{:,.2f}",
                            })
                            .set_properties(**{
                                "background-color": "#000000",
                                "color": "#00FF00",
                                "font-family": "'Courier New', Courier, monospace",
                            }),
                            use_container_width=True,
                            hide_index=True,
                        )
                with right:
                    st.markdown("#### SELL ORDERS")
                    sells = trades_only[trades_only["Action"] == "SELL"][
                        ["Ticker", "Trade_Units", "Current_Price", "Trade_Value_INR"]
                    ]
                    if sells.empty:
                        st.info("ZERO SELL ORDERS TRIGGERED.")
                    else:
                        st.dataframe(
                            sells.style
                            .format({
                                "Current_Price": "₹{:,.2f}",
                                "Trade_Value_INR": "₹{:,.2f}",
                            })
                            .set_properties(**{
                                "background-color": "#000000",
                                "color": "#FF0000",
                                "font-family": "'Courier New', Courier, monospace",
                            }),
                            use_container_width=True,
                            hide_index=True,
                        )

        # ▸ TAB 2 — Weight Comparison ──────────────────────────────────────
        with tab2:
            st.subheader("CURRENT VS OPTIMAL VS POST-TRADE WEIGHTS")
            st.plotly_chart(
                render_weight_comparison(plan_df), use_container_width=True,
            )

            col_w1, col_w2 = st.columns(2)
            with col_w1:
                st.markdown("#### TARGET PORTFOLIO [MAX SHARPE]")
                tw = st.session_state["target_weights"]
                df_tw = pd.DataFrame([
                    {"Ticker": k, "Weight (%)": round(v * 100, 2)}
                    for k, v in tw.items()
                ])
                fig_target = px.pie(
                    df_tw, names="Ticker", values="Weight (%)",
                    hole=0.45, title="OPTIMAL TARGET ALLOCATION",
                )
                fig_target.update_layout(
                    paper_bgcolor="black", plot_bgcolor="black",
                    font=dict(color="#FF9900",
                              family="'Courier New', Courier, monospace"),
                    margin=dict(t=40, b=20, l=20, r=20),
                )
                st.plotly_chart(fig_target, use_container_width=True)

            with col_w2:
                st.markdown("#### POST-TRADE COMPOSITION")
                post = plan_df[plan_df["Post_Trade_Weight"] > 0].copy()
                post["Weight (%)"] = (post["Post_Trade_Weight"] * 100).round(2)
                fig_post = px.pie(
                    post, names="Ticker", values="Weight (%)",
                    hole=0.45, title="EXPECTED POST-TRADE ALLOCATION",
                )
                fig_post.update_layout(
                    paper_bgcolor="black", plot_bgcolor="black",
                    font=dict(color="#FF9900",
                              family="'Courier New', Courier, monospace"),
                    margin=dict(t=40, b=20, l=20, r=20),
                )
                st.plotly_chart(fig_post, use_container_width=True)

        # ▸ TAB 3 — Volatility & Correlation ──────────────────────────────
        with tab3:
            st.subheader("GJR-GARCH(1,1,1) FORECASTS & COVARIANCE MATRIX")
            v1, v2 = st.columns(2)

            with v1:
                st.markdown("#### 1-STEP ANNUALISED VOLATILITY FORECASTS")
                vol_dict = st.session_state["garch_vol_forecasts"]
                er_dict = st.session_state["expected_returns"].to_dict()
                risk_df = pd.DataFrame([
                    {
                        "Ticker": t,
                        "E[Return] (ann.)": f"{er_dict.get(t, 0.0) * 100:.2f}%",
                        "GJR-GARCH σ (ann.)": f"{vol_dict.get(t, 0.0) * 100:.2f}%",
                        "Target Weight": f"{tw.get(t, 0.0) * 100:.2f}%",
                    }
                    for t in MOCK_NIFTY_UNIVERSE
                ])
                st.dataframe(risk_df, use_container_width=True, hide_index=True)

            with v2:
                st.markdown("#### ASSET CORRELATION HEATMAP")
                cov: pd.DataFrame = st.session_state["cov_matrix"]
                std = np.sqrt(np.diag(cov.values))
                corr_vals = cov.values / np.outer(std, std)
                df_corr = pd.DataFrame(
                    corr_vals, index=cov.index, columns=cov.columns,
                )
                fig_corr = px.imshow(
                    df_corr, text_auto=".2f",
                    color_continuous_scale=[[0, "#000000"], [0.5, "#FF9900"], [1, "#00FFFF"]],
                    title="NIFTY CONSTITUENTS — CORRELATION MATRIX",
                )
                fig_corr.update_layout(
                    paper_bgcolor="black", plot_bgcolor="black",
                    font=dict(color="#FF9900",
                              family="'Courier New', Courier, monospace"),
                    margin=dict(t=40, b=20, l=20, r=20),
                )
                st.plotly_chart(fig_corr, use_container_width=True)

        # ▸ TAB 4 — Uploaded Holdings & P&L ────────────────────────────────
        with tab4:
            st.subheader("UPLOADED PORTFOLIO HOLDINGS & P&L BREAKDOWN")
            curr_df: pd.DataFrame = st.session_state["current_portfolio_df"]

            if not curr_df.empty:
                s = st.session_state["portfolio_summary"]
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("INVESTMENT COST",
                          f"INR {s.get('total_investment_cost', 0.0):,.2f}")
                p2.metric("MARKET VALUE",
                          f"INR {s.get('total_portfolio_value', 0.0):,.2f}")
                p3.metric("UNREALISED P&L",
                          f"INR {s.get('total_pnl', 0.0):,.2f}")
                p4.metric("RETURN %",
                          f"{s.get('total_pnl_pct', 0.0):+.2f}%")
                st.dataframe(
                    curr_df, use_container_width=True, hide_index=True,
                )
            else:
                st.info(
                    "NO PORTFOLIO FILE UPLOADED — ALLOCATOR EXECUTED INITIAL "
                    "CASH DEPLOYMENT FROM EMPTY PORTFOLIO."
                )

    else:
        # ── Welcome / Landing Screen ──────────────────────────────────────
        st.info(
            "CONFIGURE PORTFOLIO FILE, CASH BUDGET, AND MODE IN THE "
            "SIDEBAR, THEN CLICK 'EXECUTE OPTIMIZATION'."
        )
        st.markdown(
            """
        ### SYSTEM MODULES & QUANTITATIVE PIPELINE
        1. **MARKET DATA FEED** — Ingests historical daily adjusted prices
           for liquid Nifty equities via yfinance.
        2. **GJR-GARCH VOLATILITY** — Captures asymmetric shocks & leverage
           effects to forecast conditional forward volatility.
        3. **SLSQP SHARPE OPTIMISATION** — Solves for maximum Sharpe ratio
           weights subject to concentration constraints.
        4. **SMART CAPITAL ROUTING**
           - *SIP_FIRST* — Proportional deployment into underweight assets;
             strictly zero sell orders.
           - *TACTICAL* — Evaluates drift vs threshold and triggers
             disciplined sell/buy orders to close the gap.
        5. **EXECUTION GUARDRAILS** — Integer share flooring, micro-trade
           suppression (< INR 1,000), and residual-cash tracking.
        """
        )


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
