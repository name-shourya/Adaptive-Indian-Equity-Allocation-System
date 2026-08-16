import sys
import os

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from core.ingestion import MarketDataIngestor
from core.volatility import VolatilityEngine
from core.sentiment import SentimentEngine
from core.optimization import PortfolioOptimizer
from backtest.walk_forward import WalkForwardBacktester
from wrapper.parser import PortfolioParser
from wrapper.router import AllocationRouter

# Streamlit Page Setup
st.set_page_config(
    page_title="Adaptive Indian Equity Allocation System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Executive Dark Aesthetic CSS
st.markdown("""
<style>
    /* Dark Theme Custom Palette */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #58a6ff 0%, #bc8cff 50%, #3fb950 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #8b949e;
        margin-bottom: 1.5rem;
    }
    
    /* Executive Metric Card */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8b949e;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #58a6ff;
    }
    .metric-delta {
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.2rem;
    }
    .delta-positive { color: #3fb950; }
    .delta-negative { color: #f85149; }
    
    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Streamlit Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border-radius: 8px;
        color: #8b949e;
        border: 1px solid #30363d;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important;
        color: #ffffff !important;
        border-color: #58a6ff !important;
    }
</style>
""", unsafe_allow_html=True)

# Cache Heavy Pipeline Engines
@st.cache_resource
def load_engines():
    ingestor = MarketDataIngestor()
    vol_engine = VolatilityEngine()
    sentiment_engine = SentimentEngine(use_transformer=False)  # Fast mode
    optimizer = PortfolioOptimizer()
    parser = PortfolioParser()
    router = AllocationRouter()
    return ingestor, vol_engine, sentiment_engine, optimizer, parser, router

ingestor, vol_engine, sentiment_engine, optimizer, parser, router = load_engines()

# Header Banner
st.markdown('<div class="main-header">Adaptive Indian Equity Allocation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Dynamic Quantitative Asset Allocation | GJR-GARCH Volatility | FinBERT Sentiment | Black-Litterman SLSQP</div>', unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/color/96/000000/line-chart.png", width=64)
st.sidebar.header("⚙️ Quantitative Parameters")

lookback_period = st.sidebar.selectbox("Market Data Lookback", ["1y", "3y", "5y"], index=1)
rf_rate = st.sidebar.slider("Risk-Free Rate (RBI Repo %)", min_value=4.0, max_value=9.0, value=6.5, step=0.25) / 100.0
sip_cash_inflow = st.sidebar.number_input("Monthly SIP Inflow (₹)", min_value=5000, max_value=1000000, value=50000, step=5000)

st.sidebar.markdown("---")
st.sidebar.subheader("Asset Allocation Constraints")
min_eq, max_eq = st.sidebar.slider("Equity Range (%)", 10, 90, (20, 80))
min_gold, max_gold = st.sidebar.slider("Gold Range (%)", 0, 40, (0, 25))
min_debt, max_debt = st.sidebar.slider("Debt Range (%)", 5, 60, (5, 40))

custom_bounds = {
    "Equity": (min_eq / 100.0, max_eq / 100.0),
    "Gold": (min_gold / 100.0, max_gold / 100.0),
    "Debt": (min_debt / 100.0, max_debt / 100.0),
}

# Fetch Market Data
@st.cache_data(ttl=3600)
def get_market_data(period):
    tickers = ["^NSEI", "GOLDBEES.NS", "LIQUIDBEES.NS"]
    prices, returns, _ = ingestor.fetch_ohlcv(tickers=tickers, period=period)
    returns.columns = ["Equity (Nifty 50)", "Gold ETF", "Debt / Liquid"]
    return prices, returns

prices, returns = get_market_data(lookback_period)

# Compute Current Volatility & Sentiment
equity_returns = returns.iloc[:, 0]
garch_fit = vol_engine.fit_gjr_garch(equity_returns)
vol_forecasts = vol_engine.forecast_volatility(garch_fit)

headlines = ingestor.fetch_latest_news()
sentiment_res = sentiment_engine.analyze_headlines(headlines)

# Black-Litterman Optimal Allocation
optimizer.rf = rf_rate
opt_res = optimizer.optimize_black_litterman(
    returns=returns,
    sentiment_score=sentiment_res["composite_score"],
    gjr_vol=vol_forecasts,
    custom_bounds=custom_bounds
)

target_weights = opt_res["weights"]

# Executive Metric Banner Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    ann_vol_21d = vol_forecasts.get(21, 0.15) * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">GJR-GARCH Volatility (21D)</div>
        <div class="metric-value">{ann_vol_21d:.2f}%</div>
        <div class="metric-delta delta-{'negative' if ann_vol_21d > 18 else 'positive'}">
            Model: {garch_fit['model_name']}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    sent_score = sentiment_res["composite_score"]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">FinBERT Market Sentiment</div>
        <div class="metric-value">{sentiment_res['label']}</div>
        <div class="metric-delta delta-{'positive' if sent_score > 0 else 'negative'}">
            Tilt Score: {sent_score:+.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    target_eq_pct = target_weights.get("Equity (Nifty 50)", 0.60) * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Optimal Equity Target</div>
        <div class="metric-value">{target_eq_pct:.1f}%</div>
        <div class="metric-delta delta-positive">Black-Litterman Max-Sharpe</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    opt_sharpe = opt_res["sharpe_ratio"]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Expected Sharpe Ratio</div>
        <div class="metric-value">{opt_sharpe:.2f}</div>
        <div class="metric-delta delta-positive">Rf = {rf_rate*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Dashboard Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Optimal Asset Allocation",
    "⚡ GJR-GARCH Volatility Engine",
    "📰 FinBERT Sentiment Radar",
    "🔄 Portfolio Rebalancer & SIP",
    "🧪 Walk-Forward Backtest Simulator"
])

# TAB 1: Optimal Asset Allocation
with tab1:
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.subheader("Target Asset Weights")
        df_weights = pd.DataFrame([
            {"Asset Class": k, "Optimal Weight (%)": round(v * 100, 2)}
            for k, v in target_weights.items()
        ])
        
        fig_donut = px.pie(
            df_weights,
            names="Asset Class",
            values="Optimal Weight (%)",
            hole=0.5,
            color_discrete_sequence=["#1f6feb", "#f1e05a", "#238636"]
        )
        fig_donut.update_layout(
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            font=dict(color="#c9d1d9"),
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_b:
        st.subheader("Expected Return & Risk Metrics")
        st.dataframe(
            pd.DataFrame({
                "Parameter": ["Expected Portfolio Return", "Expected Portfolio Volatility", "Optimal Sharpe Ratio", "Sentiment Tilt Applied"],
                "Value": [
                    f"{opt_res['expected_return']*100:.2f}%",
                    f"{opt_res['portfolio_volatility']*100:.2f}%",
                    f"{opt_res['sharpe_ratio']:.3f}",
                    f"{opt_res.get('sentiment_tilt_applied', 0.0)*100:+.2f}%"
                ]
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.subheader("Implied vs Black-Litterman Posterior Returns")
        df_returns_cmp = pd.DataFrame({
            "Market Implied Eq (\u03a0) %": [round(v*100, 2) for v in opt_res.get("implied_equilibrium_returns", {}).values()],
            "Black-Litterman (\u03bc_BL) %": [round(v*100, 2) for v in opt_res.get("black_litterman_returns", {}).values()]
        }, index=list(target_weights.keys()))
        st.dataframe(df_returns_cmp, use_container_width=True)

# TAB 2: GJR-GARCH Volatility Engine
with tab2:
    st.subheader("GJR-GARCH(1,1,1) Asymmetric Volatility Forecasts")
    
    col_v1, col_v2 = st.columns([1, 2])
    with col_v1:
        st.markdown(f"**Model Status**: `{garch_fit['model_name']}`")
        st.markdown(r"**$\omega$ (Omega)**: `" + f"{garch_fit['omega']:.6f}`")
        st.markdown(r"**$\alpha$ (Alpha - Shock impact)**: `" + f"{garch_fit['alpha']:.4f}`")
        st.markdown(r"**$\gamma$ (Gamma - Asymmetry/Leverage)**: `" + f"{garch_fit['gamma']:.4f}`")
        st.markdown(r"**$\beta$ (Beta - Vol Persistence)**: `" + f"{garch_fit['beta']:.4f}`")
        st.markdown(f"**Persistence Total**: `{garch_fit['persistence']:.4f}`")
        
        st.subheader("Forecast Horizons")
        st.json({f"{h}-Day Forecast": f"{v*100:.2f}%" for h, v in vol_forecasts.items()})

    with col_v2:
        if "annualized_volatility" in garch_fit:
            vol_series = garch_fit["annualized_volatility"]
            fig_vol = px.line(
                vol_series,
                title="Historical Conditional Volatility Time Series (% Annualized)",
                labels={"value": "Volatility %", "index": "Date"}
            )
            fig_vol.update_traces(line_color="#58a6ff")
            fig_vol.update_layout(
                paper_bgcolor="#161b22",
                plot_bgcolor="#161b22",
                font=dict(color="#c9d1d9"),
                showlegend=False
            )
            st.plotly_chart(fig_vol, use_container_width=True)

# TAB 3: FinBERT Sentiment Radar
with tab3:
    st.subheader("FinBERT News Headlines Sentiment Analysis")
    
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        st.markdown(f"**NLP Engine**: `{sentiment_res['engine']}`")
        st.markdown(f"**Headline Count Analyzed**: `{sentiment_res['headline_count']}`")
        st.markdown(f"**Composite Sentiment Score**: `{sentiment_res['composite_score']:+.4f}`")
        
        # Gauge Bar
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment_res['composite_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Market Sentiment Score [-1 to +1]"},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "#58a6ff"},
                'steps': [
                    {'range': [-1, -0.15], 'color': "#f85149"},
                    {'range': [-0.15, 0.15], 'color': "#8b949e"},
                    {'range': [0.15, 1], 'color': "#3fb950"}
                ]
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="#161b22", font=dict(color="#c9d1d9"), height=250)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_s2:
        st.subheader("Scraped Financial News Feed")
        df_news = pd.DataFrame(sentiment_res["details"])
        if not df_news.empty:
            st.dataframe(
                df_news[["text", "positive", "negative", "net_score"]],
                use_container_width=True,
                hide_index=True
            )

# TAB 4: Portfolio Rebalancer & SIP Router
with tab4:
    st.subheader("Upload User Portfolio CSV / Excel State")
    
    uploaded_file = st.file_uploader("Upload portfolio file (CSV or Excel)", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        parsed_portfolio = parser.parse_portfolio_file(uploaded_file)
    else:
        st.info("No file uploaded. Displaying default sample Indian equity portfolio.")
        parsed_portfolio = parser._fallback_sample_portfolio()

    st.markdown("### Portfolio Current Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Market Value", f"₹{parsed_portfolio['total_market_value']:,.2f}")
    c2.metric("Total Investment Cost", f"₹{parsed_portfolio['total_cost']:,.2f}")
    c3.metric("Unrealized P&L", f"₹{parsed_portfolio['total_pnl']:,.2f}")
    c4.metric("Unrealized Return %", f"{parsed_portfolio['total_pnl_pct']:+.2f}%")

    st.dataframe(parsed_portfolio["holdings_df"], use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Smart Rebalancing Execution Router")
    
    mode = st.radio("Select Execution Router Mode", ["SIP Inflow Distribution", "Tactical Rebalance (Buy/Sell Trades)"])
    
    if mode == "SIP Inflow Distribution":
        st.markdown(f"**Deploying Fresh Monthly Cash Inflow**: ₹{sip_cash_inflow:,.2f}")
        sip_res = router.route_sip_inflow(
            current_holdings=parsed_portfolio["holdings_df"],
            target_weights=target_weights,
            sip_amount=sip_cash_inflow
        )
        st.dataframe(sip_res["sip_recommendations"], use_container_width=True, hide_index=True)
    else:
        rebalance_res = router.route_tactical_rebalance(
            current_holdings=parsed_portfolio["holdings_df"],
            target_weights=target_weights
        )
        st.dataframe(rebalance_res["trade_recommendations"], use_container_width=True, hide_index=True)

# TAB 5: Walk-Forward Backtest Simulator
with tab5:
    st.subheader("Rolling Window Walk-Forward Strategy Simulator")
    
    if st.button("🚀 Run Walk-Forward Backtest Simulation"):
        with st.spinner("Executing rolling window backtest & computing quantitative metrics..."):
            backtester = WalkForwardBacktester(train_window=252, rebalance_freq=21, risk_free_rate=rf_rate)
            bt_results = backtester.run_backtest(returns)

            st.success("Backtest simulation completed!")

            # Plot Equity Curves
            fig_bt = px.line(
                bt_results["equity_curves"],
                title="Historical Cumulative Portfolio Growth (INR ₹100,000 Base)",
                labels={"value": "Portfolio Value (₹)", "index": "Date"}
            )
            fig_bt.update_layout(
                paper_bgcolor="#161b22",
                plot_bgcolor="#161b22",
                font=dict(color="#c9d1d9")
            )
            st.plotly_chart(fig_bt, use_container_width=True)

            # Compare Metrics Table
            st.subheader("Quantitative Metrics Comparison Table")
            df_metrics = pd.DataFrame(bt_results["metrics"]).T
            st.dataframe(df_metrics, use_container_width=True)
