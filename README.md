# Adaptive Indian Equity Allocation System

> **Quantitative Downside-Risk Forecasting & Algorithmic Portfolio Rebalancing Product (Nifty 50)**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![GJR-GARCH](https://img.shields.io/badge/Econometrics-GJR--GARCH-FF6B35?style=flat-square&logo=chartdotjs&logoColor=white)](https://arch.readthedocs.io/)
[![FinTech](https://img.shields.io/badge/Domain-FinTech%20Risk%20Analytics-00C49A?style=flat-square&logo=cashapp&logoColor=white)](https://github.com/name-shourya/Adaptive-Indian-Equity-Allocation-System)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=flat-square)]()
[![Live Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://adaptive-indian-equity-allocation-system-dwau67edyx97rhzyeghji.streamlit.app/)

> **Live Dashboard:** [adaptive-indian-equity-allocation-system-dwau67edyx97rhzyeghji.streamlit.app](https://adaptive-indian-equity-allocation-system-dwau67edyx97rhzyeghji.streamlit.app/) (or run locally via `streamlit run wrapper/app.py`)

---

## 1. Executive Problem Statement

### Why Standard Mean-Variance Models Fail in Real Markets

The classical Markowitz mean-variance framework rests on two fatal assumptions when applied to Indian equities:

1. **Symmetric Volatility:** It treats positive and negative return shocks identically, ignoring the well-documented **leverage effect** where negative news amplifies future volatility disproportionately.
2. **Constant Conditional Variance:** It assumes volatility is fixed over time, whereas Nifty 50 equities exhibit pronounced **volatility clustering** -- calm periods punctuated by sharp, self-reinforcing stress regimes (e.g. March 2020 COVID crash, 2022 rate-hike cycles, FII outflow shocks).

This means that during high-volatility regimes -- precisely when downside risk management matters most -- standard optimizers dramatically **underestimate tail risk** and produce portfolio weights dangerously concentrated in assets exhibiting leverage-driven drawdowns.

### Product Goal

**AIEAS** is a self-serve quantitative risk engine and Streamlit telemetry dashboard that:

- **Forecasts asymmetric downside risk** per Nifty 50 constituent using GJR-GARCH(1,1,1) -- capturing leverage effects that symmetric GARCH misses.
- **Solves for optimal portfolio weights** via constrained Max-Sharpe Ratio optimization (SLSQP), incorporating GJR-GARCH conditional covariance.
- **Generates actionable rebalancing instructions** (BUY/SELL/HOLD with integer-share flooring and INR trade values) via two modes: SIP cash deployment and Tactical drift-triggered rebalancing.
- **Minimizes capital drawdown** during market regime shifts by continuously recalibrating weights based on the latest conditional variance forecasts.
- **Runs as a zero-crash Streamlit dashboard** with self-healing cold-start logic, fully deployable on Streamlit Community Cloud.

---

## 2. System Architecture & Execution Pipeline

### Two-Stage Quantitative Product Architecture

```
+===========================================================================+
|       STAGE 1: BACKEND QUANTITATIVE PIPELINE (core/ + backtest/)         |
+===========================================================================+
|                                                                           |
|  +---------------------+   +---------------------------+   +-----------+ |
|  | Nifty 50 OHLCV      |-->| Asymmetric Volatility     |-->| Sharpe    | |
|  | Data Ingestion      |   | Engine (GJR-GARCH(1,1,1)) |   | Optimizer | |
|  | (yfinance + QA)     |   | * Leverage effect (gamma) |   | (SLSQP)   | |
|  | core/ingestion.py   |   | * Student-t innovations   |   | core/     | |
|  |                     |   | * EWMA fallback           |   | optim-    | |
|  | OUTPUT:             |   | core/volatility.py        |   | ization   | |
|  | data/processed/     |   |                           |   | .py       | |
|  | *.csv               |   | OUTPUT: sigma^2_t fcst    |   |           | |
|  +---------------------+   +---------------------------+   | OUTPUT:   | |
|                                                            | w* in R^N | |
|                                                            +-----------+ |
|  +--------------------------------------------------------------------+  |
|  | Walk-Forward Backtester  (backtest/walk_forward.py)                |  |
|  | 252-day train window -> 21-day rebalance -> 15bps transaction cost |  |
|  | vs Static 60/40 Benchmark & Buy-and-Hold Equity Benchmarks        |  |
|  +--------------------------------------------------------------------+  |
+===========================================================================+
|       STAGE 2: FRONTEND TELEMETRY PRODUCT (wrapper/)                     |
+===========================================================================+
|                                                                           |
|  +--------------------------------------------------------------------+  |
|  |          Streamlit Bloomberg Terminal Dashboard                     |  |
|  |                    wrapper/app.py                                   |  |
|  |                                                                     |  |
|  |  * Self-healing cold start (auto-runs pipeline if data missing)     |  |
|  |  * Live asset allocations (pie + grouped bar charts)                |  |
|  |  * GJR-GARCH volatility spike monitor per ticker                    |  |
|  |  * Automated BUY/SELL/HOLD rebalancing signal table                 |  |
|  |  * Correlation heatmap (Nifty constituent co-movement)              |  |
|  |  * P&L breakdown on uploaded user portfolio                         |  |
|  |                                                                     |  |
|  |  Portfolio CSV Input --> wrapper/parser.py --> wrapper/router.py    |  |
|  +--------------------------------------------------------------------+  |
+===========================================================================+
```

### Repository File Map

| File | Stage | Responsibility |
|------|-------|----------------|
| `core/ingestion.py` | Stage 1 | Ingests historical OHLCV data via yfinance for Nifty 50 constituents; computes log & simple daily returns; caches raw & processed CSVs; synthetic fallback if network unavailable |
| `core/volatility.py` | Stage 1 | Fits GJR-GARCH(1,1,1) model with Student-t innovations via `arch` package; returns annualized 1-step-ahead conditional variance forecasts; EWMA fallback |
| `core/optimization.py` | Stage 1 | Implements constrained Max-Sharpe Ratio (SLSQP) and Black-Litterman portfolio optimizers; returns optimal weight vector w* subject to sum-to-one and per-asset cap bounds |
| `core/sentiment.py` | Stage 1 | HuggingFace FinBERT-powered financial news sentiment scoring with exponential time-decay weighting; lexicon fallback for offline execution |
| `core/__init__.py` | Stage 1 | Package init -- exports MarketDataIngestor, VolatilityModeler, SharpeOptimizer, PortfolioOptimizer, SentimentEngine |
| `backtest/walk_forward.py` | Stage 1 | Rolling walk-forward backtesting engine: 252-day train window, 21-day rebalance cycle, 15bps transaction cost; compares Dynamic Tactical vs Static 60/40 vs Buy-and-Hold |
| `backtest/metrics.py` | Stage 1 | Computes performance metrics: annualized return, Sharpe ratio, max drawdown, Calmar ratio, VaR, CVaR |
| `wrapper/app.py` | Stage 2 | **Streamlit Bloomberg Terminal dashboard** -- self-healing cold start, 4-tab UI (trade plan, weight comparison, GJR-GARCH risk matrix, P&L), live BUY/SELL/HOLD signal generation |
| `wrapper/parser.py` | Stage 2 | Ingests user portfolio CSV/Excel; normalizes ticker formats; computes unrealized P&L, position weights, and health indicators |
| `wrapper/router.py` | Stage 2 | Allocates capital via SIP_FIRST (zero-sell cash deployment) or TACTICAL (drift-triggered full rebalance) modes; integer share flooring; micro-trade suppression |
| `wrapper/style.css` | Stage 2 | Bloomberg Terminal dark theme CSS -- amber-on-black color scheme, monospace typography |

---

## 3. Quantitative & Econometric Specification

### GJR-GARCH(1,1,1) Conditional Variance Model

The GJR-GARCH model -- developed by Glosten, Jagannathan & Runkle (1993) -- captures the **leverage effect**: negative return shocks generate larger future volatility spikes than positive shocks of equal magnitude.

$$\sigma_t^2 = \omega + (\alpha + \gamma I_{t-1}) \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

**Parameter Definitions:**

| Symbol | Definition |
|--------|------------|
| `sigma_t^2` | Conditional variance at time t -- the forward-looking risk forecast |
| `omega > 0` | Long-run variance intercept (unconditional variance floor) |
| `epsilon_{t-1}^2` | Squared innovation (unexpected return shock) from period t-1 |
| `alpha >= 0` | ARCH coefficient -- symmetric response to past squared innovations |
| `gamma >= 0` | **Leverage coefficient** -- extra variance loading triggered exclusively by negative shocks |
| `I_{t-1}` | **Asymmetric indicator:** I = 1 if epsilon_{t-1} < 0 (negative shock), else I = 0 |
| `beta >= 0` | GARCH coefficient -- persistence / memory of conditional variance |

**Persistence condition (covariance stationarity):** `alpha + gamma/2 + beta < 1`

**Implementation:** Estimated via MLE with Student-t error distribution to capture heavy tails. EWMA (lambda=0.94) serves as fallback if the MLE solver fails to converge.

---

### Constrained Sharpe Ratio Portfolio Optimization

$$\max_{w_t} \left( \frac{w_t^T \mu - r_f}{\sqrt{w_t^T \Sigma_t w_t}} \right) \quad \text{subject to} \quad \sum_{i=1}^{N} w_{i,t} = 1, \quad 0 \leq w_{i,t} \leq w_{\max}$$

**Variable Definitions:**

| Symbol | Definition |
|--------|------------|
| `w_t` in R^N | Portfolio weight vector (decision variable) at rebalance date t |
| `mu` in R^N | Annualized expected return vector (historical mean x 252 trading days) |
| `r_f` | Risk-free rate (91-day Indian T-Bill yield, default 6.5% p.a.) |
| `Sigma_t` (N x N) | Annualized return covariance matrix incorporating GJR-GARCH conditional variance |
| `w_max` | Maximum concentration cap per single asset (default 25%) |
| `N` | Number of assets in the investable universe |

**Solver:** Sequential Least Squares Programming (SLSQP) via `scipy.optimize.minimize`. Integer share flooring and an INR 1,000 minimum trade filter are applied post-optimization to produce executable order quantities.

---

## 4. Product UX & Telemetry Features

The Streamlit dashboard (`wrapper/app.py`) renders a **Bloomberg Terminal-styled black/amber interface** with the following live telemetry components:

### Sidebar Controls

- **Portfolio CSV Upload** -- accepts Ticker, Quantity, Buy_Price, Current_Price; auto-normalizes NSE/BSE ticker formats
- **Fresh Cash Budget (INR)** -- capital for SIP deployment
- **Rebalance Mode** -- `SIP_FIRST` (zero-sell cash deployment) vs `TACTICAL` (full drift-triggered rebalance)
- **Advanced Risk Settings** -- Drift Threshold %, Min Trade INR, Max Single-Asset Cap %, Historical Lookback Period
- **Re-run Pipeline Button** -- on-demand re-execution of the full backend quantitative pipeline with session state flush

### Executive KPI Banner (5 Metric Cards)

| Card | Metric |
|------|--------|
| CURRENT VALUE | Pre-rebalance portfolio market value (INR) |
| CASH BUDGET | Fresh inflow being deployed |
| TOTAL CAPITAL | Post-deployment total AUM |
| RESIDUAL CASH | Unallocated float after integer-share flooring |
| ACTIVE ROUTE | Executed mode (SIP_FIRST / TACTICAL) + Max Drift % |

### Tab 1 -- Trade Execution Plan

- Color-coded trade table: **BUY (neon green)**, **SELL (red)**, **HOLD (grey)**
- Columns: Ticker, Current Price, Current Weight, Target Weight, Drift %, Action, Trade Units, Trade Value (INR), Post-Trade Weight
- Split BUY/SELL order summary panels for immediate execution reference

### Tab 2 -- Allocation Weight Comparison

- **Grouped bar chart** -- Current vs Target (Max Sharpe) vs Post-Trade weights (%)
- **Optimal Portfolio Donut Chart** -- pie allocation of Max Sharpe target weights
- **Post-Trade Composition Donut Chart** -- expected allocation after executing all orders

### Tab 3 -- GJR-GARCH Risk Matrix

- **1-Step Annualised GJR-GARCH Volatility Table** -- per-ticker sigma forecasts alongside annualized expected return and target weight
- **Asset Correlation Heatmap** (Plotly imshow) -- Nifty constituent pairwise correlation matrix identifying diversification opportunities and concentration risk

### Tab 4 -- Holdings & P&L Dashboard

- Per-position P&L metrics: Investment Cost, Market Value, Unrealised P&L, Return %
- Full uploaded holdings DataFrame with auto-inferred current prices from yfinance

---

## 5. Local Installation & Step-by-Step Execution

### Prerequisites

- Python 3.10+ (3.11 recommended)
- pip

```bash
# Step 1: Clone and install dependencies
git clone https://github.com/name-shourya/Adaptive-Indian-Equity-Allocation-System.git
cd Adaptive-Indian-Equity-Allocation-System
pip install -r requirements.txt

# Step 2: (Optional) Pre-generate backend data artifacts
# The Streamlit dashboard self-heals on cold start -- this step is optional.
# Run this to explicitly pre-generate the processed return time-series:
python -c "
from core.ingestion import MarketDataFeed
feed = MarketDataFeed()
prices, log_returns, simple_returns = feed.fetch_ohlcv(
    tickers=['RELIANCE.NS','TCS.NS','HDFCBANK.NS','INFY.NS','ICICIBANK.NS',
             'BHARTIARTL.NS','ITC.NS','LTIM.NS','KOTAKBANK.NS','LT.NS'],
    period='3y'
)
print('Ingested', len(prices), 'trading days across', len(prices.columns), 'Nifty constituents.')
print('Data pipeline complete. Artifacts saved to data/processed/')
"

# Step 3: Launch the Streamlit Telemetry Dashboard
streamlit run wrapper/app.py
```

> **Note:** On first launch, the dashboard automatically detects missing data artifacts and runs the ingestion pipeline in the background. No manual pre-execution required.

### One-Click Streamlit Cloud Deployment

1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** -- select your fork -- set Main file path to `wrapper/app.py`
4. Click **Deploy** -- the self-healing cold start logic handles all data initialization automatically

---

## 6. Project Structure

```
Adaptive-Indian-Equity-Allocation-System/
|-- core/
|   |-- __init__.py          # Package exports
|   |-- ingestion.py         # Market data ingestion (yfinance + QA)
|   |-- volatility.py        # GJR-GARCH(1,1,1) volatility engine
|   |-- optimization.py      # Sharpe optimizer + Black-Litterman
|   `-- sentiment.py         # FinBERT NLP sentiment scoring
|-- wrapper/
|   |-- app.py               # Streamlit Bloomberg Terminal dashboard
|   |-- parser.py            # Portfolio CSV/Excel parser
|   |-- router.py            # SIP_FIRST / TACTICAL allocation router
|   `-- style.css            # Bloomberg dark theme CSS
|-- backtest/
|   |-- walk_forward.py      # Rolling walk-forward backtester
|   `-- metrics.py           # Performance metrics calculator
|-- data/
|   |-- raw/                 # Cached OHLCV price data (yfinance)
|   `-- processed/           # Computed return time-series CSVs
|-- sample_portfolio.csv     # Sample portfolio template
|-- requirements.txt         # Python dependency manifest
`-- README.md                # This file
```

---

## 7. Technical Highlights for TPM / RevOps Hiring Managers

| Dimension | Implementation |
|-----------|----------------|
| **Risk Model Sophistication** | GJR-GARCH(1,1,1) with Student-t innovations -- captures leverage effect and fat tails; EWMA fallback ensures zero crashes |
| **Optimizer Robustness** | SLSQP with NaN guard, infeasibility recovery, weight clipping, and sum-to-1 re-normalization |
| **Production Resilience** | Self-healing Streamlit cold start; synthetic price generation if yfinance unavailable; graceful degradation at every I/O boundary |
| **Execution Guardrails** | Integer share flooring, INR 1,000 micro-trade suppression, per-asset tolerance deadband to prevent churn |
| **Multi-Modal Deployment** | CLI Python invocation + Streamlit Cloud web app; all modules importable standalone |
| **Sentiment Integration** | FinBERT (HuggingFace) NLP pipeline with exponential time-decay weighting; lexicon fallback for air-gapped environments |
| **Backtesting Framework** | Walk-forward with realistic 15bps transaction costs vs Static 60/40 and Buy-and-Hold benchmarks |

---

## Author

**Shourya Singhania**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Shourya%20Singhania-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/shouryasinghania/)
[![Email](https://img.shields.io/badge/Email-shouryasinghania2017%40gmail.com-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:shouryasinghania2017@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-name--shourya-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/name-shourya)

---

*Built as a live Proof-of-Work portfolio asset demonstrating applied quantitative finance, econometric modeling, and production-grade Python engineering.*
