# Adaptive Indian Equity Allocation System (`nifty_dynamic_allocator`)

[![GitHub Repository](https://img.shields.io/badge/GitHub-Adaptive--Indian--Equity--Allocation--System-blue?logo=github)](https://github.com/name-shourya/Adaptive-Indian-Equity-Allocation-System)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end quantitative dynamic asset allocation system designed for Indian equity portfolios (Nifty 50, Nifty Gold ETF, Debt/G-Sec, and Cash). The system integrates **GJR-GARCH asymmetric volatility modeling**, **FinBERT financial sentiment scoring**, **Black-Litterman subjective view incorporation**, **Max-Sharpe SLSQP portfolio optimization**, **Walk-Forward rolling backtests**, and a **Streamlit visual dashboard**.

---

## 🏛 Directory Architecture

```
nifty_dynamic_allocator/
├── .rules                       # Quantitative rules & asset allocation boundary constraints
├── requirements.txt             # Project dependencies (pandas, arch, transformers, yfinance, etc.)
├── .gitignore                   # Git ignore policies
├── README.md                    # System documentation
├── data/                        # Local file system cache
│   ├── raw/                     # Cached raw market price & headline CSVs
│   └── processed/               # Cleaned returns & sentiment score time series
├── core/                        # Layer 1: ML & Quantitative Engine
│   ├── __init__.py
│   ├── ingestion.py             # Data Ingestion (OHLCV via yfinance + BeautifulSoup News Scraping)
│   ├── volatility.py            # GJR-GARCH(1,1,1) asymmetric volatility modeling & forecasting
│   ├── sentiment.py             # FinBERT NLP tokenizer, sentiment scorer & view matrix generator
│   └── optimization.py          # Black-Litterman & SLSQP Max-Sharpe portfolio solver
├── backtest/                    # Layer 1: Validation Engine
│   ├── __init__.py
│   ├── walk_forward.py          # Rolling-window dynamic rebalancing simulation engine
│   └── metrics.py               # Quantitative performance metrics (CAGR, Sharpe, Sortino, VaR)
└── wrapper/                     # Layer 2: Product & Presentation Layer
    ├── __init__.py
    ├── parser.py                # CSV/Excel user portfolio ingestion & schema validator
    ├── router.py                # Tactical Rebalancing vs SIP cash inflow distribution router
    └── app.py                   # Executive Streamlit Web UI Dashboard
```

---

## 📐 Mathematical Foundations

### 1. GJR-GARCH Volatility Modeling
Standard GARCH models fail to capture leverage effects—where negative market shocks trigger higher volatility than positive shocks of equal magnitude. We model conditional variance $\sigma_t^2$ using **GJR-GARCH(1,1,1)**:

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \gamma \epsilon_{t-1}^2 I_{t-1} + \beta \sigma_{t-1}^2$$

Where $I_{t-1} = 1$ if $\epsilon_{t-1} < 0$ and $0$ otherwise. If $\gamma > 0$, negative shocks increase volatility asymmetric to positive market movement.

### 2. FinBERT Sentiment Views
Financial news headlines are processed using **FinBERT** (`ProsusAI/finbert` / `yiyanghkust/finbert-tone`). The model generates probabilities for Positive ($p_+$), Negative ($p_-$), and Neutral ($p_n$) sentiments.

The composite sentiment score $S \in [-1, +1]$:
$$S = p_+ - p_-$$

This score tilts market equilibrium returns $\Pi$ to formulate Black-Litterman views vector $Q$:
$$Q = \Pi \cdot (1 + \delta \cdot S)$$

### 3. Black-Litterman Asset Allocation
Implied market equilibrium returns $\Pi$ are derived from risk aversion coefficient $\lambda$ and benchmark covariance $\Sigma$:
$$\Pi = \lambda \Sigma w_{\text{mkt}}$$

Combined Black-Litterman posterior expected returns $\mu_{BL}$ and covariance $\Sigma_{BL}$:
$$\mu_{BL} = [(\tau \Sigma)^{-1} + P^T \Omega^{-1} P]^{-1} [(\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q]$$

Subjective weights $w^*$ are optimized via Sequential Least Squares Programming (SLSQP):
$$\max_w \frac{w^T \mu_{BL} - r_f}{\sqrt{w^T \Sigma_{BL} w}} \quad \text{s.t.} \quad \sum w_i = 1, \quad l_i \le w_i \le u_i$$

---

## 🚀 Quick Start Guide

### 1. Clone & Setup Environment
```bash
git clone https://github.com/name-shourya/Adaptive-Indian-Equity-Allocation-System.git
cd Adaptive-Indian-Equity-Allocation-System

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Streamlit Dashboard
```bash
streamlit run wrapper/app.py
```

### 3. Python API Usage Example
```python
from core.ingestion import MarketDataIngestor
from core.volatility import VolatilityEngine
from core.sentiment import SentimentEngine
from core.optimization import PortfolioOptimizer

# 1. Fetch Indian Equity Data
ingestor = MarketDataIngestor()
prices, returns = ingestor.fetch_ohlcv(tickers=['^NSEI', 'GOLDBEES.NS', 'LIQUIDBEES.NS'], period='3y')

# 2. Fit GJR-GARCH Volatility
vol_engine = VolatilityEngine()
garch_res = vol_engine.fit_gjr_garch(returns['^NSEI'])
forecast_vol = vol_engine.forecast_volatility(garch_res, horizon=21)

# 3. Compute FinBERT Sentiment
sentiment_engine = SentimentEngine()
headlines = ingestor.fetch_latest_news()
sentiment_res = sentiment_engine.analyze_headlines(headlines)

# 4. Optimize Black-Litterman Allocation
optimizer = PortfolioOptimizer()
optimal_weights = optimizer.optimize_black_litterman(
    returns=returns,
    sentiment_score=sentiment_res['composite_score'],
    gjr_vol=forecast_vol
)

print("Target Asset Allocation:", optimal_weights)
```

---

## 📊 Performance Metrics

The backtest suite evaluates performance across standard quantitative metrics:
- **CAGR** (Compound Annual Growth Rate)
- **Annualized Volatility**
- **Sharpe Ratio** ($r_f = 6.0\%$)
- **Sortino Ratio** (Downside risk-adjusted)
- **Maximum Drawdown** (Peak-to-trough drop)
- **Value at Risk** (95% Historical VaR)

---

## 🔗 Repository & GitHub Integration
- **GitHub Repository**: [https://github.com/name-shourya/Adaptive-Indian-Equity-Allocation-System](https://github.com/name-shourya/Adaptive-Indian-Equity-Allocation-System)
- **Author**: Shourya
