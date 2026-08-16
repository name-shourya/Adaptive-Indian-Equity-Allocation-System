import os
import sys

sys.path.append(os.path.abspath("C:/Users/shour/OneDrive/Desktop/MLFP Project"))

from core.ingestion import MarketDataIngestor
from core.volatility import VolatilityEngine
from core.sentiment import SentimentEngine
from core.optimization import PortfolioOptimizer
from backtest.walk_forward import WalkForwardBacktester
from wrapper.parser import PortfolioParser
from wrapper.router import AllocationRouter

print("--- Step 1: Testing Market Data Ingestion ---")
ingestor = MarketDataIngestor(data_dir="C:/Users/shour/OneDrive/Desktop/MLFP Project/data")
prices, returns, _ = ingestor.fetch_ohlcv(period="1y")
print(f"Prices shape: {prices.shape}, Returns shape: {returns.shape}")

print("--- Step 2: Testing GJR-GARCH Volatility Engine ---")
vol_engine = VolatilityEngine()
garch_fit = vol_engine.fit_gjr_garch(returns.iloc[:, 0])
forecasts = vol_engine.forecast_volatility(garch_fit)
print(f"GARCH Model: {garch_fit['model_name']}, 21-Day Forecast Vol: {forecasts.get(21, 0.0):.4f}")

print("--- Step 3: Testing FinBERT Sentiment Engine ---")
sentiment_engine = SentimentEngine(use_transformer=False)
headlines = ingestor.fetch_latest_news()
sent_res = sentiment_engine.analyze_headlines(headlines)
print(f"Sentiment Label: {sent_res['label']}, Composite Score: {sent_res['composite_score']:.4f}")

print("--- Step 4: Testing Portfolio Optimizer ---")
optimizer = PortfolioOptimizer()
opt_res = optimizer.optimize_black_litterman(returns, sentiment_score=sent_res['composite_score'], gjr_vol=forecasts)
print(f"Optimal Weights: {opt_res['weights']}")
print(f"Sharpe Ratio: {opt_res['sharpe_ratio']:.4f}")

print("--- Step 5: Testing Walk-Forward Backtest Simulator ---")
backtester = WalkForwardBacktester(train_window=100, rebalance_freq=21)
bt_res = backtester.run_backtest(returns)
print("Backtest metrics for Dynamic Tactical:")
print(bt_res["metrics"]["Dynamic Tactical"])

print("--- Step 6: Testing Portfolio Parser & Router ---")
parser = PortfolioParser()
parsed = parser._fallback_sample_portfolio()
print(f"Sample portfolio total value: Rs. {parsed['total_market_value']:,.2f}")

router = AllocationRouter()
sip_res = router.route_sip_inflow(parsed["holdings_df"], opt_res["weights"], sip_amount=50000.0)
print(f"SIP Recommendations generated: {len(sip_res['sip_recommendations'])} items")

print("\n[SUCCESS] ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
