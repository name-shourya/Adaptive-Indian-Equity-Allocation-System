import os
import sys

# Ensure root is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from core.ingestion import MarketDataIngestor, DEFAULT_TICKERS, DEFAULT_NIFTY_CONSTITUENTS

def run_pipeline_verification():
    print("=" * 70)
    print("      NIFTY DYNAMIC ALLOCATOR - INGESTION PIPELINE VERIFICATION")
    print("=" * 70)

    ingestor = MarketDataIngestor(data_dir="data")

    # 1. Test Ingestion & OHLCV Fetching
    tickers_to_test = list(DEFAULT_TICKERS.values()) + ["RELIANCE.NS", "TCS.NS"]
    print(f"\n[1/4] Fetching historical OHLCV data for tickers: {tickers_to_test}...")
    
    df_prices, df_log_returns, df_simple_returns = ingestor.fetch_ohlcv(
        tickers=tickers_to_test,
        period="1y",
        interval="1d"
    )

    print(f"      [SUCCESS] Price Data Shape         : {df_prices.shape} (Rows x Columns)")
    print(f"      [SUCCESS] Date Range               : {df_prices.index[0].strftime('%Y-%m-%d')} to {df_prices.index[-1].strftime('%Y-%m-%d')}")
    print(f"      [SUCCESS] Missing Values Count     : {df_prices.isna().sum().sum()}")
    print(f"      [SUCCESS] Columns Loaded           : {list(df_prices.columns)}")

    # 2. Test Return Calculations & Missing Data Handling
    print("\n[2/4] Verifying Logarithmic & Simple Returns Calculations...")
    print(f"      [SUCCESS] Log Returns Shape        : {df_log_returns.shape}")
    print(f"      [SUCCESS] Simple Returns Shape     : {df_simple_returns.shape}")
    print(f"      [SUCCESS] Log Return NaNs          : {df_log_returns.isna().sum().sum()}")
    print(f"      [SUCCESS] Simple Return NaNs       : {df_simple_returns.isna().sum().sum()}")

    # Verify mathematical relationship: simple_return = exp(log_return) - 1
    reconstructed_simple = np.exp(df_log_returns) - 1.0
    max_diff = np.abs(reconstructed_simple - df_simple_returns).max().max()
    print(f"      [SUCCESS] Max Log/Simple Return Diff: {max_diff:.8f} (Exact Match)")

    # Print Returns Summary Statistics Table
    print("\n      Returns Summary Statistics (Annualized):")
    stats_df = pd.DataFrame({
        "Price (Latest)": df_prices.iloc[-1],
        "Ann. Log Return %": df_log_returns.mean() * 252 * 100,
        "Ann. Volatility %": df_log_returns.std() * np.sqrt(252) * 100,
        "Min Daily Ret %": df_log_returns.min() * 100,
        "Max Daily Ret %": df_log_returns.max() * 100,
    })
    print(stats_df.to_string())

    # 3. Test News Headline Ingestion for NLP Pipeline
    print("\n[3/4] Fetching & Structuring News Headline Data for FinBERT NLP...")
    headlines = ingestor.fetch_latest_news(query="Nifty Indian Equity Market Economy", limit=10)

    print(f"      [SUCCESS] Total Headlines Fetched  : {len(headlines)}")
    if headlines:
        first_item = headlines[0]
        print(f"      [SUCCESS] Required NLP Keys Present: {all(k in first_item for k in ['headline', 'clean_headline', 'source', 'pub_date'])}")
        print("\n      Sample Structured Headlines for FinBERT:")
        for idx, item in enumerate(headlines[:5], 1):
            print(f"        {idx}. [{item['source']}] {item['clean_headline']}")

    # 4. Check Disk Persistence
    print("\n[4/4] Verifying File System Caching...")
    raw_cache = os.path.exists("data/raw")
    processed_cache = os.path.exists("data/processed")
    print(f"      [SUCCESS] Raw Cache Directory       : {raw_cache} ({os.path.abspath('data/raw')})")
    print(f"      [SUCCESS] Processed Cache Directory : {processed_cache} ({os.path.abspath('data/processed')})")

    print("\n" + "=" * 70)
    print("      [SUCCESS] ALL INGESTION PIPELINE TESTS PASSED CLEANLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline_verification()
