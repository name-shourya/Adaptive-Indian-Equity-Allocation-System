import os
import logging
import datetime
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

import yfinance as yf
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TICKERS = {
    "Equity": "^NSEI",          # Nifty 50 Index
    "Gold": "GOLDBEES.NS",       # Gold ETF
    "Debt": "LIQUIDBEES.NS",     # Liquid / Cash ETF
}

DEFAULT_NIFTY_CONSTITUENTS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "BHARTIARTL.NS", "ITC.NS", "LTIM.NS", "KOTAKBANK.NS", "LT.NS"
]

class MarketDataIngestor:
    """
    Handles ingestion of historical daily adjusted OHLCV market price data via yfinance,
    automatic return calculations (logarithmic & simple returns with missing data handling),
    and structured news headline scraping for the NLP FinBERT pipeline.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def fetch_ohlcv(
        self,
        tickers: Optional[List[str]] = None,
        period: str = "3y",
        interval: str = "1d",
        force_refresh: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fetches historical price data via yfinance and computes daily log & simple returns.
        
        Parameters:
            tickers: List of Yahoo Finance ticker symbols (defaults to Nifty 50, Gold, Debt)
            period: Lookback period ('1y', '3y', '5y')
            interval: Data sampling frequency ('1d')
            force_refresh: Force fresh download ignoring local disk cache
            
        Returns:
            Tuple of (df_prices, df_log_returns, df_simple_returns)
        """
        if tickers is None:
            tickers = list(DEFAULT_TICKERS.values())

        raw_cache_path = os.path.join(
            self.raw_dir, f"prices_{'_'.join([t.replace('^', '') for t in tickers])}_{period}.csv"
        )

        if os.path.exists(raw_cache_path) and not force_refresh:
            logger.info(f"Loading cached market data from {raw_cache_path}")
            df_prices = pd.read_csv(raw_cache_path, index_col=0, parse_dates=True)
        else:
            if yf is not None:
                logger.info(f"Downloading historical market OHLCV data for tickers: {tickers}")
                try:
                    data = yf.download(tickers, period=period, interval=interval, progress=False)
                    if data is None or data.empty:
                        logger.warning("yfinance returned no data. Generating synthetic baseline prices.")
                        df_prices = self._generate_synthetic_prices(tickers, period)
                    elif isinstance(data.columns, pd.MultiIndex):
                        if 'Adj Close' in data.columns.levels[0]:
                            df_prices = data['Adj Close']
                        elif 'Close' in data.columns.levels[0]:
                            df_prices = data['Close']
                        else:
                            df_prices = data

                        if isinstance(df_prices, pd.Series):
                            df_prices = df_prices.to_frame(name=tickers[0])

                        # Drop any columns that are entirely NaN before row-level dropna
                        df_prices = df_prices.dropna(axis=1, how="all")
                        if df_prices.empty or len(df_prices) < 20:
                            logger.warning("yfinance returned insufficient data rows. Generating synthetic baseline prices.")
                            df_prices = self._generate_synthetic_prices(tickers, period)
                        else:
                            # If some tickers failed, fill missing ticker columns with synthetic series
                            for t in tickers:
                                if t not in df_prices.columns:
                                    syn = self._generate_synthetic_prices([t], period)
                                    df_prices[t] = syn[t].reindex(df_prices.index).ffill().bfill()
                    else:
                        df_prices = data[['Close']] if 'Close' in data.columns else data
                        df_prices.columns = tickers

                        if isinstance(df_prices, pd.Series):
                            df_prices = df_prices.to_frame(name=tickers[0])

                        # Drop any columns that are entirely NaN before row-level dropna
                        df_prices = df_prices.dropna(axis=1, how="all")
                        if df_prices.empty or len(df_prices) < 20:
                            logger.warning("yfinance returned insufficient data rows. Generating synthetic baseline prices.")
                            df_prices = self._generate_synthetic_prices(tickers, period)
                        else:
                            for t in tickers:
                                if t not in df_prices.columns:
                                    syn = self._generate_synthetic_prices([t], period)
                                    df_prices[t] = syn[t].reindex(df_prices.index).ffill().bfill()

                    df_prices = self._clean_missing_prices(df_prices)
                    df_prices.to_csv(raw_cache_path)
                except Exception as e:
                    logger.error(f"Error downloading data from yfinance: {e}")
                    df_prices = self._generate_synthetic_prices(tickers, period)
            else:
                logger.info("yfinance package not installed. Generating synthetic baseline market data.")
                df_prices = self._generate_synthetic_prices(tickers, period)

        # Clean missing values
        df_prices = self._clean_missing_prices(df_prices)
        if df_prices.empty or len(df_prices) < 20:
            df_prices = self._generate_synthetic_prices(tickers, period)

        # Calculate returns
        log_returns, simple_returns = self.calculate_returns(df_prices)

        # Cache processed return time-series
        log_returns.to_csv(os.path.join(self.processed_dir, "processed_log_returns.csv"))
        simple_returns.to_csv(os.path.join(self.processed_dir, "processed_simple_returns.csv"))

        return df_prices, log_returns, simple_returns

    def calculate_returns(self, df_prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculates logarithmic returns r_log = ln(P_t / P_{t-1})
        and simple returns r_simple = (P_t - P_{t-1}) / P_{t-1}
        with automatic missing-data handling.
        """
        clean_prices = self._clean_missing_prices(df_prices)

        # 1. Logarithmic Returns
        ratio = clean_prices / clean_prices.shift(1)
        log_returns = pd.DataFrame(
            np.log(ratio.values),  # type: ignore[arg-type]
            index=ratio.index,
            columns=ratio.columns,
        )
        log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna()

        # 2. Simple Returns
        simple_returns = clean_prices.pct_change()
        simple_returns = simple_returns.replace([np.inf, -np.inf], np.nan).dropna()

        return log_returns, simple_returns

    def _clean_missing_prices(self, df_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Automatic missing-data handling:
        Drop all-NaN columns -> Forward fill -> Backward fill -> Drop remaining NaN rows.
        """
        if df_prices.empty:
            return df_prices
        df_clean = df_prices.copy()
        df_clean = df_clean.replace(0.0, np.nan)
        # Drop columns with all NaNs
        df_clean = df_clean.dropna(axis=1, how="all")
        # Forward fill then backward fill
        df_clean = df_clean.ffill().bfill().dropna()
        return df_clean

    def fetch_latest_news(
        self, query: str = "Nifty Indian Equity Market Economy", limit: int = 15
    ) -> List[Dict[str, str]]:
        """
        Scrapes and structures news headline data specifically formatted for the FinBERT NLP scoring pipeline.
        
        Returns:
            List of structured dicts with keys: ['headline', 'clean_headline', 'source', 'pub_date', 'link']
        """
        encoded_query = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
        headlines = []

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        }

        try:
            req = urllib.request.Request(rss_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read()

            if BeautifulSoup is not None:
                soup = BeautifulSoup(content, features="xml")
                items = soup.find_all("item")
                for item in items[:limit]:
                    title = item.title.text if item.title else ""
                    pub_date = item.pubDate.text if item.pubDate else ""
                    link = item.link.text if item.link else ""
                    source = item.source.text if item.source else "Google News"

                    if title:
                        clean_text = self._clean_headline_text(title)
                        headlines.append({
                            "headline": title,
                            "clean_headline": clean_text,
                            "source": source,
                            "pub_date": pub_date,
                            "link": link
                        })
                logger.info(f"Scraped & structured {len(headlines)} headlines from Google News RSS")
        except Exception as e:
            logger.warning(f"Could not scrape RSS news feed: {e}. Utilizing structured fallback headlines.")

        if not headlines:
            headlines = self._get_fallback_headlines()

        raw_news_path = os.path.join(self.raw_dir, "latest_news.csv")
        pd.DataFrame(headlines).to_csv(raw_news_path, index=False)

        return headlines

    def _clean_headline_text(self, text: str) -> str:
        """Strips HTML tags, excess whitespace, and source suffix from news title."""
        # Strip trailing " - Source Name"
        if " - " in text:
            text = text.rsplit(" - ", 1)[0]
        # Clean double spaces
        return " ".join(text.split()).strip()

    def _get_fallback_headlines(self) -> List[Dict[str, str]]:
        """Fallback structured news headlines for offline execution and NLP testing."""
        now_str = datetime.datetime.now().strftime("%Y-%m-%d")
        fallback_raw = [
            ("RBI Keeps Repo Rate Unchanged as Inflation Moderates in Q2", "Economic Times"),
            ("Nifty 50 Touches New High on Strong FII Inflows and Banking Sector Surge", "Moneycontrol"),
            ("Indian GDP Growth Surpasses Expectations Driven by Infrastructure Spending", "Business Standard"),
            ("Global Crude Oil Volatility Spikes Amid Middle East Geopolitical Friction", "Reuters"),
            ("Corporate Earnings Growth Remains Resilient Across Nifty Constituents", "Livemint"),
            ("IT Stocks Face Pressure Amid US Tech Sector Slowdown Concerns", "CNBC-TV18"),
            ("Auto Sector Reports Record Festive Sales with Electric Vehicle Growth", "Financial Express"),
        ]

        structured = []
        for h, src in fallback_raw:
            structured.append({
                "headline": h,
                "clean_headline": self._clean_headline_text(h),
                "source": src,
                "pub_date": now_str,
                "link": ""
            })
        return structured

    def _generate_synthetic_prices(self, tickers: List[str], period: str) -> pd.DataFrame:
        """Generates realistic synthetic log-normal equity prices if network is unavailable."""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=750, freq="B")
        np.random.seed(42)
        prices = {}
        for ticker in tickers:
            drift = 0.0005 if "NSEI" in ticker else (0.0003 if "GOLD" in ticker else 0.0002)
            vol = 0.012 if "NSEI" in ticker else (0.008 if "GOLD" in ticker else 0.002)
            daily_returns = np.random.normal(drift, vol, len(dates))
            price = 1000.0 * np.exp(np.cumsum(daily_returns))
            prices[ticker] = price
        return pd.DataFrame(prices, index=dates)


# Alias for MarketDataIngestor
MarketDataFeed = MarketDataIngestor

