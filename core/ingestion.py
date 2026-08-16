import os
import logging
import datetime
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TICKERS = {
    "Equity": "^NSEI",          # Nifty 50 Index
    "Gold": "GOLDBEES.NS",       # Gold ETF
    "Debt": "LIQUIDBEES.NS",     # Liquid / Cash ETF
}

DEFAULT_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "BHARTIARTL.NS", "ITC.NS", "LTIM.NS", "KOTAKBANK.NS", "LT.NS"
]

class MarketDataIngestor:
    """
    Handles ingestion of historical market OHLCV price data via yfinance
    and financial news scraping via urllib/BeautifulSoup/RSS for news sentiment.
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
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetches historical price data and computes daily log returns.
        """
        if tickers is None:
            tickers = list(DEFAULT_TICKERS.values())

        raw_cache_path = os.path.join(self.raw_dir, f"prices_{'_'.join([t.replace('^', '') for t in tickers])}_{period}.csv")

        if os.path.exists(raw_cache_path) and not force_refresh:
            logger.info(f"Loading cached market data from {raw_cache_path}")
            df_prices = pd.read_csv(raw_cache_path, index_col=0, parse_dates=True)
        else:
            if yf is not None:
                logger.info(f"Downloading historical market data for tickers: {tickers}")
                try:
                    data = yf.download(tickers, period=period, interval=interval, progress=False)
                    if isinstance(data.columns, pd.MultiIndex):
                        if 'Adj Close' in data.columns.levels[0]:
                            df_prices = data['Adj Close']
                        elif 'Close' in data.columns.levels[0]:
                            df_prices = data['Close']
                        else:
                            df_prices = data
                    else:
                        df_prices = data[['Close']] if 'Close' in data.columns else data
                        df_prices.columns = tickers

                    if isinstance(df_prices, pd.Series):
                        df_prices = df_prices.to_frame(name=tickers[0])

                    df_prices = df_prices.ffill().bfill().dropna()
                    df_prices.to_csv(raw_cache_path)
                except Exception as e:
                    logger.error(f"Error downloading data from yfinance: {e}")
                    df_prices = self._generate_synthetic_prices(tickers, period)
            else:
                logger.info("yfinance package not installed. Generating synthetic baseline market data.")
                df_prices = self._generate_synthetic_prices(tickers, period)

        # Compute log returns
        returns = np.log(df_prices / df_prices.shift(1)).dropna()
        
        processed_path = os.path.join(self.processed_dir, "processed_returns.csv")
        returns.to_csv(processed_path)

        return df_prices, returns

    def fetch_latest_news(self, query: str = "Nifty Indian Equity Market Economy", limit: int = 15) -> List[Dict[str, str]]:
        """
        Scrapes financial news headlines via RSS/Google News for Indian market news.
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
                        headlines.append({
                            "headline": title,
                            "source": source,
                            "pub_date": pub_date,
                            "link": link
                        })
                logger.info(f"Scraped {len(headlines)} headlines from Google News RSS")
        except Exception as e:
            logger.warning(f"Could not scrape RSS news feed: {e}. Utilizing cached/mock headlines.")

        if not headlines:
            headlines = self._get_fallback_headlines()

        raw_news_path = os.path.join(self.raw_dir, "latest_news.csv")
        pd.DataFrame(headlines).to_csv(raw_news_path, index=False)

        return headlines

    def _get_fallback_headlines(self) -> List[Dict[str, str]]:
        """Fallback news headlines for offline execution and testing."""
        now_str = datetime.datetime.now().strftime("%Y-%m-%d")
        return [
            {"headline": "RBI Keeps Repo Rate Unchanged as Inflation Moderates in Q2", "source": "Economic Times", "pub_date": now_str, "link": ""},
            {"headline": "Nifty 50 Touches New High on Strong FII Inflows and Banking Sector Surge", "source": "Moneycontrol", "pub_date": now_str, "link": ""},
            {"headline": "Indian GDP Growth Surpasses Expectations Driven by Infrastructure Spending", "source": "Business Standard", "pub_date": now_str, "link": ""},
            {"headline": "Global Crude Oil Volatility Spikes Amid Middle East Geopolitical Friction", "source": "Reuters", "pub_date": now_str, "link": ""},
            {"headline": "Corporate Earnings Growth Remains Resilient Across Nifty constituents", "source": "Livemint", "pub_date": now_str, "link": ""},
            {"headline": "IT Stocks Face Pressure Amid US Tech Sector Slowdown Concerns", "source": "CNBC-TV18", "pub_date": now_str, "link": ""},
            {"headline": "Auto Sector Reports Record Festive Sales with Electric Vehicle Growth", "source": "Financial Express", "pub_date": now_str, "link": ""},
        ]

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
