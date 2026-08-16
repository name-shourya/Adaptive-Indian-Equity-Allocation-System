import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from core.volatility import VolatilityEngine
from core.sentiment import SentimentEngine
from core.optimization import PortfolioOptimizer
from backtest.metrics import MetricsCalculator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class WalkForwardBacktester:
    """
    Rolling Window Walk-Forward Backtesting Engine.
    Simulates dynamic asset rebalancing over historical periods and compares
    against static benchmark strategies (e.g., Static 60/40 & Buy-and-Hold Equity).
    """

    def __init__(
        self,
        train_window: int = 252,
        rebalance_freq: int = 21,
        transaction_cost_bps: float = 15.0,
        risk_free_rate: float = 0.06
    ):
        self.train_window = train_window
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost_bps / 10000.0  # 15 bps = 0.0015
        self.vol_engine = VolatilityEngine()
        self.sentiment_engine = SentimentEngine(use_transformer=False)  # Fast mode for backtest
        self.optimizer = PortfolioOptimizer(risk_free_rate=risk_free_rate)
        self.metrics_calc = MetricsCalculator(risk_free_rate=risk_free_rate)

    def run_backtest(
        self,
        returns: pd.DataFrame,
        initial_capital: float = 100000.0,
        benchmark_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Executes rolling walk-forward backtest over daily returns DataFrame.
        
        Parameters:
            returns: DataFrame of daily asset log returns
            initial_capital: Initial portfolio starting capital (e.g. 100,000 INR)
            benchmark_weights: Benchmark portfolio allocation (default 60% Eq, 20% Gold, 20% Debt)
            
        Returns:
            Dict with equity curves, rebalance history, and comparative performance metrics.
        """
        n_days, n_assets = returns.shape
        if n_days <= self.train_window + self.rebalance_freq:
            raise ValueError(f"Insufficient historical data ({n_days} rows) for train_window ({self.train_window}).")

        asset_names = list(returns.columns)
        
        if benchmark_weights is None:
            benchmark_weights = {asset_names[0]: 0.60}
            if n_assets > 1:
                benchmark_weights[asset_names[1]] = 0.20
            if n_assets > 2:
                benchmark_weights[asset_names[2]] = 0.20

        # Align simple returns
        simple_returns = np.exp(returns) - 1.0

        dates = returns.index[self.train_window:]
        
        # Portfolio value arrays
        dynamic_values = [initial_capital]
        static_values = [initial_capital]
        equity_hold_values = [initial_capital]

        current_dyn_w = np.ones(n_assets) / n_assets
        static_w = np.array([benchmark_weights.get(a, 1.0 / n_assets) for a in asset_names])
        static_w = static_w / np.sum(static_w)

        rebalance_records = []

        # Iterate day by day starting from end of first training window
        for idx in range(self.train_window, n_days):
            current_date = returns.index[idx]
            day_in_test = idx - self.train_window

            # Rebalance trigger check
            if day_in_test % self.rebalance_freq == 0:
                # 1. Historical train slice
                train_returns = returns.iloc[idx - self.train_window : idx]
                
                # 2. Fit GJR-GARCH model on equity asset
                equity_series = train_returns.iloc[:, 0]
                garch_res = self.vol_engine.fit_gjr_garch(equity_series)
                vol_forecast = self.vol_engine.forecast_volatility(garch_res)

                # 3. Simulate period macro sentiment (historical signal window)
                recent_equity_trend = train_returns.iloc[-10:, 0].mean()
                simulated_score = float(np.clip(recent_equity_trend * 50.0, -0.6, 0.6))

                # 4. Optimize Black-Litterman Target Weights
                bl_opt = self.optimizer.optimize_black_litterman(
                    returns=train_returns,
                    sentiment_score=simulated_score,
                    gjr_vol=vol_forecast
                )

                new_dyn_w = np.array([bl_opt["weights"].get(a, 1.0 / n_assets) for a in asset_names])
                new_dyn_w = new_dyn_w / np.sum(new_dyn_w)

                # Account for transaction costs on portfolio weight turnover
                turnover = np.sum(np.abs(new_dyn_w - current_dyn_w))
                cost = dynamic_values[-1] * turnover * self.transaction_cost
                dynamic_values[-1] -= cost

                current_dyn_w = new_dyn_w

                rebalance_records.append({
                    "date": current_date.strftime("%Y-%m-%d") if hasattr(current_date, "strftime") else str(current_date),
                    "weights": dict(zip(asset_names, current_dyn_w)),
                    "turnover": float(turnover),
                    "garch_vol_21d": float(vol_forecast.get(21, 0.15)),
                    "sentiment_score": float(simulated_score)
                })

            # Calculate daily returns
            day_ret = simple_returns.iloc[idx].values
            
            # Dynamic Portfolio growth
            dyn_ret = np.dot(current_dyn_w, day_ret)
            dynamic_values.append(dynamic_values[-1] * (1.0 + dyn_ret))

            # Static Benchmark growth
            stat_ret = np.dot(static_w, day_ret)
            static_values.append(static_values[-1] * (1.0 + stat_ret))

            # Buy and Hold 100% Equity Benchmark growth
            eq_ret = day_ret[0]
            equity_hold_values.append(equity_hold_values[-1] * (1.0 + eq_ret))

        # Convert to Series with DatetimeIndex
        idx_dates = returns.index[self.train_window - 1 :]
        s_dynamic = pd.Series(dynamic_values, index=idx_dates, name="Dynamic Tactical")
        s_static = pd.Series(static_values, index=idx_dates, name="Static Benchmark")
        s_equity = pd.Series(equity_hold_values, index=idx_dates, name="Buy & Hold Equity")

        # Compute Metrics
        metrics_dynamic = self.metrics_calc.compute_all_metrics(s_dynamic)
        metrics_static = self.metrics_calc.compute_all_metrics(s_static)
        metrics_equity = self.metrics_calc.compute_all_metrics(s_equity)

        equity_curves_df = pd.DataFrame({
            "Dynamic Tactical": s_dynamic,
            "Static Benchmark": s_static,
            "Buy & Hold Equity": s_equity
        })

        return {
            "equity_curves": equity_curves_df,
            "rebalance_records": rebalance_records,
            "metrics": {
                "Dynamic Tactical": metrics_dynamic,
                "Static Benchmark": metrics_static,
                "Buy & Hold Equity": metrics_equity
            }
        }
