import numpy as np
import pandas as pd
from typing import Dict, Any

class MetricsCalculator:
    """
    Computes statistical quantitative metrics for portfolio equity curves and return series:
    CAGR, Annual Volatility, Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio, VaR.
    """

    def __init__(self, risk_free_rate: float = 0.06):
        self.rf = risk_free_rate

    def compute_all_metrics(self, portfolio_values: pd.Series) -> Dict[str, Any]:
        """
        Computes full metrics battery from daily portfolio total value series.
        
        Parameters:
            portfolio_values: Series of portfolio dollar/rupee value indexed by Date
            
        Returns:
            Dict of quantitative performance metrics
        """
        if len(portfolio_values) < 2:
            return self._empty_metrics()

        # Calculate daily percentage returns
        daily_returns = portfolio_values.pct_change().dropna()

        # 1. Total Return & CAGR
        start_val = portfolio_values.iloc[0]
        end_val = portfolio_values.iloc[-1]
        total_return = (end_val - start_val) / start_val

        n_days = len(portfolio_values)
        n_years = max(n_days / 252.0, 1e-4)
        cagr = ((end_val / start_val) ** (1.0 / n_years)) - 1.0

        # 2. Annualized Volatility
        ann_vol = daily_returns.std() * np.sqrt(252)

        # 3. Sharpe Ratio
        daily_rf = (1.0 + self.rf) ** (1.0 / 252.0) - 1.0
        excess_returns = daily_returns - daily_rf
        sharpe = (excess_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0.0

        # 4. Downside Volatility & Sortino Ratio
        negative_returns = daily_returns[daily_returns < 0]
        downside_std = negative_returns.std() * np.sqrt(252)
        sortino = ((daily_returns.mean() * 252 - self.rf) / downside_std) if downside_std > 0 else 0.0

        # 5. Maximum Drawdown (Peak to Trough)
        cum_max = portfolio_values.cummax()
        drawdown = (portfolio_values - cum_max) / cum_max
        max_drawdown = abs(drawdown.min())

        # 6. Calmar Ratio
        calmar = (cagr / max_drawdown) if max_drawdown > 0 else 0.0

        # 7. Value at Risk (VaR 95% Daily)
        var_95 = float(np.percentile(daily_returns, 5))

        # 8. Win Rate (% Positive Days)
        win_rate = float((daily_returns > 0).mean())

        return {
            "start_date": portfolio_values.index[0].strftime("%Y-%m-%d") if hasattr(portfolio_values.index[0], "strftime") else str(portfolio_values.index[0]),
            "end_date": portfolio_values.index[-1].strftime("%Y-%m-%d") if hasattr(portfolio_values.index[-1], "strftime") else str(portfolio_values.index[-1]),
            "total_return_pct": float(total_return * 100),
            "cagr_pct": float(cagr * 100),
            "annualized_volatility_pct": float(ann_vol * 100),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown_pct": float(max_drawdown * 100),
            "calmar_ratio": float(calmar),
            "var_95_daily_pct": float(var_95 * 100),
            "win_rate_pct": float(win_rate * 100),
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        return {
            "total_return_pct": 0.0,
            "cagr_pct": 0.0,
            "annualized_volatility_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "calmar_ratio": 0.0,
            "var_95_daily_pct": 0.0,
            "win_rate_pct": 0.0,
        }
