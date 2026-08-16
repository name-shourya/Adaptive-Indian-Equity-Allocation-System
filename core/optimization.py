import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default Asset Constraints
DEFAULT_BOUNDS = {
    "Equity": (0.20, 0.90),
    "Gold": (0.00, 0.30),
    "Debt": (0.05, 0.50),
    "Cash": (0.00, 0.20),
}

class PortfolioOptimizer:
    """
    Portfolio Optimization Engine implementing:
    1. Black-Litterman Model combining market equilibrium with GJR-GARCH & FinBERT views.
    2. Max Sharpe Ratio / Mean-Variance solver via scipy.optimize.minimize (SLSQP).
    """

    def __init__(self, risk_free_rate: float = 0.06, risk_aversion: float = 2.5):
        """
        Parameters:
            risk_free_rate: Annualized Indian risk-free rate (default 6.0% / RBI Repo rate proxy)
            risk_aversion: Market risk aversion coefficient lambda (default 2.5)
        """
        self.rf = risk_free_rate
        self.risk_aversion = risk_aversion

    def optimize_black_litterman(
        self,
        returns: pd.DataFrame,
        sentiment_score: float = 0.0,
        gjr_vol: Optional[Dict[int, float]] = None,
        market_weights: Optional[Dict[str, float]] = None,
        custom_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Calculates optimal asset allocation weights using Black-Litterman framework.
        
        Parameters:
            returns: DataFrame of asset log daily returns
            sentiment_score: Composite FinBERT sentiment score in range [-1.0, 1.0]
            gjr_vol: GJR-GARCH forecast volatility dict {horizon: annualized_vol}
            market_weights: Initial benchmark equilibrium asset weights
            custom_bounds: Min/max allocation constraints per asset class
            
        Returns:
            Dict containing optimized weights, expected returns, portfolio volatility,
            and Sharpe ratio.
        """
        asset_names = list(returns.columns)
        n = len(asset_names)

        # Default market equilibrium weights if not specified
        if market_weights is None:
            if n == 3:
                mkt_w = np.array([0.60, 0.20, 0.20]) # Equity, Gold, Debt
            elif n == 4:
                mkt_w = np.array([0.55, 0.20, 0.20, 0.05])
            else:
                mkt_w = np.ones(n) / n
        else:
            mkt_w = np.array([market_weights.get(a, 1.0 / n) for a in asset_names])
            mkt_w = mkt_w / np.sum(mkt_w)

        # 1. Historical Sample Covariance Matrix (Annualized)
        cov_matrix = returns.cov().values * 252

        # 2. Integrate GJR-GARCH forecast into Equity Covariance component if provided
        if gjr_vol and 21 in gjr_vol:
            garch_ann_vol = gjr_vol[21]
            for i, name in enumerate(asset_names):
                if "NSEI" in name or "Equity" in name or i == 0:
                    hist_vol = np.sqrt(cov_matrix[i, i])
                    if hist_vol > 0:
                        vol_scale = garch_ann_vol / hist_vol
                        cov_matrix[i, :] *= vol_scale
                        cov_matrix[:, i] *= vol_scale

        # 3. Market Implied Equilibrium Returns (\Pi = \lambda \Sigma w_{mkt})
        pi = self.risk_aversion * np.dot(cov_matrix, mkt_w)

        # 4. Formulate Black-Litterman Views from FinBERT Sentiment
        # Sentiment score \in [-1, 1] tilts equity view returns by up to \pm 4%
        tau = 0.05  # Weight of uncertainty on prior
        
        # Single view matrix P on Equity asset
        P = np.zeros((1, n))
        P[0, 0] = 1.0  # View on equity return
        
        # Absolute view return Q = \Pi_{equity} + tilt
        tilt = sentiment_score * 0.04  # e.g., +2% return tilt if bullish score 0.5
        Q = np.array([pi[0] + tilt])

        # Diagonal view uncertainty matrix Omega
        Omega = np.array([[tau * np.dot(np.dot(P, cov_matrix), P.T)[0, 0]]])
        if Omega[0, 0] == 0:
            Omega[0, 0] = 1e-4

        # 5. Black-Litterman Posterior Return Vector (\mu_{BL})
        sigma_inv = np.linalg.inv(tau * cov_matrix)
        omega_inv = np.linalg.inv(Omega)

        middle_term = np.linalg.inv(sigma_inv + np.dot(np.dot(P.T, omega_inv), P))
        right_term = np.dot(sigma_inv, pi) + np.dot(np.dot(P.T, omega_inv), Q)
        
        mu_bl = np.dot(middle_term, right_term)

        # 6. SLSQP Max Sharpe Optimization
        opt_res = self.optimize_max_sharpe(
            expected_returns=mu_bl,
            cov_matrix=cov_matrix,
            asset_names=asset_names,
            custom_bounds=custom_bounds
        )

        opt_res["implied_equilibrium_returns"] = dict(zip(asset_names, pi))
        opt_res["black_litterman_returns"] = dict(zip(asset_names, mu_bl))
        opt_res["sentiment_tilt_applied"] = float(tilt)

        return opt_res

    def optimize_max_sharpe(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        asset_names: List[str],
        custom_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Solves Max Sharpe Ratio portfolio optimization problem using scipy SLSQP.
        """
        n = len(asset_names)

        # Objective Function: Negative Sharpe Ratio
        def negative_sharpe(weights):
            p_return = np.dot(weights, expected_returns)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if p_vol <= 0:
                return 1e6
            return -(p_return - self.rf) / p_vol

        # Constraint: Weights sum to 1.0
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Boundary Constraints
        bounds = []
        for name in asset_names:
            found_bound = False
            if custom_bounds:
                for k, v in custom_bounds.items():
                    if k.lower() in name.lower():
                        bounds.append(v)
                        found_bound = True
                        break
            if not found_bound:
                for k, v in DEFAULT_BOUNDS.items():
                    if k.lower() in name.lower():
                        bounds.append(v)
                        found_bound = True
                        break
            if not found_bound:
                bounds.append((0.0, 1.0))

        # Initial Guess: Equal Weights
        init_weights = np.ones(n) / n

        if minimize is None:
            logger.debug("scipy.optimize not installed. Using heuristic equal-weighted allocation.")
            opt_w = init_weights
            p_return = float(np.dot(opt_w, expected_returns))
            p_vol = float(np.sqrt(np.dot(opt_w.T, np.dot(cov_matrix, opt_w))))
            sharpe = float((p_return - self.rf) / p_vol) if p_vol > 0 else 0.0
            return {
                "success": True,
                "weights": dict(zip(asset_names, opt_w)),
                "expected_return": p_return,
                "portfolio_volatility": p_vol,
                "sharpe_ratio": sharpe,
                "optimizer_status": "Heuristic fallback"
            }

        if res.success:
            opt_w = res.x
        else:
            logger.warning(f"SLSQP Optimizer warning: {res.message}. Normalizing initial bounds.")
            opt_w = init_weights

        # Clean numerical precision artifacts
        opt_w = np.round(opt_w, 4)
        opt_w = opt_w / np.sum(opt_w)  # Ensure exact sum to 1.0

        p_return = float(np.dot(opt_w, expected_returns))
        p_vol = float(np.sqrt(np.dot(opt_w.T, np.dot(cov_matrix, opt_w))))
        sharpe = float((p_return - self.rf) / p_vol) if p_vol > 0 else 0.0

        weights_dict = {asset_names[i]: float(opt_w[i]) for i in range(n)}

        return {
            "success": res.success,
            "weights": weights_dict,
            "expected_return": p_return,
            "portfolio_volatility": p_vol,
            "sharpe_ratio": sharpe,
            "optimizer_status": res.message
        }
