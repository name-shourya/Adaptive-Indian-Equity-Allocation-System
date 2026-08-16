import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default Asset Allocation Constraints
DEFAULT_BOUNDS = {
    "Equity": (0.20, 0.90),
    "Gold": (0.00, 0.30),
    "Debt": (0.05, 0.50),
    "Cash": (0.00, 0.20),
}


class SharpeOptimizer:
    """Sharpe Ratio Portfolio Optimizer using Sequential Least Squares Programming (SLSQP).

    Solves for optimal asset allocation weights that maximize the portfolio Sharpe ratio
    given expected returns vector and covariance matrix, subject to weight bounds and
    full investment constraints.
    """

    def __init__(self, risk_free_rate: float = 0.065, max_weight: float = 0.15) -> None:
        """Initializes SharpeOptimizer with risk-free rate and concentration bounds.

        Args:
            risk_free_rate: Annualized risk-free interest rate (default: 0.065 / 91-day T-Bill).
            max_weight: Maximum weight cap for any single asset (default: 0.15).
        """
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight

    def optimize(
        self, expected_returns: pd.Series, cov_matrix: pd.DataFrame
    ) -> pd.Series:
        """Solves for portfolio weights that maximize the Sharpe ratio.

        Validates inputs, aligns dimensions, handles NaN values, defines the negative
        Sharpe ratio objective function, and executes SLSQP optimization subject to
        sum-to-one and single-asset weight bounds.

        Args:
            expected_returns: Pandas Series of expected annual asset returns.
            cov_matrix: Pandas DataFrame of annual asset covariance matrix.

        Returns:
            Pandas Series of optimal asset weights indexed matching expected_returns.

        Raises:
            ValueError: If input dimensions mismatch or data is invalid.
        """
        if not isinstance(expected_returns, pd.Series) or not isinstance(cov_matrix, pd.DataFrame):
            raise ValueError("expected_returns must be a pd.Series and cov_matrix must be a pd.DataFrame.")

        # Clean NaN values
        er_clean = expected_returns.fillna(0.0)
        cov_clean = cov_matrix.fillna(0.0)

        # Align dimensions
        common_assets = er_clean.index.intersection(cov_clean.index)
        if len(common_assets) == 0:
            raise ValueError("No common asset symbols found between expected_returns and cov_matrix.")

        er_clean = er_clean.loc[common_assets]
        cov_clean = cov_clean.loc[common_assets, common_assets]
        n_assets = len(common_assets)

        er_vals = er_clean.values.astype(float)
        cov_vals = cov_clean.values.astype(float)

        # Equal weighted initial guess
        init_weights = np.ones(n_assets) / n_assets

        if minimize is None:
            logger.warning("scipy.optimize not installed. Returning equal-weighted portfolio.")
            return pd.Series(init_weights, index=common_assets, name="optimal_weights")

        # Define nested objective function: negative Sharpe ratio
        def _negative_sharpe(w: np.ndarray) -> float:
            p_return = float(w.T @ er_vals)
            p_variance = float(w.T @ cov_vals @ w)
            p_vol = np.sqrt(p_variance) if p_variance > 0 else 1e-8
            return -(p_return - self.risk_free_rate) / p_vol

        # Constraint: Weights sum to 1.0
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Bound individual weights between 0.0 and upper_bound
        # Adjust upper_bound to ensure feasibility if n_assets is small (e.g. 1/n > max_weight)
        upper_cap = max(self.max_weight, 1.0 / n_assets)
        bounds = [(0.0, upper_cap) for _ in range(n_assets)]

        try:
            res = minimize(
                _negative_sharpe,
                init_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-9},
            )

            if res.success:
                opt_w = res.x
            else:
                logger.warning(f"SLSQP solver warning ({res.message}). Falling back to initial guess.")
                opt_w = init_weights

            # Clean numerical artifacts and re-normalize sum to 1.0
            opt_w = np.clip(opt_w, 0.0, 1.0)
            if np.sum(opt_w) > 0:
                opt_w = opt_w / np.sum(opt_w)

            return pd.Series(opt_w, index=common_assets, name="optimal_weights")

        except Exception as e:
            logger.error(f"Error during Sharpe optimization: {e}")
            return pd.Series(init_weights, index=common_assets, name="optimal_weights")


class PortfolioOptimizer:
    """Portfolio Optimization Engine implementing Black-Litterman and Max-Sharpe solvers."""

    def __init__(self, risk_free_rate: float = 0.06, risk_aversion: float = 2.5) -> None:
        """Initializes PortfolioOptimizer.

        Args:
            risk_free_rate: Risk-free rate (default 0.06).
            risk_aversion: Market risk aversion coefficient lambda (default 2.5).
        """
        self.rf = risk_free_rate
        self.risk_aversion = risk_aversion
        self.sharpe_opt = SharpeOptimizer(risk_free_rate=risk_free_rate)

    def optimize_black_litterman(
        self,
        returns: pd.DataFrame,
        sentiment_score: float = 0.0,
        gjr_vol: Optional[Dict[int, float]] = None,
        market_weights: Optional[Dict[str, float]] = None,
        custom_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        """Calculates optimal asset allocation weights using Black-Litterman framework."""
        asset_names = list(returns.columns)
        n = len(asset_names)

        if market_weights is None:
            if n == 3:
                mkt_w = np.array([0.60, 0.20, 0.20])
            elif n == 4:
                mkt_w = np.array([0.55, 0.20, 0.20, 0.05])
            else:
                mkt_w = np.ones(n) / n
        else:
            mkt_w = np.array([market_weights.get(a, 1.0 / n) for a in asset_names])
            mkt_w = mkt_w / np.sum(mkt_w)

        cov_matrix = returns.cov().values * 252

        if gjr_vol and 21 in gjr_vol:
            garch_ann_vol = gjr_vol[21]
            for i, name in enumerate(asset_names):
                if "NSEI" in name or "Equity" in name or i == 0:
                    hist_vol = np.sqrt(cov_matrix[i, i])
                    if hist_vol > 0:
                        vol_scale = garch_ann_vol / hist_vol
                        cov_matrix[i, :] *= vol_scale
                        cov_matrix[:, i] *= vol_scale

        pi = self.risk_aversion * np.dot(cov_matrix, mkt_w)

        tau = 0.05
        P = np.zeros((1, n))
        P[0, 0] = 1.0
        
        tilt = sentiment_score * 0.04
        Q = np.array([pi[0] + tilt])

        Omega = np.array([[tau * np.dot(np.dot(P, cov_matrix), P.T)[0, 0]]])
        if Omega[0, 0] == 0:
            Omega[0, 0] = 1e-4

        try:
            sigma_inv = np.linalg.inv(tau * cov_matrix)
            omega_inv = np.linalg.inv(Omega)
            middle_term = np.linalg.inv(sigma_inv + np.dot(np.dot(P.T, omega_inv), P))
            right_term = np.dot(sigma_inv, pi) + np.dot(np.dot(P.T, omega_inv), Q)
            mu_bl = np.dot(middle_term, right_term)
        except Exception:
            mu_bl = pi

        opt_res = self.optimize_max_sharpe(
            expected_returns=mu_bl,
            cov_matrix=cov_matrix,
            asset_names=asset_names,
            custom_bounds=custom_bounds,
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
        custom_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        """Solves Max Sharpe Ratio portfolio optimization problem using scipy SLSQP."""
        n = len(asset_names)

        def negative_sharpe(weights: np.ndarray) -> float:
            p_return = float(np.dot(weights, expected_returns))
            p_var = float(np.dot(weights.T, np.dot(cov_matrix, weights)))
            p_vol = np.sqrt(p_var) if p_var > 0 else 1e-8
            return -(p_return - self.rf) / p_vol

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

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
                "optimizer_status": "Heuristic fallback",
            }

        try:
            res = minimize(
                negative_sharpe,
                init_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-9},
            )

            if res.success:
                opt_w = res.x
            else:
                opt_w = init_weights

            opt_w = np.round(opt_w, 4)
            opt_w = opt_w / np.sum(opt_w)

            p_return = float(np.dot(opt_w, expected_returns))
            p_vol = float(np.sqrt(np.dot(opt_w.T, np.dot(cov_matrix, opt_w))))
            sharpe = float((p_return - self.rf) / p_vol) if p_vol > 0 else 0.0

            return {
                "success": res.success,
                "weights": dict(zip(asset_names, opt_w)),
                "expected_return": p_return,
                "portfolio_volatility": p_vol,
                "sharpe_ratio": sharpe,
                "optimizer_status": res.message,
            }
        except Exception as e:
            logger.error(f"Error in optimize_max_sharpe: {e}")
            return {
                "success": False,
                "weights": dict(zip(asset_names, init_weights)),
                "expected_return": float(np.dot(init_weights, expected_returns)),
                "portfolio_volatility": 0.15,
                "sharpe_ratio": 0.0,
                "optimizer_status": str(e),
            }

