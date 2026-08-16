import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from arch import arch_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class VolatilityEngine:
    """
    Asymmetric GJR-GARCH Volatility Modeling Engine.
    Implements GJR-GARCH(1,1,1) using the `arch` library to model
    leverage effects and forecast conditional volatility.
    """

    def __init__(self, p: int = 1, o: int = 1, q: int = 1, dist: str = "t"):
        self.p = p
        self.o = o
        self.q = q
        self.dist = dist

    def fit_gjr_garch(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Fits a GJR-GARCH(1,1,1) model on daily percentage/log return series.
        
        Parameters:
            returns: Daily return pandas Series (scaled to percentage for arch stability)
            
        Returns:
            Dict containing model parameters (omega, alpha, gamma/asymmetry, beta),
            conditional volatility series, persistence, and fit object.
        """
        # Rescale returns to percentage points for numerical stability in arch solver
        scale_factor = 100.0
        ret_scaled = returns.dropna() * scale_factor

        try:
            am = arch_model(
                ret_scaled,
                vol="GARCH",
                p=self.p,
                o=self.o,
                q=self.q,
                dist=self.dist,
                rescale=False
            )
            res = am.fit(disp="off", show_warning=False)
            
            # Extract parameters
            params = res.params
            omega = params.get("omega", 0.0001)
            alpha = params.get("alpha[1]", 0.05)
            gamma = params.get("gamma[1]", 0.05)  # Asymmetry coefficient
            beta = params.get("beta[1]", 0.85)

            # Calculate volatility persistence: alpha + 0.5 * gamma + beta
            persistence = alpha + 0.5 * gamma + beta

            # Conditional volatility in original scale (daily std dev)
            cond_vol_daily = res.conditional_volatility / scale_factor
            ann_cond_vol = cond_vol_daily * np.sqrt(252)

            return {
                "success": True,
                "model_name": "GJR-GARCH(1,1,1)",
                "omega": omega / (scale_factor ** 2),
                "alpha": alpha,
                "gamma": gamma,
                "beta": beta,
                "persistence": persistence,
                "last_volatility_daily": cond_vol_daily.iloc[-1],
                "last_volatility_annualized": ann_cond_vol.iloc[-1],
                "conditional_volatility": cond_vol_daily,
                "annualized_volatility": ann_cond_vol,
                "fit_result": res,
                "scale_factor": scale_factor
            }

        except Exception as e:
            logger.warning(f"GJR-GARCH fit failed with exception: {e}. Falling back to EWMA Volatility Engine.")
            return self._fit_ewma_fallback(returns)

    def forecast_volatility(self, fit_output: Dict[str, Any], horizons: Tuple[int, ...] = (1, 5, 21)) -> Dict[int, float]:
        """
        Forecasts conditional volatility for multiple horizons (1d, 5d, 21d).
        Returns annualized forecast volatility for each horizon.
        """
        if not fit_output.get("success", False) or "fit_result" not in fit_output:
            daily_vol = fit_output.get("last_volatility_daily", 0.01)
            return {h: float(daily_vol * np.sqrt(252)) for h in horizons}

        try:
            fit_res = fit_output["fit_result"]
            scale_factor = fit_output.get("scale_factor", 100.0)
            max_horizon = max(horizons)
            
            forecasts = fit_res.forecast(horizon=max_horizon, reindex=False)
            variance_forecasts = forecasts.variance.iloc[-1].values / (scale_factor ** 2)

            results = {}
            for h in horizons:
                # Average daily variance over horizon h
                avg_var = np.mean(variance_forecasts[:h])
                ann_vol = np.sqrt(avg_var * 252)
                results[h] = float(ann_vol)

            return results
        except Exception as e:
            logger.error(f"Error forecasting GJR-GARCH volatility: {e}")
            daily_vol = fit_output.get("last_volatility_daily", 0.01)
            return {h: float(daily_vol * np.sqrt(252)) for h in horizons}

    def _fit_ewma_fallback(self, returns: pd.Series, lambda_param: float = 0.94) -> Dict[str, Any]:
        """Exponentially Weighted Moving Average (EWMA) fallback volatility model."""
        clean_ret = returns.dropna()
        var = np.zeros(len(clean_ret))
        var[0] = clean_ret.var()

        for t in range(1, len(clean_ret)):
            var[t] = lambda_param * var[t - 1] + (1 - lambda_param) * (clean_ret.iloc[t - 1] ** 2)

        daily_vol = pd.Series(np.sqrt(var), index=clean_ret.index)
        ann_vol = daily_vol * np.sqrt(252)

        return {
            "success": True,
            "model_name": "EWMA Fallback (\u03bb=0.94)",
            "omega": 0.0,
            "alpha": 1 - lambda_param,
            "gamma": 0.0,
            "beta": lambda_param,
            "persistence": 1.0,
            "last_volatility_daily": daily_vol.iloc[-1],
            "last_volatility_annualized": ann_vol.iloc[-1],
            "conditional_volatility": daily_vol,
            "annualized_volatility": ann_vol,
            "scale_factor": 1.0
        }
