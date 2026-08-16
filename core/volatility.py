import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, Union

try:
    from arch import arch_model
except ImportError:
    arch_model = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class VolatilityModeler:
    """Volatility Modeler using asymmetric GJR-GARCH for conditional variance forecasting.

    Implements a GJR-GARCH(1,1,1) model via the `arch` package to capture leverage effects
    and heavy tails in financial return series. Provides 1-step ahead annualized volatility
    forecasting and in-sample conditional volatility calculation.
    """

    def __init__(
        self,
        p: int = 1,
        o: int = 1,
        q: int = 1,
        dist: str = "StudentsT",
        scale_factor: float = 100.0,
    ) -> None:
        """Initializes the VolatilityModeler with GJR-GARCH parameters.

        Args:
            p: Order of symmetric GARCH innovation term.
            o: Order of asymmetric leverage term (GJR component).
            q: Order of lagged conditional variance term.
            dist: Error distribution specification ('StudentsT' or 't', 'normal').
            scale_factor: Multiplier to scale returns to percentage for solver stability.
        """
        self.p = p
        self.o = o
        self.q = q
        self.dist = dist
        self.scale_factor = scale_factor

    def fit_predict_next_day(self, returns_series: pd.Series) -> float:
        """Fits GJR-GARCH model and returns 1-step ahead annualized volatility forecast.

        Drops missing values from the input return series, rescales returns by 100
        for optimizer stability, fits a GJR-GARCH(1,1,1) model with Student's t
        distribution, descales the 1-step variance forecast, and annualizes it over 252 days.

        Args:
            returns_series: Pandas Series containing daily log return time series.

        Returns:
            Annualized 1-step ahead forecast volatility as a floating-point number.
        """
        clean_returns = returns_series.dropna()
        if clean_returns.empty:
            logger.warning("Empty return series provided to fit_predict_next_day. Returning default 0.15.")
            return 0.15

        scaled_returns = clean_returns * self.scale_factor

        if arch_model is None:
            logger.debug("arch package not installed. Utilizing EWMA fallback forecast.")
            return float(self._fit_ewma_fallback(clean_returns)["last_volatility_annualized"])

        try:
            am = arch_model(
                scaled_returns,
                vol="GARCH",
                p=self.p,
                o=self.o,
                q=self.q,
                dist=self.dist,
                rescale=False,
            )
            fit_res = am.fit(disp="off", show_warning=False)
            forecast = fit_res.forecast(horizon=1, reindex=False)
            
            # Extract 1-step ahead forecasted variance
            var_scaled_1step = float(forecast.variance.iloc[-1].values[0])
            
            # Descale variance and compute daily standard deviation
            daily_vol = np.sqrt(var_scaled_1step) / self.scale_factor
            
            # Annualize (252 trading days)
            annualized_vol = float(daily_vol * np.sqrt(252))
            return annualized_vol

        except Exception as e:
            logger.warning(f"GJR-GARCH solver failed in fit_predict_next_day: {e}. Using EWMA fallback.")
            return float(self._fit_ewma_fallback(clean_returns)["last_volatility_annualized"])

    def get_conditional_volatility(self, returns_series: pd.Series) -> pd.Series:
        """Fits GJR-GARCH model and returns annualized in-sample conditional volatility.

        Fits the asymmetric GJR-GARCH(1,1,1) model to the input return series and
        extracts the fitted conditional volatility series in annualized terms.

        Args:
            returns_series: Pandas Series containing daily log return time series.

        Returns:
            Pandas Series of annualized in-sample conditional volatility matching input dates.
        """
        clean_returns = returns_series.dropna()
        if clean_returns.empty:
            return pd.Series(dtype=float)

        scaled_returns = clean_returns * self.scale_factor

        if arch_model is None:
            logger.debug("arch package not installed. Utilizing EWMA conditional volatility.")
            return self._fit_ewma_fallback(clean_returns)["annualized_volatility"]

        try:
            am = arch_model(
                scaled_returns,
                vol="GARCH",
                p=self.p,
                o=self.o,
                q=self.q,
                dist=self.dist,
                rescale=False,
            )
            fit_res = am.fit(disp="off", show_warning=False)
            
            # Descale daily conditional volatility and annualize
            cond_vol_daily = fit_res.conditional_volatility / self.scale_factor
            ann_cond_vol = cond_vol_daily * np.sqrt(252)
            ann_cond_vol.name = "gjr_garch_conditional_volatility"
            return ann_cond_vol

        except Exception as e:
            logger.warning(f"GJR-GARCH solver failed in get_conditional_volatility: {e}. Using EWMA fallback.")
            return self._fit_ewma_fallback(clean_returns)["annualized_volatility"]

    def fit_gjr_garch(self, returns: pd.Series) -> Dict[str, Any]:
        """Fits a GJR-GARCH(1,1,1) model and returns parameter dictionary.

        Args:
            returns: Daily return pandas Series.

        Returns:
            Dictionary containing model parameters, conditional volatility, and fit object.
        """
        clean_returns = returns.dropna()
        scaled_returns = clean_returns * self.scale_factor

        if arch_model is None:
            return self._fit_ewma_fallback(clean_returns)

        try:
            am = arch_model(
                scaled_returns,
                vol="GARCH",
                p=self.p,
                o=self.o,
                q=self.q,
                dist=self.dist,
                rescale=False,
            )
            res = am.fit(disp="off", show_warning=False)
            
            params = res.params
            omega = params.get("omega", 0.0001)
            alpha = params.get("alpha[1]", 0.05)
            gamma = params.get("gamma[1]", 0.05)
            beta = params.get("beta[1]", 0.85)

            persistence = alpha + 0.5 * gamma + beta

            cond_vol_daily = res.conditional_volatility / self.scale_factor
            ann_cond_vol = cond_vol_daily * np.sqrt(252)

            return {
                "success": True,
                "model_name": "GJR-GARCH(1,1,1)",
                "omega": omega / (self.scale_factor ** 2),
                "alpha": alpha,
                "gamma": gamma,
                "beta": beta,
                "persistence": persistence,
                "last_volatility_daily": float(cond_vol_daily.iloc[-1]),
                "last_volatility_annualized": float(ann_cond_vol.iloc[-1]),
                "conditional_volatility": cond_vol_daily,
                "annualized_volatility": ann_cond_vol,
                "fit_result": res,
                "scale_factor": self.scale_factor,
            }

        except Exception as e:
            logger.warning(f"GJR-GARCH fit failed with exception: {e}. Falling back to EWMA.")
            return self._fit_ewma_fallback(clean_returns)

    def forecast_volatility(
        self, fit_output: Dict[str, Any], horizons: Tuple[int, ...] = (1, 5, 21)
    ) -> Dict[int, float]:
        """Forecasts annualized conditional volatility over multiple horizons.

        Args:
            fit_output: Output dictionary from fit_gjr_garch.
            horizons: Tuple of forecast horizon days (e.g., 1, 5, 21).

        Returns:
            Dictionary mapping horizon day integer to annualized forecast volatility.
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
                avg_var = np.mean(variance_forecasts[:h])
                ann_vol = np.sqrt(avg_var * 252)
                results[h] = float(ann_vol)

            return results
        except Exception as e:
            logger.error(f"Error forecasting GJR-GARCH volatility: {e}")
            daily_vol = fit_output.get("last_volatility_daily", 0.01)
            return {h: float(daily_vol * np.sqrt(252)) for h in horizons}

    def _fit_ewma_fallback(self, returns: pd.Series, lambda_param: float = 0.94) -> Dict[str, Any]:
        """Calculates Exponentially Weighted Moving Average (EWMA) volatility as fallback.

        Args:
            returns: Daily return pandas Series.
            lambda_param: EWMA decay factor (default 0.94).

        Returns:
            Dictionary matching GJR-GARCH fit output structure.
        """
        clean_ret = returns.dropna()
        n = len(clean_ret)
        if n == 0:
            empty_s = pd.Series(dtype=float)
            return {
                "success": True,
                "model_name": "EWMA Fallback (lambda=0.94)",
                "omega": 0.0,
                "alpha": 0.06,
                "gamma": 0.0,
                "beta": 0.94,
                "persistence": 1.0,
                "last_volatility_daily": 0.01,
                "last_volatility_annualized": 0.15,
                "conditional_volatility": empty_s,
                "annualized_volatility": empty_s,
                "scale_factor": 1.0,
            }

        var = np.zeros(n)
        var[0] = clean_ret.var() if clean_ret.var() > 0 else 0.0001

        for t in range(1, n):
            var[t] = lambda_param * var[t - 1] + (1.0 - lambda_param) * (clean_ret.iloc[t - 1] ** 2)

        daily_vol = pd.Series(np.sqrt(var), index=clean_ret.index)
        ann_vol = daily_vol * np.sqrt(252)

        return {
            "success": True,
            "model_name": "EWMA Fallback (lambda=0.94)",
            "omega": 0.0,
            "alpha": 1.0 - lambda_param,
            "gamma": 0.0,
            "beta": lambda_param,
            "persistence": 1.0,
            "last_volatility_daily": float(daily_vol.iloc[-1]),
            "last_volatility_annualized": float(ann_vol.iloc[-1]),
            "conditional_volatility": daily_vol,
            "annualized_volatility": ann_vol,
            "scale_factor": 1.0,
        }


# Class Alias for Backwards Compatibility
VolatilityEngine = VolatilityModeler

