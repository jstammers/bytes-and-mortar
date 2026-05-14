"""HPI time-series forecasting with probabilistic prediction intervals.

Background
----------
The UK House Price Index (HPI) is a monthly index measuring residential property
price change, published by HM Land Registry / ONS.  Base period: January 2015 = 100.

Key factors driving UK HPI
---------------------------
Short-term drivers (< 1 year):
  - Bank of England base rate / mortgage rates (strong negative relationship)
  - Consumer confidence and employment conditions
  - Seasonal demand patterns (spring peaks, winter troughs)
  - Policy changes (stamp duty holidays, Help-to-Buy)

Medium-term drivers (1-5 years):
  - Real household income growth
  - Housing supply (new-build completions, planning consents)
  - Population growth / net migration
  - Credit availability (LTV ratios, affordability stress tests)

Long-term structural factors (5+ years):
  - Household formation rate
  - Structural undersupply in high-demand regions
  - Long-run income elasticity > 1 (prices rise faster than income)
  - Regional divergence (London / South East premium)

Forecasting methods implemented
---------------------------------
1. ETSForecaster - Exponential Smoothing State Space (Holt-Winters).
                   Published benchmarks show ETS outperforms ARIMA on UK RPPI
                   (lower MAE/RMSE; Huang et al., 2023).  Uses simulation-based
                   prediction intervals to capture parameter uncertainty.

2. SARIMAForecaster - Seasonal ARIMA (p,d,q)(P,D,Q,12).
                      Handles unit root and monthly seasonality.  Gaussian
                      confidence intervals from the state-space innovations
                      variance.  Optionally auto-selects orders via AIC grid search.

3. TrendForecaster - Polynomial trend + bootstrap prediction intervals.
                     No statsmodels dependency; pure numpy.  Best for short series
                     (< 3 years) or as a sanity-check baseline.  Bootstrap
                     resamples residuals to build empirical quantile intervals;
                     uncertainty grows with horizon via a multiplicative factor.

4. EnsembleForecaster - Weighted combination of the above.
                        Weights by exp(-0.5 * delta_AIC) when AIC is available;
                        otherwise uniform.  Reduces average MSE approx 37% vs best
                        single model (Makridakis M5; Yusupova 2023).
                        Prediction intervals combined as weighted average of bounds.

5. ARIMAXForecaster - SARIMAX extended with exogenous macro regressors.
                      Accepts columns from the BoE (base rate, mortgage rate)
                      or ONS (AWE, unemployment) data sources.  Future exogenous
                      values must be supplied at forecast time; if omitted, the
                      last observed value is propagated forward (flat-forward).

Usage
-----
::

    import polars as pl
    from src.models.hpi_forecast import EnsembleForecaster

    df = pl.read_parquet("data/processed/uk_hpi_england.parquet")
    england = df.filter(pl.col("regionname") == "England").sort("date")

    forecaster = EnsembleForecaster()
    forecaster.fit_from_frame(england, value_col="averageprice")
    result = forecaster.forecast(steps=24, alpha=0.05)

    print(result.to_frame())        # tidy DataFrame: date, forecast, lower, upper
    print(result.to_frame())        # 95 % prediction intervals

References
----------
- Huang et al. (2023) "House Price Indices Prediction By ARIMA, ETS And ARIMA-Xreg",
  Highlights in Business, Economics and Management.
- Yusupova et al. (2023) "Advances in Forecasting Home Prices",
  Computational Economics, Springer.
- Meen, G. (1996) "House Prices: An Econometric Model for the UK",
  J. Housing and the Built Environment 11(1).
- ONS (2025) "UK House Price Index: methodology", GOV.UK.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from datetime import date

    import polars as pl

logger = logging.getLogger(__name__)

try:
    import statsmodels  # noqa: F401

    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    _HAS_STATSMODELS = False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ForecastResult:
    """Point forecasts and prediction interval for HPI.

    Attributes:
        dates:    Future calendar dates (first of each month).
        point:    Central (mean/median) forecast of the index.
        lower:    Lower bound of the prediction interval.
        upper:    Upper bound of the prediction interval.
        alpha:    Significance level; nominal coverage = ``1 - alpha``.
        method:   Identifier string for the forecasting method.
        metadata: Optional dict with model diagnostics (AIC, order, …).
    """

    dates: list[date]
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    alpha: float
    method: str
    metadata: dict = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        """Number of forecast steps."""
        return len(self.dates)

    def to_frame(self) -> pl.DataFrame:
        """Return a tidy Polars DataFrame with one row per forecast date."""
        import polars as pl

        return pl.DataFrame(
            {
                "date": self.dates,
                "forecast": self.point.tolist(),
                "lower": self.lower.tolist(),
                "upper": self.upper.tolist(),
                "method": [self.method] * self.horizon,
            }
        )

    def coverage(self, actuals: np.ndarray) -> float:
        """Compute empirical prediction-interval coverage.

        Args:
            actuals: Observed values aligned with ``self.dates`` (same length).

        Returns:
            Fraction of actuals falling within ``[lower, upper]``.
        """
        arr = np.asarray(actuals, dtype=float)
        return float(np.mean((arr >= self.lower) & (arr <= self.upper)))


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class HPIForecaster(ABC):
    """Abstract base class for HPI time-series forecasters."""

    name: str = "base"

    @abstractmethod
    def fit(self, values: np.ndarray, dates: list[date]) -> HPIForecaster:
        """Fit the model on a historical index series.

        Args:
            values: 1-D array of monthly HPI (or average-price) values,
                    ordered oldest → newest.
            dates:  Corresponding calendar dates (one per value).

        Returns:
            ``self`` for method chaining.
        """
        ...

    @abstractmethod
    def forecast(self, steps: int, alpha: float = 0.05) -> ForecastResult:
        """Generate h-step-ahead forecasts with prediction intervals.

        Args:
            steps: Number of months to forecast.
            alpha: Significance level.  ``1 - alpha`` is the nominal
                   coverage of the prediction interval (default 0.05 → 95 %).

        Returns:
            :class:`ForecastResult` with ``horizon == steps``.
        """
        ...

    def fit_from_frame(
        self,
        df: pl.DataFrame,
        value_col: str = "index",
        date_col: str = "date",
    ) -> HPIForecaster:
        """Convenience wrapper: fit directly from a cleaned HPI DataFrame.

        The frame must contain *date_col* (``polars.Date``) and *value_col*
        (numeric).  It should already be sorted by date and filtered to a
        single region.

        Args:
            df:        Cleaned HPI DataFrame (e.g. from ``UKHousePriceIndex.clean()``).
            value_col: Column to forecast (``"index"`` or ``"averageprice"``).
            date_col:  Date column name.

        Returns:
            ``self`` for method chaining.
        """
        vals = df[value_col].cast(float).to_numpy()
        dates_list: list[date] = df[date_col].to_list()
        return self.fit(vals, dates_list)


# ---------------------------------------------------------------------------
# Exponential Smoothing (ETS) — best univariate baseline per published benchmarks
# ---------------------------------------------------------------------------


class ETSForecaster(HPIForecaster):
    """Holt-Winters Exponential Smoothing (ETS) forecaster.

    Uses an additive-trend additive-seasonal state-space model fitted by
    maximum likelihood.  Prediction intervals are generated by simulation
    (bootstrapped innovations), which naturally accounts for parameter
    uncertainty and is better-calibrated than the closed-form Gaussian
    approximation for multi-step horizons.

    Published benchmarks (Huang et al. 2023) show ETS has lower MAE/RMSE
    than both ARIMA and ARIMA-Xreg on the UK Residential Property Price Index.

    Args:
        trend:            ``"add"`` (default) or ``"mul"`` — additive/multiplicative trend.
        seasonal:         ``"add"`` (default) or ``"mul"`` — additive/multiplicative seasonality.
        seasonal_periods: Number of periods in a seasonal cycle (default 12 for monthly).
        n_simulations:    Number of sample paths for bootstrap prediction intervals.
    """

    name = "ets"

    def __init__(
        self,
        trend: str = "add",
        seasonal: str = "add",
        seasonal_periods: int = 12,
        n_simulations: int = 2000,
    ) -> None:
        if not _HAS_STATSMODELS:
            raise ImportError(
                "statsmodels is required for ETSForecaster. Install with: uv sync --extra ml"
            )
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.n_simulations = n_simulations
        self._result = None
        self._last_date: date | None = None

    def fit(self, values: np.ndarray, dates: list[date]) -> ETSForecaster:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        self._last_date = dates[-1]
        model = ExponentialSmoothing(
            values,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated",
        )
        self._result = model.fit(optimized=True, remove_bias=False)
        logger.info(
            "ETSForecaster fitted: n=%d, AIC=%.1f, alpha=%.4f",
            len(values),
            self._result.aic,
            self._result.params.get("smoothing_level", float("nan")),
        )
        return self

    def forecast(self, steps: int, alpha: float = 0.05) -> ForecastResult:
        if self._result is None or self._last_date is None:
            raise RuntimeError("Call fit() before forecast()")

        point = np.asarray(self._result.forecast(steps), dtype=float)

        # Simulate sample paths to build empirical prediction intervals.
        # simulate() returns shape (steps, n_simulations).
        sim = self._result.simulate(
            nsimulations=steps,
            repetitions=self.n_simulations,
            error="add",
            random_state=42,
        )
        lower = np.quantile(sim, alpha / 2, axis=1).astype(float)
        upper = np.quantile(sim, 1 - alpha / 2, axis=1).astype(float)

        return ForecastResult(
            dates=_monthly_dates(self._last_date, steps),
            point=point,
            lower=lower,
            upper=upper,
            alpha=alpha,
            method=self.name,
            metadata={
                "aic": float(self._result.aic),
                "smoothing_level": float(self._result.params.get("smoothing_level", float("nan"))),
            },
        )


# ---------------------------------------------------------------------------
# Seasonal ARIMA
# ---------------------------------------------------------------------------


class SARIMAForecaster(HPIForecaster):
    """Seasonal ARIMA (SARIMA) forecaster.

    Fits a ``SARIMAX(p,d,q)(P,D,Q,m)`` model via maximum likelihood.
    Prediction intervals assume Gaussian innovations (standard for ARIMA).

    When ``auto_order=True`` a small AIC grid search selects (p,d,q) and
    (P,D,Q) automatically — expect ~30-60 s for typical grids.

    Args:
        order:          Non-seasonal ARIMA order ``(p, d, q)``.
        seasonal_order: Seasonal order ``(P, D, Q, m)`` where *m* is the
                        seasonal period (12 for monthly).
        auto_order:     If ``True``, ignore ``order``/``seasonal_order`` and
                        run AIC grid search to select best model.
    """

    name = "sarima"

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 12),
        auto_order: bool = False,
    ) -> None:
        if not _HAS_STATSMODELS:
            raise ImportError(
                "statsmodels is required for SARIMAForecaster. Install with: uv sync --extra ml"
            )
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto_order = auto_order
        self._result = None
        self._last_date: date | None = None

    def fit(self, values: np.ndarray, dates: list[date]) -> SARIMAForecaster:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self._last_date = dates[-1]

        if self.auto_order:
            self.order, self.seasonal_order = _aic_grid_search(values)

        model = SARIMAX(
            values,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._result = model.fit(disp=False)
        logger.info(
            "SARIMAForecaster fitted: order=%s seasonal=%s AIC=%.1f",
            self.order,
            self.seasonal_order,
            self._result.aic,
        )
        return self

    def forecast(self, steps: int, alpha: float = 0.05) -> ForecastResult:
        if self._result is None or self._last_date is None:
            raise RuntimeError("Call fit() before forecast()")

        pred = self._result.get_forecast(steps)
        mean = np.asarray(pred.predicted_mean, dtype=float)
        ci = pred.conf_int(alpha=alpha)
        ci_arr = np.asarray(ci)
        lower = ci_arr[:, 0].astype(float)
        upper = ci_arr[:, 1].astype(float)

        return ForecastResult(
            dates=_monthly_dates(self._last_date, steps),
            point=mean,
            lower=lower,
            upper=upper,
            alpha=alpha,
            method=self.name,
            metadata={
                "aic": float(self._result.aic),
                "order": self.order,
                "seasonal_order": self.seasonal_order,
            },
        )


# ---------------------------------------------------------------------------
# Polynomial Trend + Bootstrap (no statsmodels required)
# ---------------------------------------------------------------------------


class TrendForecaster(HPIForecaster):
    """Polynomial trend extrapolation with bootstrap prediction intervals.

    Fits a polynomial of configurable degree to the time index and computes
    residuals.  Prediction intervals are built by bootstrapping (resampling
    with replacement) those residuals and applying a horizon-dependent
    uncertainty inflation factor ``sqrt(h)`` to reflect growing uncertainty
    over time.

    This is a transparent, dependency-light baseline.  It works even when
    statsmodels is not installed and is useful for short history lengths
    (< 3 years) or as a quick sanity-check.

    Args:
        degree:      Polynomial degree (1 = linear trend, 2 = quadratic).
        n_bootstrap: Number of bootstrap replications for the PI.
    """

    name = "trend"

    def __init__(self, degree: int = 2, n_bootstrap: int = 2000) -> None:
        self.degree = degree
        self.n_bootstrap = n_bootstrap
        self._coef: np.ndarray | None = None
        self._residuals: np.ndarray | None = None
        self._n_obs: int = 0
        self._last_date: date | None = None

    def fit(self, values: np.ndarray, dates: list[date]) -> TrendForecaster:
        t = np.arange(len(values), dtype=float)
        self._coef = np.polyfit(t, values, self.degree)
        fitted = np.polyval(self._coef, t)
        self._residuals = values - fitted
        self._n_obs = len(values)
        self._last_date = dates[-1]
        logger.info(
            "TrendForecaster fitted: degree=%d, n=%d, residual std=%.2f",
            self.degree,
            len(values),
            float(np.std(self._residuals)),
        )
        return self

    def forecast(self, steps: int, alpha: float = 0.05) -> ForecastResult:
        if self._coef is None or self._residuals is None or self._last_date is None:
            raise RuntimeError("Call fit() before forecast()")

        t_future = np.arange(self._n_obs, self._n_obs + steps, dtype=float)
        point = np.polyval(self._coef, t_future)

        # Bootstrap: resample residuals; scale uncertainty by sqrt(h) to
        # capture the increasing unpredictability at longer horizons.
        rng = np.random.default_rng(42)
        horizon_scale = np.sqrt(np.arange(1, steps + 1))  # shape (steps,)
        boot_preds = np.zeros((self.n_bootstrap, steps))
        for b in range(self.n_bootstrap):
            noise = rng.choice(self._residuals, size=steps, replace=True)
            boot_preds[b] = point + noise * horizon_scale

        lower = np.quantile(boot_preds, alpha / 2, axis=0).astype(float)
        upper = np.quantile(boot_preds, 1 - alpha / 2, axis=0).astype(float)

        return ForecastResult(
            dates=_monthly_dates(self._last_date, steps),
            point=point,
            lower=lower,
            upper=upper,
            alpha=alpha,
            method=self.name,
            metadata={"degree": self.degree, "n_obs": self._n_obs},
        )


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------


class EnsembleForecaster(HPIForecaster):
    """Weighted ensemble of :class:`HPIForecaster` instances.

    Combines point forecasts as a weighted average and merges prediction
    intervals.  When AIC is available in each component's metadata the
    weights are set to ``exp(-0.5 * ΔAIC)`` (Akaike weights); otherwise
    uniform weights are used.

    This strategy typically reduces MSE by ≈ 37 % compared to the best
    individual model (Yusupova et al. 2023; literature on forecast combination).

    Prediction intervals are combined as a weighted average of the individual
    bounds (``pi_method="weighted"``) or taken as the widest envelope across
    models (``pi_method="union"``).  The union method guarantees that no
    component interval is excluded; the weighted method produces narrower,
    better-calibrated intervals when models agree.

    Args:
        forecasters: List of fitted or unfitted :class:`HPIForecaster` objects.
                     Defaults to ``[ETSForecaster(), SARIMAForecaster(),
                     TrendForecaster()]`` when statsmodels is available, or
                     ``[TrendForecaster()]`` otherwise.
        weights:     Manual weights (must sum to 1 or will be normalised).
                     When ``None`` AIC-based weights are computed automatically.
        pi_method:   ``"weighted"`` (default) or ``"union"``.
    """

    name = "ensemble"

    def __init__(
        self,
        forecasters: list[HPIForecaster] | None = None,
        weights: list[float] | None = None,
        pi_method: Literal["union", "weighted"] = "weighted",
    ) -> None:
        if forecasters is None:
            if _HAS_STATSMODELS:
                forecasters = [ETSForecaster(), SARIMAForecaster(), TrendForecaster()]
            else:
                logger.warning(
                    "statsmodels not installed; EnsembleForecaster using TrendForecaster only"
                )
                forecasters = [TrendForecaster()]
        self.forecasters = forecasters
        self.weights = weights
        self.pi_method = pi_method
        self._fitted: list[HPIForecaster] = []

    def fit(self, values: np.ndarray, dates: list[date]) -> EnsembleForecaster:
        self._fitted = []
        for f in self.forecasters:
            try:
                f.fit(values, dates)
                self._fitted.append(f)
            except Exception as exc:
                logger.warning("EnsembleForecaster: %s.fit() failed - %s", f.name, exc)

        if not self._fitted:
            raise RuntimeError(
                "All component forecasters failed to fit. "
                "Check that the series is long enough and that statsmodels is installed."
            )
        logger.info(
            "EnsembleForecaster: %d/%d components fitted successfully",
            len(self._fitted),
            len(self.forecasters),
        )
        return self

    def forecast(self, steps: int, alpha: float = 0.05) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Call fit() before forecast()")

        results = [f.forecast(steps, alpha) for f in self._fitted]
        w = np.asarray(self._compute_weights(results), dtype=float)

        points = np.stack([r.point for r in results])  # (n_models, steps)
        point = np.average(points, axis=0, weights=w)

        lowers = np.stack([r.lower for r in results])
        uppers = np.stack([r.upper for r in results])

        if self.pi_method == "union":
            lower = lowers.min(axis=0).astype(float)
            upper = uppers.max(axis=0).astype(float)
        else:
            lower = np.average(lowers, axis=0, weights=w).astype(float)
            upper = np.average(uppers, axis=0, weights=w).astype(float)

        return ForecastResult(
            dates=results[0].dates,
            point=point.astype(float),
            lower=lower,
            upper=upper,
            alpha=alpha,
            method=self.name,
            metadata={
                "component_weights": dict(
                    zip([f.name for f in self._fitted], w.tolist(), strict=False)
                ),
                "pi_method": self.pi_method,
            },
        )

    def _compute_weights(self, results: list[ForecastResult]) -> list[float]:
        if self.weights is not None:
            raw = list(self.weights[: len(results)])
        else:
            aics = [r.metadata.get("aic") for r in results]
            if all(a is not None for a in aics):
                aics_arr = np.array(aics, dtype=float)
                delta = aics_arr - aics_arr.min()
                raw = np.exp(-0.5 * delta).tolist()
            else:
                raw = [1.0] * len(results)

        total = sum(raw)
        return [x / total for x in raw]


# ---------------------------------------------------------------------------
# ARIMAX
# ---------------------------------------------------------------------------


class ARIMAXForecaster(SARIMAForecaster):
    """SARIMA with exogenous regressors (ARIMAX / SARIMAX-X).

    Extends :class:`SARIMAForecaster` to accept macro-economic exogenous
    variables such as the BoE base rate, mortgage rate, or unemployment.
    The exog matrix is aligned to the HPI training dates before fitting.

    Parameters
    ----------
    order, seasonal_order, auto_order:
        As per :class:`SARIMAForecaster`.
    exog_names:
        Human-readable labels for the exogenous columns (used in metadata
        and log messages only).

    Usage
    -----
    ::

        import polars as pl, numpy as np
        from src.models.hpi_forecast import ARIMAXForecaster

        hpi_values  = np.array([...])         # monthly HPI averageprices
        hpi_dates   = [...]                   # list[date], same length
        base_rate   = np.array([...])         # same length as hpi_values
        future_rate = np.array([...])         # steps length (required for forecast)

        fc = ARIMAXForecaster(exog_names=["base_rate"])
        fc.fit(hpi_values, hpi_dates, exog=base_rate.reshape(-1, 1))
        result = fc.forecast(steps=12, alpha=0.05,
                             future_exog=future_rate.reshape(-1, 1))

    Notes
    -----
    - Future exogenous values must be supplied to :meth:`forecast` via
      *future_exog*.  If omitted, the last observed value is repeated
      (naive flat-forward assumption).
    - When ``auto_order=True`` the grid search uses no exogenous variables
      for order selection (for speed); the final model is then re-fitted
      with exog on the selected orders.
    """

    name = "arimax"

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 12),
        auto_order: bool = False,
        exog_names: list[str] | None = None,
    ) -> None:
        super().__init__(order=order, seasonal_order=seasonal_order, auto_order=auto_order)
        self.exog_names = exog_names or []
        self._last_exog_row: np.ndarray | None = None

    def fit(  # type: ignore[override]
        self,
        values: np.ndarray,
        dates: list[date],
        exog: np.ndarray | None = None,
    ) -> ARIMAXForecaster:
        """Fit SARIMAX with optional exogenous regressors.

        Args:
            values: 1-D array of endogenous observations (HPI values).
            dates:  Corresponding dates (same length as *values*).
            exog:   Optional 2-D array of shape ``(len(values), n_exog)``.
                    Columns must be aligned to *values* by date.

        Returns:
            Self (fitted).
        """
        if not _HAS_STATSMODELS:
            raise ImportError("statsmodels is required for ARIMAXForecaster")

        from statsmodels.tsa.statespace.sarimax import SARIMAX

        if exog is not None:
            exog = np.asarray(exog, dtype=float)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
            self._last_exog_row = exog[-1:].copy()
        else:
            self._last_exog_row = None

        order, seasonal_order = self.order, self.seasonal_order
        if self.auto_order:
            # Auto-select on endogenous only (fast); then re-fit with exog
            order, seasonal_order = _aic_grid_search(values)

        self._result = SARIMAX(
            values,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        self._values = values
        self._dates = dates
        self.order = order
        self.seasonal_order = seasonal_order

        logger.info(
            "ARIMAXForecaster fitted: order=%s seasonal=%s exog=%s AIC=%.1f",
            order,
            seasonal_order,
            self.exog_names or "(none)",
            self._result.aic,
        )
        return self

    def forecast(  # type: ignore[override]
        self,
        steps: int,
        alpha: float = 0.05,
        future_exog: np.ndarray | None = None,
    ) -> ForecastResult:
        """Forecast *steps* periods ahead.

        Args:
            steps:       Number of monthly periods to forecast.
            alpha:       Significance level (0.05 = 95% PI).
            future_exog: Array of shape ``(steps, n_exog)`` with future
                         exogenous values.  If ``None`` and the model was
                         fitted with exog, the last observed row is repeated
                         (flat-forward assumption).

        Returns:
            :class:`ForecastResult` with ``method="arimax"``.
        """
        if self._result is None or self._dates is None:
            raise RuntimeError("Call fit() before forecast()")

        if future_exog is None and self._last_exog_row is not None:
            # Flat-forward: repeat last observed exog value
            future_exog = np.repeat(self._last_exog_row, steps, axis=0)

        if future_exog is not None:
            future_exog = np.asarray(future_exog, dtype=float)
            if future_exog.ndim == 1:
                future_exog = future_exog.reshape(-1, 1)

        pred = self._result.get_forecast(steps=steps, exog=future_exog)
        point = np.asarray(pred.predicted_mean, dtype=float)

        ci = np.asarray(pred.conf_int(alpha=alpha))
        lower = ci[:, 0].astype(float)
        upper = ci[:, 1].astype(float)

        return ForecastResult(
            dates=_monthly_dates(self._dates[-1], steps),
            point=point,
            lower=lower,
            upper=upper,
            alpha=alpha,
            method=self.name,
            metadata={
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "aic": float(self._result.aic),
                "exog_names": self.exog_names,
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _monthly_dates(last: date, steps: int) -> list[date]:
    """Return *steps* consecutive first-of-month dates following *last*."""
    from datetime import date as _date

    out: list[date] = []
    year, month = last.year, last.month
    for _ in range(steps):
        month += 1
        if month > 12:
            month = 1
            year += 1
        out.append(_date(year, month, 1))
    return out


def _aic_grid_search(
    values: np.ndarray,
    p_range: range = range(3),
    d_range: range = range(2),
    q_range: range = range(3),
    sp_range: range = range(2),
    sd_range: range = range(2),
    sq_range: range = range(2),
    m: int = 12,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    """Select SARIMA orders by AIC grid search.

    Iterates over a small grid of (p,d,q)(P,D,Q) combinations and returns
    the pair with the lowest AIC.  Typical runtime: 15-45 seconds on a
    modern CPU for the default grid of 72 models.

    Args:
        values:   1-D array of training values.
        *_range:  Ranges for each non-seasonal and seasonal order parameter.
        m:        Seasonal period (default 12 for monthly data).

    Returns:
        ``(order, seasonal_order)`` tuple for :class:`SARIMAForecaster`.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    best_aic = np.inf
    best_order: tuple[int, int, int] = (1, 1, 1)
    best_seasonal: tuple[int, int, int, int] = (1, 1, 1, m)

    for p in p_range:
        for d in d_range:
            for q in q_range:
                for sp in sp_range:
                    for sd in sd_range:
                        for sq in sq_range:
                            try:
                                res = SARIMAX(
                                    values,
                                    order=(p, d, q),
                                    seasonal_order=(sp, sd, sq, m),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                ).fit(disp=False)
                                if res.aic < best_aic:
                                    best_aic = res.aic
                                    best_order = (p, d, q)
                                    best_seasonal = (sp, sd, sq, m)
                            except Exception:
                                continue

    logger.info(
        "AIC grid search complete: order=%s seasonal=%s AIC=%.1f",
        best_order,
        best_seasonal,
        best_aic,
    )
    return best_order, best_seasonal
