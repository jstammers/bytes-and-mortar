"""Tests for src.models.hpi_forecast — HPI time-series forecasting.

These tests use synthetic data to keep CI fast and avoid any dependency on
downloaded HPI files.  The statsmodels-backed forecasters (ETS, SARIMA) are
skipped when the package is not installed so the basic dev environment can
still run the full test suite with `just test`.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from src.models.hpi_forecast import (
    EnsembleForecaster,
    ForecastResult,
    TrendForecaster,
    _monthly_dates,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_hpi(n: int = 48, seed: int = 0) -> tuple[np.ndarray, list[date]]:
    """Generate a simple upward-trending monthly series with mild seasonality."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    seasonal = 5.0 * np.sin(2 * np.pi * t / 12)
    trend = 100.0 + 0.5 * t
    noise = rng.normal(0, 2, size=n)
    values = trend + seasonal + noise

    constructed: list[date] = []
    year, month = 2020, 1
    for _ in range(n):
        constructed.append(date(year, month, 1))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return values, constructed


# ---------------------------------------------------------------------------
# _monthly_dates helper
# ---------------------------------------------------------------------------


class TestMonthlyDates:
    def test_basic(self) -> None:
        result = _monthly_dates(date(2024, 11, 1), 3)
        assert result == [date(2024, 12, 1), date(2025, 1, 1), date(2025, 2, 1)]

    def test_year_rollover(self) -> None:
        result = _monthly_dates(date(2024, 12, 1), 2)
        assert result == [date(2025, 1, 1), date(2025, 2, 1)]

    def test_length(self) -> None:
        assert len(_monthly_dates(date(2020, 6, 1), 24)) == 24


# ---------------------------------------------------------------------------
# ForecastResult
# ---------------------------------------------------------------------------


class TestForecastResult:
    def _make_result(self, steps: int = 12) -> ForecastResult:
        dates = _monthly_dates(date(2024, 1, 1), steps)
        point = np.linspace(200, 210, steps)
        return ForecastResult(
            dates=dates,
            point=point,
            lower=point - 5,
            upper=point + 5,
            alpha=0.05,
            method="dummy",
        )

    def test_horizon(self) -> None:
        r = self._make_result(12)
        assert r.horizon == 12

    def test_to_frame_columns(self) -> None:
        r = self._make_result(6)
        df = r.to_frame()
        assert set(df.columns) == {"date", "forecast", "lower", "upper", "method"}

    def test_to_frame_shape(self) -> None:
        r = self._make_result(6)
        assert r.to_frame().shape == (6, 5)

    def test_to_frame_method_constant(self) -> None:
        r = self._make_result(4)
        assert r.to_frame()["method"].to_list() == ["dummy"] * 4

    def test_coverage_perfect(self) -> None:
        r = self._make_result(12)
        # Actuals equal to point forecast — all inside interval
        assert r.coverage(r.point) == 1.0

    def test_coverage_none(self) -> None:
        r = self._make_result(12)
        # Actuals far outside interval
        actuals = r.point + 1000
        assert r.coverage(actuals) == 0.0

    def test_pi_ordering(self) -> None:
        r = self._make_result(12)
        assert np.all(r.lower <= r.point)
        assert np.all(r.point <= r.upper)


# ---------------------------------------------------------------------------
# TrendForecaster (no statsmodels dependency)
# ---------------------------------------------------------------------------


class TestTrendForecaster:
    def test_fit_returns_self(self) -> None:
        values, dates = _synthetic_hpi(36)
        f = TrendForecaster()
        assert f.fit(values, dates) is f

    def test_forecast_shape(self) -> None:
        values, dates = _synthetic_hpi(36)
        result = TrendForecaster().fit(values, dates).forecast(12)
        assert result.point.shape == (12,)
        assert result.lower.shape == (12,)
        assert result.upper.shape == (12,)
        assert len(result.dates) == 12

    def test_forecast_pi_ordering(self) -> None:
        values, dates = _synthetic_hpi(36)
        result = TrendForecaster().fit(values, dates).forecast(12)
        assert np.all(result.lower <= result.point)
        assert np.all(result.point <= result.upper)

    def test_forecast_method_name(self) -> None:
        values, dates = _synthetic_hpi(36)
        result = TrendForecaster().fit(values, dates).forecast(6)
        assert result.method == "trend"

    def test_forecast_alpha_stored(self) -> None:
        values, dates = _synthetic_hpi(36)
        result = TrendForecaster().fit(values, dates).forecast(6, alpha=0.10)
        assert result.alpha == pytest.approx(0.10)

    def test_wider_interval_at_long_horizon(self) -> None:
        """PI width should grow with forecast horizon due to sqrt(h) scaling."""
        values, dates = _synthetic_hpi(48)
        result = TrendForecaster().fit(values, dates).forecast(24)
        widths = result.upper - result.lower
        # Width at month 24 should be wider than at month 1
        assert widths[-1] > widths[0]

    def test_forecast_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            TrendForecaster().forecast(6)

    def test_fit_from_frame(self) -> None:
        values, dates = _synthetic_hpi(36)
        df = pl.DataFrame({"date": dates, "index": values.tolist()})
        f = TrendForecaster()
        f.fit_from_frame(df, value_col="index")
        result = f.forecast(6)
        assert result.horizon == 6

    def test_linear_trend(self) -> None:
        values, dates = _synthetic_hpi(36)
        result = TrendForecaster(degree=1).fit(values, dates).forecast(12)
        assert result.point.shape == (12,)

    def test_metadata_contains_degree(self) -> None:
        values, dates = _synthetic_hpi(24)
        result = TrendForecaster(degree=2).fit(values, dates).forecast(6)
        assert result.metadata["degree"] == 2


# ---------------------------------------------------------------------------
# EnsembleForecaster (statsmodels-optional: uses TrendForecaster when absent)
# ---------------------------------------------------------------------------


class TestEnsembleForecaster:
    def test_fit_with_trend_only(self) -> None:
        """Ensemble with a single TrendForecaster should always work."""
        values, dates = _synthetic_hpi(48)
        f = EnsembleForecaster(forecasters=[TrendForecaster()])
        f.fit(values, dates)
        result = f.forecast(12)
        assert result.horizon == 12

    def test_ensemble_two_trend_forecasters(self) -> None:
        values, dates = _synthetic_hpi(48)
        f = EnsembleForecaster(forecasters=[TrendForecaster(degree=1), TrendForecaster(degree=2)])
        f.fit(values, dates)
        result = f.forecast(12)
        assert np.all(result.lower <= result.point)
        assert np.all(result.point <= result.upper)

    def test_uniform_weights_applied(self) -> None:
        values, dates = _synthetic_hpi(48)
        f1 = TrendForecaster(degree=1)
        f2 = TrendForecaster(degree=2)
        ensemble = EnsembleForecaster(
            forecasters=[f1, f2],
            weights=[0.5, 0.5],
        )
        ensemble.fit(values, dates)
        result = ensemble.forecast(6)
        # Point forecast should be midpoint of two trend forecasts
        r1 = f1.forecast(6)
        r2 = f2.forecast(6)
        expected = (r1.point + r2.point) / 2
        np.testing.assert_allclose(result.point, expected, rtol=1e-10)

    def test_union_pi_wider_than_weighted(self) -> None:
        values, dates = _synthetic_hpi(48)
        forecasters = [TrendForecaster(degree=1), TrendForecaster(degree=2)]
        union_f = EnsembleForecaster(forecasters=forecasters, pi_method="union")
        union_f.fit(values, dates)
        union_result = union_f.forecast(12)

        # Re-fit with fresh instances for weighted
        forecasters2 = [TrendForecaster(degree=1), TrendForecaster(degree=2)]
        weighted_f = EnsembleForecaster(forecasters=forecasters2, pi_method="weighted")
        weighted_f.fit(values, dates)
        weighted_result = weighted_f.forecast(12)

        union_width = union_result.upper - union_result.lower
        weighted_width = weighted_result.upper - weighted_result.lower
        assert np.all(union_width >= weighted_width - 1e-9)

    def test_forecast_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            EnsembleForecaster().forecast(6)

    def test_method_name(self) -> None:
        values, dates = _synthetic_hpi(36)
        f = EnsembleForecaster(forecasters=[TrendForecaster()])
        f.fit(values, dates)
        assert f.forecast(6).method == "ensemble"

    def test_metadata_contains_weights(self) -> None:
        values, dates = _synthetic_hpi(36)
        f = EnsembleForecaster(forecasters=[TrendForecaster()])
        f.fit(values, dates)
        result = f.forecast(6)
        assert "component_weights" in result.metadata


# ---------------------------------------------------------------------------
# Statsmodels-backed forecasters (conditional skip)
# ---------------------------------------------------------------------------

statsmodels_missing = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("statsmodels"),
    reason="statsmodels not installed",
)


@statsmodels_missing
class TestETSForecaster:
    def test_fit_and_forecast(self) -> None:
        from src.models.hpi_forecast import ETSForecaster

        values, dates = _synthetic_hpi(48)
        result = ETSForecaster().fit(values, dates).forecast(12)
        assert result.horizon == 12
        assert result.method == "ets"

    def test_pi_ordering(self) -> None:
        from src.models.hpi_forecast import ETSForecaster

        values, dates = _synthetic_hpi(48)
        result = ETSForecaster().fit(values, dates).forecast(12)
        assert np.all(result.lower <= result.upper)

    def test_aic_in_metadata(self) -> None:
        from src.models.hpi_forecast import ETSForecaster

        values, dates = _synthetic_hpi(48)
        result = ETSForecaster().fit(values, dates).forecast(6)
        assert "aic" in result.metadata
        assert np.isfinite(result.metadata["aic"])

    def test_forecast_before_fit_raises(self) -> None:
        from src.models.hpi_forecast import ETSForecaster

        with pytest.raises(RuntimeError, match="fit"):
            ETSForecaster().forecast(6)


@statsmodels_missing
class TestSARIMAForecaster:
    def test_fit_and_forecast(self) -> None:
        from src.models.hpi_forecast import SARIMAForecaster

        values, dates = _synthetic_hpi(60)
        result = SARIMAForecaster().fit(values, dates).forecast(12)
        assert result.horizon == 12
        assert result.method == "sarima"

    def test_pi_ordering(self) -> None:
        from src.models.hpi_forecast import SARIMAForecaster

        values, dates = _synthetic_hpi(60)
        result = SARIMAForecaster().fit(values, dates).forecast(12)
        assert np.all(result.lower <= result.upper)

    def test_aic_in_metadata(self) -> None:
        from src.models.hpi_forecast import SARIMAForecaster

        values, dates = _synthetic_hpi(60)
        result = SARIMAForecaster().fit(values, dates).forecast(6)
        assert "aic" in result.metadata
        assert np.isfinite(result.metadata["aic"])

    def test_custom_order(self) -> None:
        from src.models.hpi_forecast import SARIMAForecaster

        values, dates = _synthetic_hpi(60)
        f = SARIMAForecaster(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
        result = f.fit(values, dates).forecast(6)
        assert result.metadata["order"] == (0, 1, 1)

    def test_forecast_before_fit_raises(self) -> None:
        from src.models.hpi_forecast import SARIMAForecaster

        with pytest.raises(RuntimeError, match="fit"):
            SARIMAForecaster().forecast(6)


@statsmodels_missing
class TestEnsembleWithAllForecasters:
    def test_all_three_components(self) -> None:
        from src.models.hpi_forecast import ETSForecaster, EnsembleForecaster, SARIMAForecaster  # noqa: I001

        values, dates = _synthetic_hpi(60)
        f = EnsembleForecaster(forecasters=[ETSForecaster(), SARIMAForecaster(), TrendForecaster()])
        f.fit(values, dates)
        result = f.forecast(12)
        assert result.horizon == 12
        assert len(result.metadata["component_weights"]) == 3

    def test_aic_based_weights_sum_to_one(self) -> None:
        from src.models.hpi_forecast import ETSForecaster, EnsembleForecaster, SARIMAForecaster  # noqa: I001

        values, dates = _synthetic_hpi(60)
        f = EnsembleForecaster(forecasters=[ETSForecaster(), SARIMAForecaster(), TrendForecaster()])
        f.fit(values, dates)
        f.forecast(6)  # triggers weight computation via _compute_weights
        weights = f._compute_weights([c.forecast(6) for c in f._fitted])
        assert pytest.approx(sum(weights), abs=1e-9) == 1.0
