"""Tests for src/models/evaluate.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.evaluate import (
    RegressionMetrics,
    compute_metrics,
    cv_metrics_summary,
    evaluate_on_test,
)


@pytest.fixture
def perfect_predictions() -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array([100_000, 200_000, 300_000, 400_000, 500_000], dtype=float)
    return y_true, y_true.copy()


@pytest.fixture
def noisy_predictions() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y_true = np.linspace(100_000, 900_000, 100)
    noise = rng.normal(0, 50_000, 100)
    return y_true, y_true + noise


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_perfect_predictions_zero_error(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        metrics = compute_metrics(y_true, y_pred)
        assert metrics.rmse == pytest.approx(0.0, abs=1e-6)
        assert metrics.mae == pytest.approx(0.0, abs=1e-6)
        assert metrics.mape == pytest.approx(0.0, abs=1e-6)
        assert metrics.r2 == pytest.approx(1.0, abs=1e-6)

    def test_returns_regression_metrics_instance(self, noisy_predictions):
        y_true, y_pred = noisy_predictions
        metrics = compute_metrics(y_true, y_pred)
        assert isinstance(metrics, RegressionMetrics)

    def test_n_samples_correct(self, noisy_predictions):
        y_true, y_pred = noisy_predictions
        metrics = compute_metrics(y_true, y_pred)
        assert metrics.n_samples == len(y_true)

    def test_drops_nan_rows(self):
        y_true = np.array([100_000, np.nan, 300_000], dtype=float)
        y_pred = np.array([110_000, 200_000, 290_000], dtype=float)
        metrics = compute_metrics(y_true, y_pred)
        assert metrics.n_samples == 2

    def test_rmse_greater_than_mae(self, noisy_predictions):
        """RMSE >= MAE always (Jensen's inequality)."""
        y_true, y_pred = noisy_predictions
        metrics = compute_metrics(y_true, y_pred)
        assert metrics.rmse >= metrics.mae

    def test_mape_positive(self, noisy_predictions):
        y_true, y_pred = noisy_predictions
        metrics = compute_metrics(y_true, y_pred)
        assert metrics.mape >= 0

    def test_to_dict_keys(self, noisy_predictions):
        y_true, y_pred = noisy_predictions
        d = compute_metrics(y_true, y_pred).to_dict()
        assert set(d.keys()) == {"rmse", "mae", "mape", "mdape", "r2", "n_samples"}

    def test_summary_returns_string(self, noisy_predictions):
        y_true, y_pred = noisy_predictions
        metrics = compute_metrics(y_true, y_pred)
        summary = metrics.summary("test")
        assert isinstance(summary, str)
        assert "RMSE" in summary


# ---------------------------------------------------------------------------
# evaluate_on_test
# ---------------------------------------------------------------------------


class TestEvaluateOnTest:
    def test_returns_model_and_baseline_keys(self):
        y = np.log1p(np.array([200_000, 300_000, 400_000], dtype=float))
        result = evaluate_on_test(
            model_pred=y + 0.01,
            baseline_pred=y + 0.1,
            y_true=y,
            log_transformed=True,
        )
        assert "model" in result
        assert "baseline" in result

    def test_model_beats_baseline_when_closer(self):
        y_true = np.log1p(np.array([200_000, 300_000, 400_000], dtype=float))
        model_pred = y_true + 0.01  # very close
        baseline_pred = y_true + 0.5  # far off

        result = evaluate_on_test(model_pred, baseline_pred, y_true, log_transformed=True)
        assert result["model"].rmse < result["baseline"].rmse

    def test_no_log_transform_path(self):
        y_true = np.array([200_000.0, 300_000.0, 400_000.0])
        result = evaluate_on_test(
            model_pred=y_true * 1.05,
            baseline_pred=y_true * 1.3,
            y_true=y_true,
            log_transformed=False,
        )
        assert result["model"].rmse < result["baseline"].rmse


# ---------------------------------------------------------------------------
# cv_metrics_summary
# ---------------------------------------------------------------------------


class TestCVMetricsSummary:
    def test_empty_list_returns_empty_dict(self):
        assert cv_metrics_summary([]) == {}

    def test_summary_contains_mean_and_std(self):
        folds = [
            compute_metrics(
                np.linspace(100_000, 500_000, 50),
                np.linspace(100_000, 500_000, 50) + i * 5_000,
            )
            for i in range(3)
        ]
        summary = cv_metrics_summary(folds)
        assert "cv_rmse_mean" in summary
        assert "cv_rmse_std" in summary
        assert "cv_r2_mean" in summary

    def test_single_fold_std_is_zero(self):
        metrics = compute_metrics(
            np.array([200_000.0, 300_000.0]),
            np.array([210_000.0, 295_000.0]),
        )
        summary = cv_metrics_summary([metrics])
        assert summary["cv_rmse_std"] == pytest.approx(0.0, abs=1e-6)
