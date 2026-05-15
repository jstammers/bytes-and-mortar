"""Tests for the inference helpers used by the model-exploration notebook."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.models.property_price import predict


def _fake_pipeline(point_log: float):
    pipe = MagicMock()
    pipe.predict.side_effect = lambda X: np.full(len(X), point_log, dtype=float)
    return pipe


def test_predict_price_inverts_log1p():
    pipe = _fake_pipeline(np.log1p(250_000))
    X = pd.DataFrame({"x": [0, 0, 0]})
    out = predict.predict_price(pipe, X)
    assert out.shape == (3,)
    assert np.allclose(out, 250_000)


def test_bootstrap_distribution_centres_on_point_estimate():
    pipe = _fake_pipeline(np.log1p(500_000))
    X = pd.DataFrame({"x": [0]})
    samples = predict.bootstrap_distribution(pipe, X, np.zeros(100), n=2000, seed=0)
    assert samples.shape == (2000,)
    assert np.allclose(samples, 500_000)


def test_bootstrap_distribution_uses_residual_pool():
    """With residuals = +/- 0.1 in log space, samples bracket the point estimate."""
    pipe = _fake_pipeline(np.log1p(400_000))
    X = pd.DataFrame({"x": [0]})
    residuals = np.array([-0.1, 0.1])
    samples = predict.bootstrap_distribution(pipe, X, residuals, n=10_000, seed=0)
    expected_lo = np.expm1(np.log1p(400_000) - 0.1)
    expected_hi = np.expm1(np.log1p(400_000) + 0.1)
    unique = np.unique(np.round(samples, 4))
    assert np.allclose(sorted(unique), sorted([expected_lo, expected_hi]), rtol=1e-4)


def test_bootstrap_requires_single_row():
    pipe = _fake_pipeline(0.0)
    X = pd.DataFrame({"x": [0, 0]})
    with pytest.raises(ValueError, match="exactly one row"):
        predict.bootstrap_distribution(pipe, X, np.zeros(3), n=10)


def test_bootstrap_requires_non_empty_residuals():
    pipe = _fake_pipeline(0.0)
    X = pd.DataFrame({"x": [0]})
    with pytest.raises(ValueError, match="residuals_log is empty"):
        predict.bootstrap_distribution(pipe, X, np.array([]), n=10)


def test_holdout_residuals_log():
    """Residuals are y_true_log - y_pred_log."""
    pipe = _fake_pipeline(np.log1p(100_000))
    X = pd.DataFrame({"x": [0, 0, 0]})
    y_true_log = np.array([np.log1p(100_000), np.log1p(120_000), np.log1p(80_000)])
    residuals = predict.holdout_residuals_log(pipe, X, y_true_log)
    assert residuals[0] == pytest.approx(0.0)
    assert residuals[1] > 0
    assert residuals[2] < 0


def test_load_pipeline_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="No trained pipeline"):
        predict.load_pipeline(tmp_path / "missing.joblib")
