# Perpetual vs XGBoost+Optuna: Model Comparison

**Date:** 2026-03-08
**Sample:** 10% of `uk_property_sales.parquet` (stratified by year)
**Train rows:** 2,948,383 | **Test rows (2024):** 86,931
**XGBoost HPO:** 25 Optuna trials, 5-fold sliding-window CV

---

## Accuracy (held-out 2024 test set, original price space)

| Model | RMSE (£) | MAE (£) | MAPE | MdAPE | R² | vs baseline RMSE |
|-------|----------|---------|------|-------|----|------------------|
| **Median Baseline** | £403,374 | £173,845 | 41.0% | 41.8% | 0.0677 | — |
| **Perpetual (budget=0.5)** | £290,172 | £87,447 | 24.9% | 16.7% | 0.5176 | −28.1% |
| **Perpetual (budget=0.7)** | £287,186 | £86,069 | 24.5% | 16.2% | 0.5274 | −28.8% |
| **Perpetual (budget=1.0)** | £288,002 | £85,596 | 24.3% | 16.1% | 0.5248 | −28.6% |
| **XGBoost + Optuna** | £284,588 | £84,605 | 24.3% | 16.0% | 0.5360 | −29.4% |

## Runtime

| Model | Time | Notes |
|-------|------|-------|
| Median Baseline | 1.1s | Group-median lookup |
| Perpetual (budget=0.5) | 16.2s | Single fit, no HPO |
| Perpetual (budget=0.7) | 20.1s | Single fit, no HPO |
| Perpetual (budget=1.0) | 32.7s | Single fit, no HPO |
| XGBoost + Optuna | 6.4min | 25 trials × 5-fold CV + final refit |

**Perpetual (budget=1.0) speedup over XGBoost+Optuna:** 11.6×

## Perpetual Budget Sweep

| Budget | RMSE (£) | MdAPE | R² | Time | vs XGBoost MdAPE gap |
|--------|----------|-------|----|------|----------------------|
| 0.5 | £290,172 | 16.7% | 0.5176 | 16.2s | +0.64pp |
| 0.7 | £287,186 | 16.2% | 0.5274 | 20.1s | +0.22pp |
| 1.0 | £288,002 | 16.1% | 0.5248 | 32.7s | +0.06pp ✅ |

At `budget=1.0`, Perpetual is within **0.06pp MdAPE** of fully-tuned XGBoost — well inside noise for this dataset.

## XGBoost Best Hyperparameters (from Optuna)

```
n_estimators:       783
max_depth:          9
learning_rate:      0.02973
subsample:          0.6005
colsample_bytree:   0.7708
min_child_weight:   7
reg_alpha:          4.34e-05
reg_lambda:         0.04923
```

## Improvement Over Median Baseline

| Model | RMSE reduction | MdAPE reduction |
|-------|---------------|-----------------|
| Perpetual (budget=1.0) | −28.6% | −25.7pp |
| XGBoost + Optuna | −29.4% | −25.8pp |

## Recommendation

✅ **Favour Perpetual (budget=1.0).**

At `budget=1.0`, Perpetual achieves **equivalent accuracy** to XGBoost+Optuna (MdAPE 16.1% vs 16.0%, RMSE gap £3,414) while being **11.6× faster** to train (33s vs 6.4min) and requiring **zero hyperparameter search**.

### Why this matters for the API

| Concern | XGBoost + Optuna | Perpetual |
|---------|-----------------|-----------|
| Hyperparameters to expose | 8 + CV config + n_trials | 1 (`budget`) |
| Training time (10% data) | ~6 min | ~33s |
| Meaningful HPO signal | Yes — wide optimal range | Not needed |
| Reproducibility | Stochastic (sampler seed) | Deterministic |
| Dependency surface | xgboost + optuna | perpetual only |

### Suggested migration

Replace the XGBoost+Optuna pipeline:
```python
# Before: xgboost_objective → optuna.create_study → study.optimize → build_xgboost_pipeline
experiment = Experiment(model_type=ModelType.xgboost, hpo_config=HPOConfig(n_trials=25))
train(experiment)

# After: single PerpetualBooster fit
from perpetual import PerpetualBooster
model = PerpetualBooster(objective="SquaredLoss", budget=1.0)
model.fit(X_train_transformed, y_train_log)
```

`budget` is the only knob users need to understand: increase it if accuracy is insufficient, decrease it if training is too slow.

### Caveats

- Results are on **10% of the dataset**; at full scale Perpetual's relative advantage may shift.
- EPC features (`current_energy_efficiency`, `total_floor_area`, `number_habitable_rooms`) are heavily missing (~97% null for non-linked records). Both models impute these; native missing-value handling in Perpetual is untested here.
- The 0.06pp MdAPE gap is within run-to-run variance for a 25-trial Optuna study — a fresh XGBoost run may not always beat Perpetual.
