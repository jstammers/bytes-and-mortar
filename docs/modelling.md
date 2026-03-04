# Modelling: UK Property Price Prediction

This document describes the design philosophy, modelling pipeline, and usage instructions for the `src/models` and `src/features` modules.

---

## Overview

The modelling layer sits downstream of the data pipeline. It reads the linked parquet output from `bytes-and-mortar run` and trains a regression model to predict residential sale price.

```
data/processed/uk_property_sales.parquet
        │
        ▼
  Feature Engineering          src/features/build_features.py
  ──────────────────
  • Column selection & validation
  • Missing value handling
  • Ordinal encoding (low-cardinality categoricals)
  • Target encoding (district, ~300 categories)
  • Standard scaling (numerics)
        │
        ▼
  Temporal Train/Test Split    src/models/cv.py
  ─────────────────────────
  • Hold out one or more calendar years as final test set
  • Log class-distribution balance by property_type
        │
        ▼
  Optuna HPO + Time-Series CV  src/models/train_model.py
  ───────────────────────────
  • 3 CV strategies (sliding window, expanding window, year-based)
  • Each Optuna trial is a nested MLflow child run
  • Optimise CV RMSE in log-price space
        │
        ▼
  Retrain on Full Training Set
        │
        ▼
  Evaluate vs Baseline         src/models/evaluate.py
  ────────────────────
  • RMSE, MAE, MAPE, MdAPE, R²
  • Compare against MedianByGroupBaseline(property_type × district)
        │
        ▼
  Log to MLflow + Save Locally  src/models/registry.py
  ────────────────────────────
  • Params, metrics, model artefact per run
  • Optional: push to MLflow model registry
  • Always: save .joblib file to models/
```

---

## Feature Engineering

### Configuration

Feature selection is driven by `FeatureConfig`, which is constructed by the CLI and wired into every downstream component. The defaults cover the most informative columns available in the linked dataset:

| Group | Columns | Transformer |
|---|---|---|
| **Numeric** | `year`, `month`, `total_floor_area`, `current_energy_efficiency`, `number_habitable_rooms`, `averageprice` | `StandardScaler` |
| **Categorical** | `property_type`, `old_new`, `duration`, `county`, `postcode_outward` | `OrdinalEncoder` |
| **Target-encoded** | `district` | sklearn `TargetEncoder` (CV=5) |

Override any group from the CLI:

```bash
bytes-and-mortar train run \
  --numeric-features "year,month,total_floor_area" \
  --categorical-features "property_type,old_new"
```

All columns are validated against the actual DataFrame schema before training begins, raising a clear `ValueError` listing any missing columns.

### Missing value strategies

EPC data is only available for ~50–70% of sales (the join is a left join on postcode + address). The three strategies handle this differently:

| Strategy | Behaviour | Best for |
|---|---|---|
| `impute` (default) | Median imputation for numerics, most-frequent for categoricals | Linear models; all models when EPC coverage is low |
| `drop` | Remove rows with any missing feature | Clean comparison; reduces dataset size |
| `passthrough` | Leave `NaN` in place | XGBoost only — it handles missing values natively via the `hist` tree method |

```bash
bytes-and-mortar train run --missing-strategy passthrough --model-type xgboost
```

### Target encoding for `district`

`district` has ~300 unique values and is the strongest geographic signal available. One-hot encoding would add 300 sparse columns; label encoding would impose a false ordinal relationship. Target encoding maps each district to a smoothed mean of `log(price)` computed with cross-validation inside the training fold, preventing target leakage.

For the Bayesian model (see below), district is encoded as an integer index and given a hierarchical prior — a more statistically principled alternative.

### Log transformation of price

UK house prices are strongly right-skewed (£50k bedsits to £10M penthouses). A `log1p` transformation maps the distribution to approximate normality, which is a prerequisite for the linear model and also benefits XGBoost by making the loss landscape more balanced. All metrics are reported in original GBP space after `expm1` inversion.

---

## Model Architectures

### 1. Ridge regression on log-price

**File:** `src/models/linear.py`

A simple baseline linear model: sklearn `Ridge` trained on `log1p(price)`. Ridge (L2 regularisation) is preferred over plain OLS because:

- UK property features are correlated (floor area, number of rooms, energy efficiency)
- Regularisation prevents overfitting on smaller regional datasets

**Hyperparameter search space:**

| Parameter | Range |
|---|---|
| `alpha` (regularisation strength) | Log-uniform [1e-3, 1e3] |

Linear models require imputed inputs — if `missing_strategy=passthrough` is specified for a linear model, the pipeline automatically falls back to `impute` with a warning.

### 2. XGBoost

**File:** `src/models/xgboost_model.py`

XGBoost with the `hist` tree method. Handles missing values natively (no imputation required), captures non-linear feature interactions, and is typically the strongest single model on tabular property data.

**Hyperparameter search space:**

| Parameter | Range |
|---|---|
| `n_estimators` | int [100, 1000] |
| `max_depth` | int [3, 10] |
| `learning_rate` | Log-uniform [1e-3, 0.3] |
| `subsample` | [0.6, 1.0] |
| `colsample_bytree` | [0.6, 1.0] |
| `min_child_weight` | int [1, 10] |
| `reg_alpha` (L1) | Log-uniform [1e-8, 10] |
| `reg_lambda` (L2) | Log-uniform [1e-8, 10] |

### 3. Bayesian hierarchical model (experimental)

**File:** `notebooks/pymc_property_price.py`

A PyMC-based Bayesian hierarchical regression explored in a [marimo](https://marimo.io) interactive notebook. The hierarchy is:

```
price_i ~ LogNormal(μ_i, σ)
μ_i  = α[district_i] + β_floor · floor_area_i + β_eff · energy_eff_i
α[d] ~ Normal(μ_α, σ_α)      ← district-level random intercepts
μ_α  ~ Normal(log(300k), 1.5) ← global mean log-price prior
σ_α  ~ HalfNormal(1)          ← between-district variation
σ    ~ HalfNormal(0.5)         ← within-district noise
```

**Why a hierarchical model?**

Partial pooling allows data-sparse districts (e.g. rural areas with few transactions) to borrow strength from the global distribution, rather than producing unreliable point estimates. This is particularly valuable when training on sub-regional datasets.

**Running the notebook:**

```bash
just install-notebooks
uv run marimo edit notebooks/pymc_property_price.py
```

**Integration path:** The notebook documents a `BayesianPropertyModel` sklearn wrapper stub. Once the experimental phase is complete, this can be promoted to `src/models/bayesian.py` and added to `ModelType` in `src/models/config.py`.

---

## Baseline

Every training run is automatically compared against a `MedianByGroupBaseline`:

- Predicts the median sale price for each `(property_type, district)` pair observed during training
- Fallback chain: group median → `property_type` median → global median (for unseen combinations)
- Logged alongside model metrics under `baseline_*` keys in MLflow

**Interpretation:** A model that does not beat the baseline adds no value over a simple lookup table. This is a minimum viable bar, not a high one.

---

## Cross-Validation Strategies

Property data has a strong temporal structure — prices in 2024 are correlated with prices in 2023 but not with prices in 2010 in the same way. Standard k-fold CV would allow future data to leak into training folds, producing optimistic estimates. All three supported strategies respect temporal order:

### Sliding window (default)

A fixed-size training window advances one fold at a time. Each fold uses approximately the same amount of training data, making fold-to-fold comparisons fair.

```
Fold 1:  [──train──]  [gap]  [val]
Fold 2:        [──train──]  [gap]  [val]
Fold 3:              [──train──]  [gap]  [val]
```

Best for: comparing model architectures on equal footing.

### Expanding window

Training set grows with each fold — all history up to the cutoff. Rewards models that benefit from more data.

```
Fold 1:  [──train────────────]  [gap]  [val]
Fold 2:  [────────train──────────────]  [gap]  [val]
Fold 3:  [──────────────train──────────────────]  [gap]  [val]
```

Best for: assessing whether the model improves with more historical data.

### Year-based

One calendar year is held out per fold. Training set is all transactions before the held-out year minus the gap.

```
Fold 1:  [───all prior years───]  [gap]  [2022]
Fold 2:  [───all prior years───]  [gap]  [2023]
Fold 3:  [───all prior years───]  [gap]  [2024]
```

Best for: year-over-year performance reporting.

### The `gap_months` parameter

All strategies support a `gap_months` buffer (default: 1) between the end of the training window and the start of the validation window. Land Registry transactions can take weeks to register after completion — a gap prevents boundary-period leakage.

---

## Hyperparameter Optimisation

[Optuna](https://optuna.org) is used for HPO with the Tree-structured Parzen Estimator (TPE) sampler. Each trial:

1. Samples a hyperparameter configuration
2. Runs time-series cross-validation on the training set
3. Returns mean CV RMSE (log-price space) as the objective
4. Reports intermediate fold results to Optuna's pruner (median pruning) so poor trials are stopped early
5. Logs params and CV RMSE to a nested MLflow child run

The best configuration is then retrained on the full training set before final evaluation on the held-out test years.

---

## Evaluation Metrics

All metrics are computed in **original GBP price space** (after `expm1` inversion) for interpretability:

| Metric | Formula | Interpretation |
|---|---|---|
| **RMSE** | √(mean((ŷ − y)²)) | Penalises large errors heavily; sensitive to outliers |
| **MAE** | mean(\|ŷ − y\|) | Average error in £; interpretable and robust |
| **MAPE** | mean(\|ŷ − y\| / y) | Scale-independent; useful across price ranges |
| **MdAPE** | median(\|ŷ − y\| / y) | Robust version of MAPE; less affected by outlier properties |
| **R²** | 1 − SS_res/SS_tot | Proportion of variance explained |

---

## MLflow Experiment Tracking

Each `bytes-and-mortar train run` invocation creates:

- A **parent run** containing all experiment params, CV summary metrics, and the final model artefact
- **Child runs** for each Optuna trial (params + CV RMSE)

Experiments are organised as `<name>/<model_type>` (e.g. `uk_property_price/xgboost`) so different architectures can be filtered and compared in the MLflow UI.

### Launching the UI

```bash
just mlflow-ui
# Opens at http://localhost:5000
```

> The `just mlflow-ui` recipe uses a SQLite backend (`mlruns.db`) which supports the full MLflow model registry. The default `file://` store supports artefact logging only.

### Model registry

To register the best model after training:

```bash
bytes-and-mortar train run --register --mlflow-uri sqlite:///mlruns.db
```

Registered models can be loaded by name and stage:

```python
from src.models.registry import load_model

pipeline = load_model("uk_property_price", stage="Production")
predictions = pipeline.predict(X_new)
```

---

## CLI Reference

```
bytes-and-mortar train run [OPTIONS]

Options:
  -m, --model-type     [linear|xgboost]    Model architecture (default: xgboost)
  --data-path          PATH                Path to processed parquet (default: auto)
  --nrows              INT                 Limit rows loaded for development
  --test-years         TEXT                Comma-separated holdout years (default: 2024)
  --cv-strategy        [sliding_window|    Time-series CV strategy
                        expanding_window|  (default: sliding_window)
                        year_based]
  --cv-n-splits        INT                 Number of CV folds (default: 5)
  --cv-gap-months      INT                 Months gap between train/val (default: 1)
  --cv-window-months   INT                 Training window in months (default: 24)
  --n-trials           INT                 Optuna HPO trials (default: 50)
  --timeout            INT                 Max HPO time in seconds
  --missing-strategy   [impute|drop|       Missing value handling
                        passthrough]       (default: impute)
  --log-transform /    --no-log-transform  Log1p-transform price target (default: on)
  --register /         --no-register       Register in MLflow registry (default: off)
  --mlflow-uri         TEXT                MLflow tracking URI
  --numeric-features   TEXT                Comma-separated numeric feature columns
  --categorical-features TEXT              Comma-separated categorical feature columns
```

---

## Python API

```python
from src.models.config import Experiment, CVConfig, HPOConfig, CVStrategy, ModelType
from src.features.build_features import FeatureConfig, MissingStrategy
from src.models.train_model import train, predict

experiment = Experiment(
    model_type=ModelType.xgboost,
    feature_config=FeatureConfig(
        numeric_features=["year", "month", "total_floor_area", "current_energy_efficiency"],
        categorical_features=["property_type", "old_new", "duration"],
        target_encode_features=["district"],
        missing_strategy=MissingStrategy.impute,
    ),
    cv_config=CVConfig(
        strategy=CVStrategy.sliding_window,
        n_splits=5,
        gap_months=1,
        window_months=24,
    ),
    hpo_config=HPOConfig(n_trials=100),
    test_years=[2024],
    register_model=True,
)

pipeline, metrics = train(experiment)
print(metrics.summary("xgboost test"))

# Later: load best model and predict
predictions = predict(input_df, model_name="uk_property_price")
```

---

## Adding a New Model Architecture

1. Create `src/models/<name>.py` with two functions:
   - `build_<name>_pipeline(feature_config, **params) -> Pipeline`
   - `<name>_objective(trial, X_train, y_train, cv_splits, feature_config) -> float`
2. Add `<name>` to `ModelType` in `src/models/config.py`
3. Wire into `_build_objective()` and `_build_best_pipeline()` in `src/models/train_model.py`
4. Add tests in `tests/models/test_train_model.py`

The LightGBM case, for example, would be:

```python
# src/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from src.features.build_features import FeatureConfig, build_feature_pipeline

def build_lightgbm_pipeline(feature_config: FeatureConfig | None = None, **lgb_params) -> Pipeline:
    feature_pipeline = build_feature_pipeline(feature_config or FeatureConfig())
    return Pipeline([
        ("feature_pipeline", feature_pipeline),
        ("model", lgb.LGBMRegressor(**lgb_params)),
    ])
```
