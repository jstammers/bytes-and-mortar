# bytes-and-mortar

A data pipeline and modelling toolkit for UK property valuation, combining Land Registry transactions, Energy Performance Certificates, and House Price Index data.

## Data sources

| Source | Description | License |
|--------|-------------|---------|
| [HM Land Registry Price Paid](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads) | Property sale transactions in England & Wales since 1995 | OGL v3.0 |
| [EPC Register](https://epc.opendatacommunities.org/) | Energy ratings, floor area, construction age, heating type | OGL v3.0 |
| [UK House Price Index](https://www.gov.uk/government/statistical-data-sets/uk-house-price-index-data-downloads-may-2025) | ONS monthly area-level price indices and volumes | OGL v3.0 |

## Getting started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- [just](https://github.com/casey/just) for task running (optional)

### Installation

```bash
# Data pipeline only
just install

# Data pipeline + ML training (scikit-learn, Perpetual, MLflow)
just install-ml

# All of the above + Bayesian notebook (PyMC, marimo)
just install-notebooks
```

### Configuration

Create a `.env` file in the project root:

```env
EPC_API_TOKEN=your_token_here
```

Register for a free EPC API key at https://epc.opendatacommunities.org/login

## Usage

### 1. Build the dataset

```bash
# Download Land Registry data for 2024
uv run bytes-and-mortar download-land-registry --year 2024

# Download EPC data for Westminster
uv run bytes-and-mortar download-epc --local-authority E09000033

# Download UK House Price Index
uv run bytes-and-mortar download-hpi

# Run the full pipeline (download + clean + link + save)
uv run bytes-and-mortar run --year 2024

# Skip EPC if you don't have an API token
uv run bytes-and-mortar run --year 2024 --skip-epc
```

### 2. Train a model

```bash
# Train with defaults (Perpetual, budget=1.0, 2024 held out)
just train

# Tune budget for faster training or higher accuracy
uv run bytes-and-mortar train run --budget 0.5   # faster, slightly lower accuracy
uv run bytes-and-mortar train run --budget 1.0   # default — best accuracy

# Linear regression (RidgeCV — alpha auto-selected, no HPO needed)
uv run bytes-and-mortar train run --model-type linear

# Fast dev run (1000 rows)
uv run bytes-and-mortar train run --nrows 1000
```

### 3. Explore results in MLflow

```bash
just mlflow-ui
# Opens http://localhost:5000
```

### Using just

```bash
just download-lr 2024           # Download Land Registry for 2024
just download-epc E09000033     # Download EPC for Westminster
just download-hpi               # Download UK HPI
just run --year 2024            # Run full pipeline
just train                      # Train with defaults
just mlflow-ui                  # Launch MLflow experiment browser
```

## Development

```bash
just check       # Run all checks (lint + format + typecheck + tests)
just test        # Run tests only
just test-v      # Run tests with verbose output
just lint        # Run ruff linter
just fmt         # Format code with ruff
just typecheck   # Run ty type checker
just clean       # Remove caches and compiled files
```

## Project structure

```
bytes-and-mortar/
├── src/
│   ├── data/
│   │   ├── config.py               # URLs, column definitions, path constants
│   │   ├── make_dataset.py         # Typer CLI entrypoint (data + train commands)
│   │   ├── pipeline.py             # Data linking and merging logic
│   │   └── sources/
│   │       ├── base.py             # Abstract DataSource base class
│   │       ├── land_registry.py
│   │       ├── epc.py
│   │       └── uk_hpi.py
│   ├── features/
│   │   └── build_features.py       # FeatureConfig, pipeline builder, MedianByGroupBaseline
│   └── models/
│       ├── config.py               # Experiment, PerpetualConfig, ModelType dataclasses
│       ├── cv.py                   # Time-series CV utilities (available for analysis)
│       ├── evaluate.py             # RegressionMetrics, compute_metrics
│       ├── perpetual_model.py      # Perpetual GBM pipeline (primary model)
│       ├── linear.py               # RidgeCV pipeline
│       ├── registry.py             # MLflow tracking and model registry helpers
│       └── train_model.py          # train() orchestrator + Typer CLI subcommands
├── notebooks/
│   └── pymc_property_price.py      # Bayesian hierarchical model (marimo notebook)
├── tests/
│   ├── features/
│   │   └── test_build_features.py
│   ├── models/
│   │   ├── test_cv.py
│   │   ├── test_evaluate.py
│   │   └── test_train_model.py     # Integration tests with synthetic data
│   └── test_pipeline.py, ...
├── data/
│   ├── raw/                        # Downloaded source files
│   ├── interim/                    # Intermediate transforms
│   └── processed/                  # Final linked datasets (parquet)
├── models/                         # Serialised model artefacts (.joblib)
├── docs/
│   └── modelling.md                # Modelling methodology and full CLI reference
├── pyproject.toml                  # Dependencies (core / ml / notebooks / dev)
└── justfile                        # Task runner commands
```

## Pipeline overview

### Data pipeline

1. **Download** raw CSV data from each source
2. **Load** into pandas DataFrames with correct dtypes
3. **Clean** each source (standardise postcodes, parse dates, filter invalid records)
4. **Link** Land Registry sales to EPC records on postcode + address matching
5. **Enrich** with UK HPI area-level statistics on district + year-month
6. **Save** the linked dataset as Parquet or CSV

### Modelling pipeline

1. **Feature engineering** — configurable column selection, missing value handling (drop / impute / passthrough), target encoding for `district`, standard scaling
2. **Temporal split** — hold out one or more calendar years as the final test set
3. **Fit** — `PerpetualBooster(budget=1.0)` self-tunes tree count; no HPO loop required
4. **Evaluate** — RMSE, MAE, MAPE, MdAPE, R² compared against a `MedianByGroupBaseline(property_type × district)`
5. **Log** — all params, metrics, and the fitted model artefact to MLflow; optional push to model registry

See [docs/modelling.md](docs/modelling.md) for the full methodology, model architectures, and extension guide.
