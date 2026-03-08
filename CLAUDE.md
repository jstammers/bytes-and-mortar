# CLAUDE.md

This file provides context to Claude Code about the bytes-and-mortar project.

## Project overview

UK property data ingestion pipeline that downloads, cleans, links, and saves data from three public sources:

- **HM Land Registry Price Paid Data** - property transaction records for England and Wales
- **EPC (Energy Performance Certificates)** - building energy efficiency data (requires free API key)
- **UK House Price Index** - ONS area-level price statistics

## Tech stack

- **Python 3.11+** with **uv** for dependency management
- **pandas** / **polars** for data processing
- **perpetual** for GBM model training (self-tuning, no HPO required)
- **scikit-learn** for pipelines, RidgeCV, preprocessing, evaluation
- **mlflow** for experiment tracking
- **typer** for CLI interface
- **ruff** for linting and formatting
- **ty** for type checking
- **pytest** for testing
- **just** for task running

## Common commands

```bash
just install        # Install all dependencies
just test           # Run tests
just check          # Run all checks (lint + format + typecheck + tests)
just fmt            # Format code
just lint           # Run linter
just typecheck      # Run type checker
```

## Project structure

```
src/
  data/
    config.py           # URLs, column definitions, mappings
    make_dataset.py     # Typer CLI entrypoint
    pipeline.py         # Data linking/merging logic
    sources/
      base.py           # Abstract DataSource base class
      land_registry.py  # HM Land Registry Price Paid
      epc.py            # Energy Performance Certificates
      uk_hpi.py         # UK House Price Index
  features/
    build_features.py   # FeatureConfig, pipeline builder, MedianByGroupBaseline
  models/
    config.py           # Experiment, PerpetualConfig, ModelType dataclasses
    perpetual_model.py  # Perpetual GBM pipeline (primary model)
    linear.py           # RidgeCV pipeline
    cv.py               # Time-series CV utilities
    evaluate.py         # RegressionMetrics, compute_metrics
    registry.py         # MLflow tracking and model registry helpers
    train_model.py      # train() orchestrator + Typer CLI subcommands
tests/                  # pytest tests mirroring src structure
```

## Model training

The primary model is **Perpetual GBM** — a self-tuning gradient booster with a single `budget` hyperparameter. No HPO loop or cross-validation is needed:

```bash
# Default: perpetual, budget=1.0, test_years=2024
just train

# Custom budget
uv run bytes-and-mortar train run --budget 0.5
```

`budget=1.0` matches XGBoost+Optuna accuracy (MdAPE 16.1% vs 16.0%) while being ~12× faster (33s vs 6.4min on 3M rows).

## Code style

- Formatted with ruff (line length 100)
- Type-checked with ty
- All public data sources inherit from `DataSource` base class
- Each source implements `download()`, `load()`, `clean()` methods

## Commit messages

All commits must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <short summary>

[optional body]

[optional footer(s)]
```

**Types:**

| Type | When to use |
|---|---|
| `feat` | New feature or user-facing capability |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change that is neither a fix nor a feature |
| `test` | Adding or correcting tests |
| `chore` | Maintenance, dependency updates, tooling |
| `style` | Formatting, whitespace (no logic change) |
| `perf` | Performance improvement |

**Scopes** (optional but encouraged):

- `app` — FastAPI backend (`src/app/`)
- `frontend` — React frontend (`frontend/`)
- `data` — Data pipeline (`src/data/`)
- `ci` — CI/CD configuration
- `deps` — Dependency changes

**Examples:**

```
feat(app): add valuation sensitivity endpoint
fix(data): handle missing postcode in land registry clean step
docs: add user guide for property valuation app
chore(deps): bump ruff to 0.4.0
```

Breaking changes must include `!` after the type/scope and a `BREAKING CHANGE:` footer.

## Environment variables

- `EPC_API_TOKEN` - required for EPC data. Register at https://epc.opendatacommunities.org/login
