# bytes-and-mortar justfile
# Run `just` or `just --list` to see available commands

# Install all dependencies (including dev)
install:
    uv sync --extra dev

# Run the test suite
test *args='':
    uv run pytest tests/ {{ args }}

# Run tests with verbose output
test-v:
    uv run pytest tests/ -v

# Run ruff linter
lint:
    uv run ruff check src/ tests/

# Auto-fix lint issues
lint-fix:
    uv run ruff check --fix src/ tests/

# Check formatting
fmt-check:
    uv run ruff format --check src/ tests/

# Format all source files
fmt:
    uv run ruff format src/ tests/

# Run ty type checker
typecheck:
    uv run ty check src/

# Run all checks (lint + format + typecheck + tests)
check: lint fmt-check typecheck test

# Clean compiled Python files and caches
clean:
    find . -type f -name "*.py[co]" -delete
    find . -type d -name "__pycache__" -delete
    rm -rf .pytest_cache .ruff_cache

# Download Land Registry data for a given year
download-lr year='':
    uv run bytes-and-mortar download-land-registry {{ if year != '' { "--year " + year } else { "" } }}

# Download EPC data for a local authority
download-epc la='':
    uv run bytes-and-mortar download-epc {{ if la != '' { "--local-authority " + la } else { "" } }}

# Download UK House Price Index data
download-hpi:
    uv run bytes-and-mortar download-hpi

# Run the full ingestion pipeline
run *args='':
    uv run bytes-and-mortar run {{ args }}

# Install ML dependencies (scikit-learn, xgboost, optuna, mlflow)
install-ml:
    uv sync --extra dev --extra ml

# Install notebook dependencies (marimo, pymc)
install-notebooks:
    uv sync --extra dev --extra ml --extra notebooks

# Train a model (default: xgboost, sliding window CV, 50 HPO trials)
train *args='':
    uv run bytes-and-mortar train run {{ args }}

# Launch MLflow UI (uses sqlite backend for model registry support)
mlflow-ui:
    uv run mlflow ui --backend-store-uri sqlite:///mlruns.db
# Install app (API) dependencies
install-app:
    uv sync --extra app

# Start the FastAPI development server
dev-api:
    uv run uvicorn src.app.main:app --reload --port 8000

# Install frontend dependencies
install-frontend:
    cd frontend && npm install

# Start the frontend development server (requires install-frontend first)
dev-frontend:
    cd frontend && npm run dev

# Build frontend for production
build-frontend:
    cd frontend && npm run build
# Create M19 postcode fixtures for integration tests (requires data/raw/ files)
create-fixtures src_dir='':
    PYTHONPATH=. uv run python scripts/create_m19_fixtures.py {{ if src_dir != '' { "--src-dir " + src_dir } else { "" } }}

# Run integration tests only (requires fixtures — run create-fixtures first)
test-integration *args='':
    uv run pytest tests/test_integration_pipeline.py -m integration -v {{ args }}
