"""Configuration for UK property data ingestion pipeline."""

from pathlib import Path

import polars as pl

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

# --- HM Land Registry Price Paid Data ---
# Full historical file (~4GB CSV, transactions from 1995 to present)
LAND_REGISTRY_BASE_URL = (
    "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
)
LAND_REGISTRY_COMPLETE_CSV = f"{LAND_REGISTRY_BASE_URL}/pp-complete.csv"
# Current year file (smaller, only current year)
LAND_REGISTRY_YEARLY_CSV = f"{LAND_REGISTRY_BASE_URL}/pp-{{year}}.csv"
# Monthly update file
LAND_REGISTRY_MONTHLY_CSV = f"{LAND_REGISTRY_BASE_URL}/pp-monthly-update-new-version.csv"

LAND_REGISTRY_COLUMNS = [
    "transaction_id",
    "price",
    "date_of_transfer",
    "postcode",
    "property_type",
    "old_new",
    "duration",
    "paon",
    "saon",
    "street",
    "locality",
    "town_city",
    "district",
    "county",
    "ppd_category",
    "record_status",
]

PROPERTY_TYPE_MAP = {
    "D": "Detached",
    "S": "Semi-Detached",
    "T": "Terraced",
    "F": "Flats/Maisonettes",
    "O": "Other",
}

OLD_NEW_MAP = {
    "Y": "New build",
    "N": "Established",
}

DURATION_MAP = {
    "F": "Freehold",
    "L": "Leasehold",
    "U": "Unknown",
}

PPD_CATEGORY_MAP = {
    "A": "Standard Price Paid",
    "B": "Additional Price Paid",
}

# --- EPC (Energy Performance Certificate) Data ---
EPC_API_BASE_URL = "https://epc.opendatacommunities.org/api/v1"
EPC_DOMESTIC_SEARCH = f"{EPC_API_BASE_URL}/domestic/search"
# EPC API requires free registration for an API key.
# Set EPC_API_TOKEN in your .env file.
# Register at: https://epc.opendatacommunities.org/login

EPC_KEY_FIELDS = [
    "address1",
    "address2",
    "address3",
    "postcode",
    "building-reference-number",
    "current-energy-rating",
    "current-energy-efficiency",
    "potential-energy-rating",
    "potential-energy-efficiency",
    "property-type",
    "built-form",
    "inspection-date",
    "lodgement-date",
    "transaction-type",
    "environment-impact-current",
    "environment-impact-potential",
    "energy-consumption-current",
    "energy-consumption-potential",
    "co2-emissions-current",
    "co2-emissions-potential",
    "total-floor-area",
    "number-habitable-rooms",
    "number-heated-rooms",
    "floor-level",
    "main-heating-controls",
    "multi-glaze-proportion",
    "glazed-type",
    "number-open-fireplaces",
    "hotwater-description",
    "floor-description",
    "windows-description",
    "walls-description",
    "roof-description",
    "mainheat-description",
    "main-fuel",
    "construction-age-band",
    "tenure",
]

# Polars schema for domestic EPC certificates based on the glossary of terms.
EPC_DOMESTIC_SCHEMA = {
    "lmk-key": pl.Utf8,
    "address1": pl.Utf8,
    "address2": pl.Utf8,
    "address3": pl.Utf8,
    "postcode": pl.Utf8,
    "building-reference-number": pl.Utf8,
    "current-energy-rating": pl.Utf8,
    "potential-energy-rating": pl.Utf8,
    "current-energy-efficiency": pl.Int64,
    "potential-energy-efficiency": pl.Int64,
    "property-type": pl.Utf8,
    "built-form": pl.Utf8,
    "inspection-date": pl.Date,
    "local-authority": pl.Utf8,
    "constituency": pl.Utf8,
    "county": pl.Utf8,
    "lodgement-date": pl.Date,
    "transaction-type": pl.Utf8,
    "environment-impact-current": pl.Int64,
    "environment-impact-potential": pl.Int64,
    "energy-consumption-current": pl.Float64,
    "energy-consumption-potential": pl.Float64,
    "co2-emissions-current": pl.Float64,
    "co2-emiss-curr-per-floor-area": pl.Float64,
    "co2-emissions-potential": pl.Float64,
    "lighting-cost-current": pl.Float64,
    "lighting-cost-potential": pl.Float64,
    "heating-cost-current": pl.Float64,
    "heating-cost-potential": pl.Float64,
    "hot-water-cost-current": pl.Float64,
    "hot-water-cost-potential": pl.Float64,
    "total-floor-area": pl.Float64,
    "energy-tariff": pl.Utf8,
    "mains-gas-flag": pl.Utf8,
    "floor-level": pl.Utf8,
    "flat-top-storey": pl.Utf8,
    "flat-storey-count": pl.Float64,
    "main-heating-controls": pl.Utf8,
    "multi-glaze-proportion": pl.Float64,
    "glazed-type": pl.Utf8,
    "glazed-area": pl.Utf8,
    "extension-count": pl.Float64,
    "number-habitable-rooms": pl.Float64,
    "number-heated-rooms": pl.Float64,
    "low-energy-lighting": pl.Int64,
    "number-open-fireplaces": pl.Int64,
    "hotwater-description": pl.Utf8,
    "hot-water-energy-eff": pl.Utf8,
    "hot-water-env-eff": pl.Utf8,
    "floor-description": pl.Utf8,
    "floor-energy-eff": pl.Utf8,
    "floor-env-eff": pl.Utf8,
    "windows-description": pl.Utf8,
    "windows-energy-eff": pl.Utf8,
    "windows-env-eff": pl.Utf8,
    "walls-description": pl.Utf8,
    "walls-energy-eff": pl.Utf8,
    "walls-env-eff": pl.Utf8,
    "secondheat-description": pl.Utf8,
    "sheating-energy-eff": pl.Utf8,
    "sheating-env-eff": pl.Utf8,
    "roof-description": pl.Utf8,
    "roof-energy-eff": pl.Utf8,
    "roof-env-eff": pl.Utf8,
    "mainheat-description": pl.Utf8,
    "mainheat-energy-eff": pl.Utf8,
    "mainheat-env-eff": pl.Utf8,
    "mainheatcont-description": pl.Utf8,
    "mainheatc-energy-eff": pl.Utf8,
    "mainheatc-env-eff": pl.Utf8,
    "lighting-description": pl.Utf8,
    "lighting-energy-eff": pl.Utf8,
    "lighting-env-eff": pl.Utf8,
    "main-fuel": pl.Utf8,
    "wind-turbine-count": pl.Float64,
    "heat-loss-corridor": pl.Utf8,
    "unheated-corridor-length": pl.Float64,
    "floor-height": pl.Float64,
    "photo-supply": pl.Float64,
    "solar-water-heating-flag": pl.Utf8,
    "mechanical-ventilation": pl.Utf8,
    "address": pl.Utf8,
    "local-authority-label": pl.Utf8,
    "constituency-label": pl.Utf8,
    "posttown": pl.Utf8,
    "construction-age-band": pl.Utf8,
    "lodgement-datetime": pl.Datetime,
    "tenure": pl.Utf8,
    "fixed-lighting-outlets-count": pl.Float64,
    "low-energy-fixed-light-count": pl.Float64,
    "uprn": pl.Int64,
    "uprn-source": pl.Utf8,
    "report-type": pl.Int64,
}

# --- UK House Price Index ---
UK_HPI_BASE_URL = "https://publicdata.landregistry.gov.uk/market-trend-data"
UK_HPI_DOWNLOAD_URL = f"{UK_HPI_BASE_URL}/house-price-index-data/UK-HPI-full-file-2025-04.csv"
# SPARQL endpoint for programmatic access
UK_HPI_SPARQL_ENDPOINT = "http://landregistry.data.gov.uk/landregistry/query"

# --- Chunk sizes for large file downloads ---
DOWNLOAD_CHUNK_SIZE = 8192  # bytes
CSV_CHUNK_SIZE = 50_000  # rows per chunk when reading large CSVs

# --- ML model paths ---
MODELS_DIR = PROJECT_DIR / "models"
MLRUNS_DIR = PROJECT_DIR / "mlruns"
PROCESSED_DATA_PATH = PROCESSED_DIR / "uk_property_sales.parquet"
