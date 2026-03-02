"""Configuration for UK property data ingestion pipeline."""

from pathlib import Path

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

# --- UK House Price Index ---
UK_HPI_BASE_URL = "http://publicdata.landregistry.gov.uk/market-trend"
UK_HPI_DOWNLOAD_URL = f"{UK_HPI_BASE_URL}/house-price-index-data/UK-HPI-full-file-2025-03.csv"
# SPARQL endpoint for programmatic access
UK_HPI_SPARQL_ENDPOINT = "http://landregistry.data.gov.uk/landregistry/query"

# --- Chunk sizes for large file downloads ---
DOWNLOAD_CHUNK_SIZE = 8192  # bytes
CSV_CHUNK_SIZE = 50_000  # rows per chunk when reading large CSVs
