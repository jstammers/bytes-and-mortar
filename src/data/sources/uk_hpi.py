"""UK House Price Index (UK HPI) data ingestion.

Data source: https://www.gov.uk/government/statistical-data-sets/uk-house-price-index-data-downloads-may-2025
Linked data: http://landregistry.data.gov.uk/app/ukhpi
License: Open Government Licence v3.0

The UK HPI is calculated by the ONS using data from HM Land Registry,
Registers of Scotland, and Land and Property Services Northern Ireland.
Published monthly, covering England, Scotland, Wales and Northern Ireland.
"""

import logging
from pathlib import Path

import polars as pl

from src.data.config import UK_HPI_DOWNLOAD_URL
from src.data.sources.base import DataSource

logger = logging.getLogger(__name__)


class UKHousePriceIndex(DataSource):
    """Ingest UK House Price Index data."""

    name = "uk_hpi"

    def download(self, url: str | None = None, **kwargs) -> Path:
        """Download UK HPI full dataset CSV.

        Args:
            url: Override the default download URL if a newer version
                 is available from GOV.UK.
        """
        download_url = url or UK_HPI_DOWNLOAD_URL
        dest = self.raw_dir / "uk_hpi_full.csv"
        return self._download_file(download_url, dest, desc="UK HPI")

    def load(self, filepath: Path | None = None, **kwargs) -> pl.DataFrame:
        """Load UK HPI CSV into DataFrame."""
        if filepath is None:
            filepath = self.raw_dir / "uk_hpi_full.csv"

        if not filepath.exists():
            raise FileNotFoundError(
                f"UK HPI data file not found: {filepath}. Run download() first."
            )

        logger.info("Loading UK HPI data from %s", filepath)
        df = pl.read_csv(filepath, infer_schema_length=10000)
        logger.info("Loaded %d UK HPI records", len(df))
        return df

    def clean(self, df: pl.DataFrame) -> pl.DataFrame:
        """Clean and standardise UK HPI data."""
        logger.info("Cleaning UK HPI data (%d rows)", len(df))

        # Standardise column names
        df = df.rename({col: col.strip().lower().replace(" ", "_") for col in df.columns})

        # Parse date column
        if "date" in df.columns:
            df = df.with_columns(pl.col("date").str.to_date(format="%Y-%m-%d", strict=False))
            df = df.with_columns(
                [
                    pl.col("date").dt.year().alias("year"),
                    pl.col("date").dt.month().alias("month"),
                ]
            )

        # Ensure numeric columns are numeric
        price_cols = [
            c
            for c in df.columns
            if any(
                kw in c
                for kw in [
                    "average_price",
                    "index",
                    "percentage_change",
                    "sales_volume",
                ]
            )
        ]
        for col in price_cols:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

        # Drop rows with no region/area name
        area_col = None
        for candidate in ["regionname", "region_name", "areacode", "area_code"]:
            if candidate in df.columns:
                area_col = candidate
                break
        if area_col:
            df = df.drop_nulls(subset=[area_col])

        logger.info("After cleaning: %d rows", len(df))
        return df

    def get_area_prices(self, df: pl.DataFrame, area_name: str) -> pl.DataFrame:
        """Filter UK HPI data to a specific area/region."""
        area_col = None
        for candidate in ["regionname", "region_name"]:
            if candidate in df.columns:
                area_col = candidate
                break
        if area_col is None:
            raise KeyError("No region/area name column found in UK HPI data")

        return df.filter(pl.col(area_col).str.contains(f"(?i){area_name}"))
