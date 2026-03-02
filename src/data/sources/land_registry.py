"""HM Land Registry Price Paid Data ingestion.

Data source: https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads
License: Open Government Licence v3.0

The Price Paid Data tracks property transactions in England and Wales
received by HM Land Registry for registration since 1 January 1995.
Updated on the 20th working day of each month.

Fields:
    - transaction_id: Unique identifier
    - price: Sale price (GBP)
    - date_of_transfer: Date of transfer
    - postcode: Postcode
    - property_type: D=Detached, S=Semi, T=Terraced, F=Flat, O=Other
    - old_new: Y=New build, N=Established
    - duration: F=Freehold, L=Leasehold, U=Unknown
    - paon: Primary addressable object name (house number/name)
    - saon: Secondary addressable object name (flat number)
    - street, locality, town_city, district, county
    - ppd_category: A=Standard, B=Additional (e.g. repossessions)
    - record_status: A=Addition, C=Change, D=Deletion
"""

import logging
from pathlib import Path

import polars as pl

from src.data.config import (
    CSV_CHUNK_SIZE,
    LAND_REGISTRY_COLUMNS,
    LAND_REGISTRY_COMPLETE_CSV,
    LAND_REGISTRY_YEARLY_CSV,
)
from src.data.sources.base import DataSource

logger = logging.getLogger(__name__)

_LR_SCHEMA = {col: pl.String for col in LAND_REGISTRY_COLUMNS}


class LandRegistryPricePaid(DataSource):
    """Ingest HM Land Registry Price Paid Data."""

    name = "land_registry_price_paid"

    def download(self, year: int | None = None, **kwargs) -> Path:
        """Download Price Paid Data.

        Args:
            year: If provided, download only that year's data.
                  If None, download the complete historical file.
        """
        if year:
            url = LAND_REGISTRY_YEARLY_CSV.format(year=year)
            dest = self.raw_dir / f"pp-{year}.csv"
            desc = f"Price Paid {year}"
        else:
            url = LAND_REGISTRY_COMPLETE_CSV
            dest = self.raw_dir / "pp-complete.csv"
            desc = "Price Paid (complete)"

        return self._download_file(url, dest, desc=desc)

    def load(
        self,
        filepath: Path | None = None,
        nrows: int | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        """Load Price Paid CSV into DataFrame.

        Args:
            filepath: Path to the CSV file. Defaults to complete file.
            nrows: Limit number of rows loaded (useful for testing).
        """
        if filepath is None:
            filepath = self.raw_dir / "pp-complete.csv"

        if not filepath.exists():
            raise FileNotFoundError(
                f"Price Paid Data file not found: {filepath}. Run download() first."
            )

        logger.info("Loading Price Paid Data from %s", filepath)

        # The Land Registry CSV has no header row and values are enclosed in braces: {value}
        # Read all columns as strings so we can strip the braces before casting.
        df = pl.read_csv(
            filepath,
            has_header=False,
            schema=_LR_SCHEMA,
            n_rows=nrows,
        )

        # Strip braces and surrounding whitespace, replace empty strings with null.
        # Apply per-column to avoid duplicate-name errors from type selectors.
        str_cols = LAND_REGISTRY_COLUMNS  # all columns are String at this point
        df = df.with_columns([
            pl.col(c).str.strip_chars("{}").str.strip_chars().alias(c) for c in str_cols
        ])
        df = df.with_columns([
            pl.when(pl.col(c) == "")
            .then(pl.lit(None, dtype=pl.String))
            .otherwise(pl.col(c))
            .alias(c)
            for c in str_cols
        ])

        # Cast price to integer and date to datetime
        df = df.with_columns([
            pl.col("price").cast(pl.Int64, strict=False),
            pl.col("date_of_transfer").str.to_datetime(
                format="%Y-%m-%d %H:%M", strict=False
            ),
        ])

        logger.info("Loaded %d transactions", len(df))
        return df

    def clean(self, df: pl.DataFrame) -> pl.DataFrame:
        """Clean and standardise Price Paid Data."""
        logger.info("Cleaning Price Paid Data (%d rows)", len(df))

        # Drop deletions — we only want current/added records
        if "record_status" in df.columns:
            df = df.filter(pl.col("record_status") != "D")

        # Remove transactions with missing postcodes (can't join to EPC)
        df = df.drop_nulls(subset=["postcode"])

        # Standardise postcode format (uppercase, single space)
        df = df.with_columns(
            pl.col("postcode").str.to_uppercase().str.strip_chars().str.replace_all(r"\s+", " ")
        )

        # Filter to residential property types only
        residential_types = ["D", "S", "T", "F"]
        df = df.filter(pl.col("property_type").is_in(residential_types))

        # Filter out extreme prices (likely errors or commercial)
        df = df.filter((pl.col("price") > 10_000) & (pl.col("price") < 50_000_000))

        # Extract year and month for time-based analysis
        df = df.with_columns([
            pl.col("date_of_transfer").dt.year().alias("year"),
            pl.col("date_of_transfer").dt.month().alias("month"),
        ])

        # Extract outward postcode (area-level grouping)
        df = df.with_columns(
            pl.col("postcode").str.split(" ").list.first().alias("postcode_outward")
        )

        logger.info("After cleaning: %d rows", len(df))
        return df

    def load_chunked(self, filepath: Path, chunk_size: int = CSV_CHUNK_SIZE):
        """Generator that yields cleaned chunks for large files.

        Useful when the complete file is too large for memory.
        """
        reader = pl.read_csv_batched(
            filepath,
            has_header=False,
            schema=_LR_SCHEMA,
            batch_size=chunk_size,
        )
        while True:
            batches = reader.next_batches(1)
            if not batches:
                break
            chunk = batches[0]
            chunk = chunk.with_columns([
                pl.col(c).str.strip_chars("{}").str.strip_chars().alias(c)
                for c in LAND_REGISTRY_COLUMNS
            ])
            chunk = chunk.with_columns([
                pl.when(pl.col(c) == "")
                .then(pl.lit(None, dtype=pl.String))
                .otherwise(pl.col(c))
                .alias(c)
                for c in LAND_REGISTRY_COLUMNS
            ])
            chunk = chunk.with_columns([
                pl.col("price").cast(pl.Int64, strict=False),
                pl.col("date_of_transfer").str.to_datetime(
                    format="%Y-%m-%d %H:%M", strict=False
                ),
            ])
            yield self.clean(chunk)
