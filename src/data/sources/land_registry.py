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

import pandas as pd

from src.data.config import (
    CSV_CHUNK_SIZE,
    LAND_REGISTRY_COLUMNS,
    LAND_REGISTRY_COMPLETE_CSV,
    LAND_REGISTRY_YEARLY_CSV,
)
from src.data.sources.base import DataSource

logger = logging.getLogger(__name__)


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
    ) -> pd.DataFrame:
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

        # The Land Registry CSV has no header row and values are
        # enclosed in braces: {value}
        df = pd.read_csv(
            filepath,
            names=LAND_REGISTRY_COLUMNS,
            header=None,
            nrows=nrows,
            dtype={
                "transaction_id": str,
                "price": "Int64",
                "postcode": str,
                "property_type": str,
                "old_new": str,
                "duration": str,
                "ppd_category": str,
                "record_status": str,
            },
            parse_dates=["date_of_transfer"],
            low_memory=False,
        )

        # Strip braces that Land Registry wraps values in
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].str.strip("{}").str.strip().replace("", pd.NA)

        logger.info("Loaded %d transactions", len(df))
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardise Price Paid Data."""
        logger.info("Cleaning Price Paid Data (%d rows)", len(df))

        # Drop deletions — we only want current/added records
        if "record_status" in df.columns:
            df = df[df["record_status"] != "D"].copy()

        # Remove transactions with missing postcodes (can't join to EPC)
        df = df.dropna(subset=["postcode"])

        # Standardise postcode format (uppercase, single space)
        df["postcode"] = df["postcode"].str.upper().str.strip().str.replace(r"\s+", " ", regex=True)

        # Filter to residential property types only
        residential_types = {"D", "S", "T", "F"}
        df = df[df["property_type"].isin(residential_types)].copy()

        # Filter out extreme prices (likely errors or commercial)
        df = df[(df["price"] > 10_000) & (df["price"] < 50_000_000)].copy()

        # Extract year and month for time-based analysis
        df["year"] = df["date_of_transfer"].dt.year
        df["month"] = df["date_of_transfer"].dt.month

        # Extract outward postcode (area-level grouping)
        df["postcode_outward"] = df["postcode"].str.split(" ").str[0]

        logger.info("After cleaning: %d rows", len(df))
        return df

    def load_chunked(self, filepath: Path, chunk_size: int = CSV_CHUNK_SIZE):
        """Generator that yields cleaned chunks for large files.

        Useful when the complete file is too large for memory.
        """
        for chunk in pd.read_csv(
            filepath,
            names=LAND_REGISTRY_COLUMNS,
            header=None,
            chunksize=chunk_size,
            dtype={
                "transaction_id": str,
                "price": "Int64",
                "postcode": str,
                "property_type": str,
                "old_new": str,
                "duration": str,
                "ppd_category": str,
                "record_status": str,
            },
            parse_dates=["date_of_transfer"],
            low_memory=False,
        ):
            str_cols = chunk.select_dtypes(include="object").columns
            for col in str_cols:
                chunk[col] = chunk[col].str.strip("{}").str.strip().replace("", pd.NA)
            yield self.clean(chunk)
