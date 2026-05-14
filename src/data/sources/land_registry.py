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
from typing import cast

import polars as pl

from src.data import manifest
from src.data.config import (
    LAND_REGISTRY_COLUMNS,
    LAND_REGISTRY_COMPLETE_CSV,
    LAND_REGISTRY_MONTHLY_CSV,
    LAND_REGISTRY_YEARLY_CSV,
)
from src.data.sources.base import DataSource

logger = logging.getLogger(__name__)

_LR_SCHEMA: dict[str, type[pl.DataType]] = {col: pl.String for col in LAND_REGISTRY_COLUMNS}


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

    def download_monthly(self) -> Path:
        """Download the latest monthly update file.

        Always re-fetches: the upstream URL is replaced each month with a new
        snapshot of A/C/D rows, so a cached copy at this path is by definition
        stale on the next invocation.
        """
        dest = self.raw_dir / "pp-monthly-update-new-version.csv"
        if dest.exists():
            dest.unlink()
        return self._download_file(LAND_REGISTRY_MONTHLY_CSV, dest, desc="Price Paid monthly")

    def merge_monthly(self, monthly_lf: pl.LazyFrame) -> pl.LazyFrame:
        """Apply a monthly A/C/D delta to the cached complete dataset.

        Returns a LazyFrame of the merged frame in the same shape as ``load()``.
        Does not write to disk — callers (e.g. ``ingest(incremental=True)``)
        are responsible for persisting the result.

        Semantics (record_status, per HM Land Registry):
            A — Addition: insert if not already present.
            C — Change:   replace any existing row with the same transaction_id.
            D — Deletion: drop the row with that transaction_id.
        """
        existing = self.load()  # raw parsed frame from cached parquet/csv
        monthly = cast("pl.DataFrame", monthly_lf.collect())

        deletes = monthly.filter(pl.col("record_status") == "D").select("transaction_id").unique()
        changes = monthly.filter(pl.col("record_status") == "C").select("transaction_id").unique()
        # The monthly file's A and C rows together provide the new authoritative
        # state for those transactions. D rows are pure removals.
        new_rows = monthly.filter(pl.col("record_status").is_in(["A", "C"]))

        # Drop superseded ids (both changes and deletes) from the existing frame.
        superseded = pl.concat([deletes, changes], how="vertical").unique()
        kept_existing = existing.join(superseded.lazy(), on="transaction_id", how="anti")

        merged = pl.concat(
            [kept_existing, new_rows.lazy()],
            how="diagonal_relaxed",
        )
        return merged

    def ingest(self, *, incremental: bool = False, **kwargs) -> pl.LazyFrame:
        """Full ingestion plan.

        With ``incremental=True``: download the monthly delta, merge into the
        cached bulk, rewrite the parquet atomically, update the manifest, then
        return the standard cleaned plan against the refreshed file.
        Otherwise behaves identically to the base implementation.
        """
        if not incremental:
            return super().ingest(**kwargs)

        logger.info("Running incremental update for %s", self.name)
        monthly_path = self.download_monthly()
        monthly_lf = self.load(filepath=monthly_path)
        merged_lf = self.merge_monthly(monthly_lf)
        merged_df = cast("pl.DataFrame", merged_lf.collect())

        target = self.raw_dir / "pp-complete.parquet"
        tmp = target.with_suffix(".parquet.tmp")
        merged_df.write_parquet(tmp)
        tmp.replace(target)
        logger.info("Rewrote %s with %d rows after monthly merge", target, merged_df.height)

        last_through = None
        if "date_of_transfer" in merged_df.columns:
            max_date = merged_df.select(pl.col("date_of_transfer").max()).item()
            if max_date is not None:
                last_through = max_date.date() if hasattr(max_date, "date") else max_date

        manifest.write_manifest(
            self.name,
            raw_dir=self.raw_dir,
            last_data_through=last_through,
            source_url=LAND_REGISTRY_MONTHLY_CSV,
            row_count=merged_df.height,
        )

        return self.clean(self.load(filepath=target))

    def load(
        self,
        filepath: Path | None = None,
        nrows: int | None = None,
        **kwargs,
    ) -> pl.LazyFrame:
        """Build a lazy scan of the Price Paid Data.

        Prefers a parquet file over CSV when available — parquet is smaller and
        loads significantly faster.  Pass an explicit *filepath* to override the
        default file-discovery logic.

        The data is not read into memory until the lazy plan is collected.

        Args:
            filepath: Path to a ``.parquet`` or ``.csv`` file.  When *None* the
                method looks for ``pp-complete.parquet`` in ``raw_dir`` first,
                then falls back to ``pp-complete.csv``.  When a ``.csv`` path is
                supplied and a same-stem ``.parquet`` exists alongside it, the
                parquet is used transparently.
            nrows: Limit number of rows scanned (useful for testing).
        """
        if filepath is None:
            parquet = self.raw_dir / "pp-complete.parquet"
            csv = self.raw_dir / "pp-complete.csv"
            if parquet.exists():
                filepath = parquet
            elif csv.exists():
                filepath = csv
            else:
                raise FileNotFoundError(
                    f"Price Paid Data not found in {self.raw_dir}. Run download() first."
                )
        elif filepath.suffix == ".csv":
            # Transparently prefer a parquet alongside the given CSV.
            parquet = filepath.with_suffix(".parquet")
            if parquet.exists():
                filepath = parquet

        if not filepath.exists():
            raise FileNotFoundError(
                f"Price Paid Data file not found: {filepath}. Run download() first."
            )

        logger.info("Building lazy scan of Price Paid Data from %s", filepath)

        if filepath.suffix == ".parquet":
            lf = pl.scan_parquet(filepath)
            if nrows is not None:
                lf = lf.head(nrows)
            return lf

        # The Land Registry CSV has no header row; all values are braces-wrapped strings.
        # Use an explicit string schema so we can strip braces lazily before casting.
        lf = pl.scan_csv(
            filepath,
            has_header=False,
            schema=_LR_SCHEMA,
            n_rows=nrows,
        )

        # Strip braces and surrounding whitespace; replace empty strings with null.
        lf = lf.with_columns(
            [pl.col(c).str.strip_chars("{}").str.strip_chars() for c in LAND_REGISTRY_COLUMNS]
        )
        lf = lf.with_columns(
            [
                pl.when(pl.col(c) == "")
                .then(pl.lit(None, dtype=pl.String))
                .otherwise(pl.col(c))
                .alias(c)
                for c in LAND_REGISTRY_COLUMNS
            ]
        )

        # Cast price to integer and date_of_transfer to datetime.
        lf = lf.with_columns(
            [
                pl.col("price").cast(pl.Int64, strict=False),
                pl.col("date_of_transfer").str.to_datetime(format="%Y-%m-%d %H:%M", strict=False),
            ]
        )

        return lf

    def clean(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Clean and standardise Price Paid Data (lazy)."""
        logger.info("Building lazy cleaning plan for Price Paid Data")

        return (
            lf
            # Drop deletions — we only want current/added records
            .filter(pl.col("record_status") != "D")
            # Remove transactions with missing postcodes (can't join to EPC)
            .drop_nulls(subset=["postcode"])
            # Standardise postcode format (uppercase, single space)
            .with_columns(
                pl.col("postcode").str.to_uppercase().str.strip_chars().str.replace_all(r"\s+", " ")
            )
            # Filter to residential property types only
            .filter(pl.col("property_type").is_in(["D", "S", "T", "F"]))
            # Filter out extreme prices (likely errors or commercial)
            .filter((pl.col("price") > 10_000) & (pl.col("price") < 50_000_000))
            # Extract year and month for time-based analysis
            .with_columns(
                [
                    pl.col("date_of_transfer").dt.year().alias("year"),
                    pl.col("date_of_transfer").dt.month().alias("month"),
                ]
            )
            # Extract outward postcode (area-level grouping)
            .with_columns(pl.col("postcode").str.split(" ").list.first().alias("postcode_outward"))
        )
