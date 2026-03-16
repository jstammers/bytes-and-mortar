"""Bank of England macro-economic series ingestion.

Downloads monthly/daily series from the Bank of England Interactive Database:
- IUDBEDR  — Official Bank Rate (base rate), daily
- IUMBV42  — Effective mortgage interest rate, monthly (%)
- LPMVTVB  — Mortgage approvals for house purchase, monthly (thousands)

Data source: https://www.bankofengland.co.uk/boeapps/database/
License: Open Government Licence v3.0 (no registration required)
"""

import logging
from pathlib import Path

import polars as pl

from src.data.config import BOE_CSV_URL, BOE_SERIES_CODES, RAW_DIR
from src.data.sources.base import DataSource

logger = logging.getLogger(__name__)


class BankOfEnglandSeries(DataSource):
    """Ingest macro-economic time series from the Bank of England.

    Downloads a combined CSV containing all configured series codes
    (``BOE_SERIES_CODES``), then cleans and pivots to a long-format
    monthly LazyFrame with columns:

    - ``date``   — first day of the period (``pl.Date``)
    - ``series`` — BoE series code (e.g. ``"IUDBEDR"``)
    - ``value``  — numeric observation (``pl.Float64``, null if missing)

    Daily series (e.g. base rate) are resampled to month-end averages.
    """

    name = "bank_of_england"

    def __init__(
        self,
        raw_dir: Path | None = None,
        series_codes: list[str] | None = None,
    ) -> None:
        super().__init__(raw_dir=raw_dir or RAW_DIR)
        self.series_codes = series_codes or BOE_SERIES_CODES

    def download(self, **kwargs) -> Path:
        """Download a combined CSV for all configured series from the BoE.

        The BoE bulk export endpoint returns a multi-column CSV with one
        column per series code.  The file is cached locally; re-running
        will skip the download if the file already exists.

        Returns:
            Path to the downloaded CSV file.
        """
        codes = ",".join(self.series_codes)
        url = BOE_CSV_URL.format(codes=codes)
        dest = self.raw_dir / "boe_series.csv"
        return self._download_file(url, dest, desc="Bank of England series")

    def load(self, filepath: Path | None = None, **kwargs) -> pl.LazyFrame:
        """Build a lazy scan of the downloaded BoE CSV.

        Args:
            filepath: Override the default path (``raw_dir/boe_series.csv``).

        Returns:
            LazyFrame with raw BoE multi-column CSV structure.
        """
        path = filepath or (self.raw_dir / "boe_series.csv")
        if not path.exists():
            raise FileNotFoundError(f"BoE series data not found at {path}. Run download() first.")
        logger.info("Building lazy scan of BoE series from %s", path)
        # BoE CSV has a header row with series codes and a date column.
        # infer_schema_length=0 keeps everything as strings for robust parsing.
        return pl.scan_csv(path, infer_schema_length=0, null_values=["", "N/A", "n/a"])

    def clean(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Parse, melt, and resample BoE series to a monthly long-format LazyFrame.

        Parsing steps:
        1. Identify the date column (first column, any name).
        2. Cast date strings (``DD MMM YYYY`` or ``YYYY-MM-DD``) to ``pl.Date``.
        3. Melt value columns into ``(date, series, value)`` long format.
        4. Cast values to ``Float64`` (nulls where conversion fails).
        5. Resample daily observations to monthly averages (group by year-month).

        Returns:
            LazyFrame with columns: ``date`` (pl.Date), ``series`` (pl.String),
            ``value`` (pl.Float64).
        """
        logger.info("Building cleaning plan for BoE series data")
        col_names = lf.collect_schema().names()
        if not col_names:
            return lf

        date_col = col_names[0]
        value_cols = col_names[1:]

        # Attempt to parse the BoE date format "DD Mon YYYY" (e.g. "01 Jan 2020").
        # Fall back to ISO format if that fails.
        lf = lf.with_columns(
            pl.coalesce(
                pl.col(date_col).str.to_date(format="%d %b %Y", strict=False),
                pl.col(date_col).str.to_date(format="%Y-%m-%d", strict=False),
            ).alias("date")
        ).drop(date_col)

        # Remove rows where date could not be parsed (header lines, footnotes, etc.)
        lf = lf.filter(pl.col("date").is_not_null())

        # Pivot to long format: one row per (date, series)
        lf = lf.unpivot(index=["date"], on=value_cols, variable_name="series")

        # Cast value column to Float64
        lf = lf.with_columns(pl.col("value").cast(pl.Float64, strict=False))

        # Resample to monthly average (group by year-month, series)
        lf = (
            lf.with_columns(pl.col("date").dt.truncate("1mo").alias("date"))
            .group_by(["date", "series"])
            .agg(pl.col("value").mean())
            .sort(["series", "date"])
        )

        return lf
