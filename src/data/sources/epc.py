"""Energy Performance Certificate (EPC) data ingestion.

Data source: https://epc.opendatacommunities.org/
License: Open Government Licence v3.0

EPC data provides detailed property characteristics including:
- Energy efficiency ratings (A-G)
- Total floor area
- Number of habitable rooms
- Construction age band
- Wall/roof/window types
- Heating system details
- Tenure (owner-occupied, rented, etc.)

Requires a free API key: register at https://epc.opendatacommunities.org/login
Set EPC_API_TOKEN in your .env file.
"""

import logging
import os
import time
from io import BytesIO
from pathlib import Path

import polars as pl
import requests

from src.data.config import EPC_DOMESTIC_SEARCH
from src.data.sources.base import DataSource

logger = logging.getLogger(__name__)

# EPC API rate limits: 30 requests per minute
EPC_RATE_LIMIT_DELAY = 2.1  # seconds between requests
EPC_PAGE_SIZE = 5000  # max rows per request


class EPCData(DataSource):
    """Ingest EPC domestic certificate data via the API."""

    name = "epc_domestic"

    def __init__(self, raw_dir: Path | None = None, api_token: str | None = None):
        super().__init__(raw_dir)
        self.api_token = api_token or os.environ.get("EPC_API_TOKEN", "")
        if not self.api_token:
            logger.warning(
                "No EPC_API_TOKEN found. Set it in .env or pass api_token= "
                "to use the EPC API. Register at: "
                "https://epc.opendatacommunities.org/login"
            )

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Basic {self.api_token}",
            "Accept": "text/csv",
        }

    def download(
        self,
        postcodes: list[str] | None = None,
        local_authority: str | None = None,
        from_month: int | None = None,
        from_year: int | None = None,
        max_pages: int = 10,
        **kwargs,
    ) -> Path:
        """Download EPC data via the API.

        Args:
            postcodes: List of postcodes to query (e.g. ["SW1A 1AA"]).
            local_authority: Local authority code (e.g. "E09000033").
            from_month: Filter certificates from this month (1-12).
            from_year: Filter certificates from this year.
            max_pages: Maximum number of pages to fetch (each page = 5000 rows).
        """
        if not self.api_token:
            raise ValueError(
                "EPC_API_TOKEN is required. Register at "
                "https://epc.opendatacommunities.org/login "
                "and set EPC_API_TOKEN in your .env file."
            )

        params = {"size": EPC_PAGE_SIZE}
        filename_parts = ["epc_domestic"]

        if postcodes:
            # API supports single postcode per request; we batch
            return self._download_by_postcodes(postcodes, max_pages)

        if local_authority:
            params["local-authority"] = local_authority
            filename_parts.append(local_authority)

        if from_year:
            params["from-year"] = from_year
            filename_parts.append(f"from_{from_year}")
        if from_month:
            params["from-month"] = from_month

        dest = self.raw_dir / ("_".join(filename_parts) + ".csv")

        if dest.exists():
            logger.info("EPC data already exists: %s", dest)
            return dest

        all_rows = []
        for page in range(max_pages):
            params["from"] = page * EPC_PAGE_SIZE
            logger.info(
                "Fetching EPC page %d (from row %d)",
                page + 1,
                params["from"],
            )

            response = requests.get(
                EPC_DOMESTIC_SEARCH,
                headers=self._get_headers(),
                params=params,
                timeout=60,
            )
            response.raise_for_status()

            text = response.text.strip()
            if not text or text.count("\n") <= 1:
                logger.info("No more EPC data at page %d", page + 1)
                break

            all_rows.append(text)
            time.sleep(EPC_RATE_LIMIT_DELAY)

        if not all_rows:
            raise RuntimeError("No EPC data returned from API.")

        # Combine pages: first page has header, subsequent don't need it
        header = all_rows[0].split("\n")[0]
        lines = [header]
        for chunk in all_rows:
            chunk_lines = chunk.split("\n")
            lines.extend(chunk_lines[1:])

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(lines))
        logger.info("Saved EPC data to %s", dest)
        return dest

    def _download_by_postcodes(self, postcodes: list[str], max_pages: int) -> Path:
        """Download EPC data for a list of postcodes."""
        dest = self.raw_dir / "epc_domestic_postcodes.csv"
        if dest.exists():
            logger.info("EPC postcode data already exists: %s", dest)
            return dest

        all_dfs = []
        for postcode in postcodes:
            logger.info("Fetching EPC data for postcode %s", postcode)
            params = {
                "postcode": postcode,
                "size": EPC_PAGE_SIZE,
            }
            response = requests.get(
                EPC_DOMESTIC_SEARCH,
                headers=self._get_headers(),
                params=params,
                timeout=60,
            )
            response.raise_for_status()

            text = response.text.strip()
            if text and text.count("\n") > 1:
                chunk_df = pl.read_csv(BytesIO(text.encode()))
                all_dfs.append(chunk_df)

            time.sleep(EPC_RATE_LIMIT_DELAY)

        if not all_dfs:
            raise RuntimeError("No EPC data returned for given postcodes.")

        df = pl.concat(all_dfs, how="diagonal_relaxed")
        df.write_csv(dest)
        logger.info("Saved EPC data for %d postcodes to %s", len(postcodes), dest)
        return dest

    def load(self, filepath: Path | None = None, **kwargs) -> pl.DataFrame:
        """Load EPC CSV into DataFrame."""
        if filepath is None:
            # Find the most recent EPC file
            epc_files = sorted(self.raw_dir.glob("epc_domestic*.csv"))
            if not epc_files:
                raise FileNotFoundError("No EPC data files found. Run download() first.")
            filepath = epc_files[-1]

        logger.info("Loading EPC data from %s", filepath)
        df = pl.read_csv(filepath, infer_schema_length=10000)
        logger.info("Loaded %d EPC records", len(df))
        return df

    def clean(self, df: pl.DataFrame) -> pl.DataFrame:
        """Clean and standardise EPC data."""
        logger.info("Cleaning EPC data (%d rows)", len(df))

        # Standardise column names to snake_case
        df = df.rename({
            col: col.lower().replace("-", "_").replace(" ", "_")
            for col in df.columns
        })

        # Standardise postcode format
        if "postcode" in df.columns:
            df = df.with_columns(
                pl.col("postcode")
                .cast(pl.String)
                .str.to_uppercase()
                .str.strip_chars()
                .str.replace_all(r"\s+", " ")
            )

        # Parse inspection_date
        if "inspection_date" in df.columns:
            df = df.with_columns(
                pl.col("inspection_date").str.to_date(format="%Y-%m-%d", strict=False)
            )

        # Numeric conversions
        numeric_cols = [
            "current_energy_efficiency",
            "potential_energy_efficiency",
            "total_floor_area",
            "number_habitable_rooms",
            "number_heated_rooms",
            "co2_emissions_current",
            "co2_emissions_potential",
            "energy_consumption_current",
            "energy_consumption_potential",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

        # Drop rows with no postcode (can't link to sales data)
        df = df.drop_nulls(subset=["postcode"])

        # Filter to domestic properties with valid energy ratings
        valid_ratings = ["A", "B", "C", "D", "E", "F", "G"]
        if "current_energy_rating" in df.columns:
            df = df.filter(pl.col("current_energy_rating").is_in(valid_ratings))

        # Keep most recent EPC per property (by building reference)
        if "building_reference_number" in df.columns and "inspection_date" in df.columns:
            df = df.sort("inspection_date", descending=True, nulls_last=True).unique(
                subset=["building_reference_number"], keep="first", maintain_order=True
            )

        logger.info("After cleaning: %d rows", len(df))
        return df
