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

import base64
import logging
import os
import shutil
import time
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import polars as pl
import requests

from src.data.config import EPC_DOMESTIC_SCHEMA, EPC_DOMESTIC_SEARCH
from src.data.sources.base import DataSource

logger = logging.getLogger(__name__)

# EPC API endpoints
EPC_FILES_API = "https://epc.opendatacommunities.org/api/v1/files"
EPC_FILES_DOWNLOAD = "https://epc.opendatacommunities.org/api/v1/files/{file_name}"

# EPC API rate limits: 30 requests per minute
EPC_RATE_LIMIT_DELAY = 2.1  # seconds between requests
EPC_PAGE_SIZE = 5000  # max rows per request


class EPCData(DataSource):
    """Ingest EPC domestic certificate data via the API."""

    name = "epc_domestic"

    def __init__(
        self,
        raw_dir: Path | None = None,
        api_user: str | None = None,
        api_pass: str | None = None,
    ):
        super().__init__(raw_dir)
        self.api_pass = api_pass or os.environ.get("EPC_API_PASS", "")
        self.api_user = api_user or os.environ.get("EPC_API_USER", "")
        if not self.api_user or not self.api_pass:
            logger.warning(
                "EPC_API_USER and EPC_API_PASS not found. Set them in .env or pass api_user= and api_pass= to use the EPC API. Register at: https://epc.opendatacommunities.org/login"
            )
        # convert into token using base64 encoding of "user:pass"
        self.api_token = (
            self._encode_token(self.api_user, self.api_pass)
            if self.api_user and self.api_pass
            else None
        )

    def _encode_token(self, user: str, password: str) -> str:
        token_str = f"{user}:{password}"
        return base64.b64encode(token_str.encode()).decode()

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Basic {self.api_token}",
        }

    def _list_available_files(self) -> dict:
        """Fetch list of available bulk download files from the API."""
        if not self.api_token:
            raise ValueError(
                "EPC API token not set. Set EPC_API_USER and EPC_API_PASS in .env or pass them to the constructor."
            )

        headers = self._get_headers()
        headers["Accept"] = "application/json"

        logger.info("Fetching list of available EPC bulk download files")
        response = requests.get(EPC_FILES_API, headers=headers, timeout=60)
        response.raise_for_status()

        data = response.json()
        return data.get("files", {})

    def download(
        self,
        use_search: bool = False,
        bulk_file: str | None = None,
        postcodes: list[str] | None = None,
        local_authority: str | None = None,
        from_month: int | None = None,
        from_year: int | None = None,
        max_pages: int = 100000,
        **kwargs,
    ) -> Path:
        """Download EPC data.

        Default behavior downloads the entire dataset via bulk file.
        For filtered/specific data, use the search API.

        Args:
            use_search: If True, use the search API instead of bulk download.
                Defaults to False (use bulk download).
            bulk_file: Specific bulk file to download (e.g., "all-domestic-certificates.zip").
                Defaults to None (auto-select the most complete file).
            postcodes: List of postcodes to query (e.g. ["SW1A 1AA"]). Only used with use_search=True.
            local_authority: Local authority code (e.g. "E09000033"). Only used with use_search=True.
            from_month: Filter certificates from this month (1-12). Only used with use_search=True.
            from_year: Filter certificates from this year. Only used with use_search=True.
            max_pages: Maximum number of pages to fetch (each page = 5000 rows). Only used with use_search=True.

        Returns:
            Path to the downloaded/extracted EPC data file (CSV).
        """
        if not self.api_token:
            raise ValueError(
                "EPC API token not set. Set EPC_API_USER and EPC_API_PASS in .env or pass them to the constructor."
            )

        if use_search:
            # Delegate to search-based download
            if postcodes:
                return self._search_by_postcodes(postcodes, max_pages)
            return self._search_with_filters(
                local_authority=local_authority,
                from_month=from_month,
                from_year=from_year,
                max_pages=max_pages,
            )

        # Bulk download (default)
        return self._download_bulk(bulk_file=bulk_file)

    def _download_bulk(self, bulk_file: str | None = None) -> Path:
        """Download entire EPC dataset from bulk file.

        Args:
            bulk_file: Specific file to download. If None, downloads all-domestic-certificates.zip.

        Returns:
            Path to the extracted CSV file.
        """
        if bulk_file is None:
            bulk_file = "all-domestic-certificates.zip"

        # Check if file is already extracted
        csv_path = self.raw_dir / "epc_domestic_bulk.csv"
        if csv_path.exists():
            logger.info("EPC bulk data already exists: %s", csv_path)
            return csv_path

        # Download the bulk file
        logger.info("Downloading EPC bulk file: %s", bulk_file)
        download_url = EPC_FILES_DOWNLOAD.format(file_name=bulk_file)

        headers = self._get_headers()
        response = requests.get(download_url, headers=headers, timeout=300, stream=True)
        response.raise_for_status()

        # Save zip file temporarily
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.raw_dir / bulk_file
        total_size = int(response.headers.get("content-length", 0))

        if not zip_path.exists():
            with open(zip_path, "wb") as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            progress = (downloaded / total_size) * 100
                            logger.debug("Download progress: %.1f%%", progress)

            logger.info("Downloaded %s (%d bytes)", bulk_file, zip_path.stat().st_size)
        else:
            logger.info("Bulk file already downloaded: %s", zip_path)

        # Extract CSV files from zip
        logger.info("Extracting CSV files from %s", zip_path)
        temp_extract = self.raw_dir / "temp_extract"
        temp_extract.mkdir(exist_ok=True)

        with ZipFile(zip_path) as zf:
            # Find and extract all CSV files
            csv_files = [
                n for n in zf.namelist() if n.endswith(".csv") and "certificates" in n.lower()
            ]
            if not csv_files:
                raise RuntimeError(f"No CSV files found in {bulk_file}")

            for csv_file in csv_files:
                logger.info("Extracting %s", csv_file)
                zf.extract(csv_file, temp_extract)

        extracted_csvs = list(temp_extract.rglob("*certificates*.csv"))
        if not extracted_csvs:
            raise RuntimeError(f"No CSV files found after extracting {bulk_file}")

        if len(extracted_csvs) == 1:
            shutil.move(str(extracted_csvs[0]), str(csv_path))
        else:
            logger.info("Merging %d CSV files into %s", len(extracted_csvs), csv_path)

            dfs = [
                pl.read_csv(
                    csv_file,
                    schema=EPC_DOMESTIC_SCHEMA,
                    null_values=["", "N/A", "NO DATA!", "INVALID!", "null", "NULL"],
                )
                for csv_file in extracted_csvs
            ]
            merged_df = pl.concat(dfs, how="diagonal_relaxed")
            merged_df.write_csv(csv_path)

        # Cleanup
        shutil.rmtree(temp_extract, ignore_errors=True)
        zip_path.unlink()

        logger.info("Saved EPC bulk data to %s", csv_path)
        return csv_path

    def _search_with_filters(
        self,
        local_authority: str | None = None,
        from_month: int | None = None,
        from_year: int | None = None,
        max_pages: int = 100000,
    ) -> Path:
        """Download EPC data using search API with filters."""
        params = {"size": EPC_PAGE_SIZE}
        filename_parts = ["epc_domestic"]

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
            logger.info("EPC search data already exists: %s", dest)
            return dest

        all_rows = []
        for page in range(max_pages):
            params["search-after"] = page * EPC_PAGE_SIZE
            logger.info(
                "Fetching EPC page %d (from row %d)",
                page + 1,
                params["search-after"],
            )

            headers = self._get_headers()
            headers["Accept"] = "text/csv"
            response = requests.get(
                EPC_DOMESTIC_SEARCH,
                headers=headers,
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
        logger.info("Saved EPC search data to %s", dest)
        return dest

    def _search_by_postcodes(self, postcodes: list[str], max_pages: int) -> Path:
        """Download EPC data for a list of postcodes using search API."""
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
            headers = self._get_headers()
            headers["Accept"] = "text/csv"
            response = requests.get(
                EPC_DOMESTIC_SEARCH,
                headers=headers,
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

    def load(self, filepath: Path | None = None, **kwargs) -> pl.LazyFrame:
        """Build a lazy scan of the EPC CSV.

        The data is not read into memory until the lazy plan is collected.
        """
        if filepath is None:
            # Find the most recent EPC file
            epc_files = sorted(self.raw_dir.glob("epc_domestic*.csv"))
            if not epc_files:
                raise FileNotFoundError("No EPC data files found. Run download() first.")
            filepath = epc_files[-1]

        logger.info("Building lazy scan of EPC data from %s", filepath)
        return pl.scan_csv(filepath, infer_schema_length=10000)

    def clean(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Clean and standardise EPC data (lazy).

        Note: All EPC certificate records are retained (no deduplication by building
        reference). The pipeline selects the temporally closest certificate per sale,
        so preserving the full history is essential.
        """
        logger.info("Building lazy cleaning plan for EPC data")

        # Resolve the schema once upfront to avoid repeated (expensive) schema resolutions.
        schema = lf.collect_schema()
        col_names = schema.names()

        # Standardise column names to snake_case
        rename_map = {col: col.lower().replace("-", "_").replace(" ", "_") for col in col_names}
        lf = lf.rename(rename_map)
        # Re-resolve after rename so subsequent checks are accurate.
        col_names = [rename_map.get(c, c) for c in col_names]

        # Standardise postcode format
        if "postcode" in col_names:
            lf = lf.with_columns(
                pl.col("postcode")
                .cast(pl.String)
                .str.to_uppercase()
                .str.strip_chars()
                .str.replace_all(r"\s+", " ")
            )

        # Parse inspection_date
        if "inspection_date" in col_names:
            lf = lf.with_columns(
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
        lf = lf.with_columns(
            [pl.col(col).cast(pl.Float64, strict=False) for col in numeric_cols if col in col_names]
        )

        # Drop rows with no postcode (can't link to sales data)
        lf = lf.drop_nulls(subset=["postcode"])

        # Filter to domestic properties with valid energy ratings
        if "current_energy_rating" in col_names:
            lf = lf.filter(
                pl.col("current_energy_rating").is_in(["A", "B", "C", "D", "E", "F", "G"])
            )

        return lf
