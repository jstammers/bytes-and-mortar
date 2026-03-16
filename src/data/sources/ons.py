"""ONS (Office for National Statistics) time series ingestion.

Downloads monthly/quarterly macro-economic indicators from the ONS
Time Series Explorer API:
- KAB9  / EARN  — Average Weekly Earnings (AWE), total pay (£/week, SA)
- MGSX  / LMS   — Unemployment rate (%, SA)
- IHYQ  / QNA   — GDP quarter-on-quarter growth (%)

Data source: https://api.ons.gov.uk/v1/
License: Open Government Licence v3.0 (no registration required)
"""

import json
import logging
from pathlib import Path

import polars as pl
import requests

from src.data.config import ONS_API_BASE, ONS_SERIES, RAW_DIR
from src.data.sources.base import DataSource

logger = logging.getLogger(__name__)

_TIMEOUT = 30  # seconds per API call


class ONSTimeSeries(DataSource):
    """Ingest macro-economic time series from the ONS API.

    Downloads each ``(series_id, dataset_id)`` pair in ``ONS_SERIES``
    via the ONS Time Series Explorer API, then concatenates them into a
    single long-format monthly LazyFrame with columns:

    - ``date``    — first day of the period (``pl.Date``)
    - ``series``  — ONS series code (e.g. ``"KAB9"``)
    - ``value``   — numeric observation (``pl.Float64``, null if missing)

    Quarterly series are assigned to the first month of the quarter.
    """

    name = "ons"

    def __init__(
        self,
        raw_dir: Path | None = None,
        series: list[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(raw_dir=raw_dir or RAW_DIR)
        self.series = series or list(ONS_SERIES)

    def download(self, **kwargs) -> Path:
        """Fetch all configured ONS series and write them to a single JSON file.

        Each series is fetched from::

            GET {ONS_API_BASE}/datasets/{dataset}/timeseries/{series}/data

        Responses are stored as a list of ``{"series": code, "data": [...]}``
        objects in ``raw_dir/ons_series.json``.

        Returns:
            Path to the saved JSON file.
        """
        dest = self.raw_dir / "ons_series.json"
        if dest.exists():
            logger.info("ONS series already cached at %s; skipping download.", dest)
            return dest

        results = []
        for series_id, dataset_id in self.series:
            url = f"{ONS_API_BASE}/datasets/{dataset_id}/timeseries/{series_id}/data"
            logger.info("Fetching ONS series %s from %s", series_id, url)
            try:
                resp = requests.get(url, timeout=_TIMEOUT)
                resp.raise_for_status()
                payload = resp.json()
                results.append({"series": series_id, "data": payload})
            except requests.RequestException as exc:
                logger.warning("Failed to fetch ONS series %s: %s", series_id, exc)

        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(results, fh)

        logger.info("Saved ONS series to %s (%d series)", dest, len(results))
        return dest

    def load(self, filepath: Path | None = None, **kwargs) -> pl.LazyFrame:
        """Load the cached ONS JSON file into a LazyFrame.

        The JSON structure is a list of::

            {"series": "<code>", "data": <ONS API response dict>}

        The ``data`` dict contains ``"months"``, ``"quarters"``, and/or
        ``"years"`` keys, each holding a list of ``{"date": "YYYY MMM",
        "value": "<number>"}`` dicts.

        Args:
            filepath: Override the default ``raw_dir/ons_series.json`` path.

        Returns:
            LazyFrame with columns: ``series`` (str), ``period`` (str),
            ``raw_date`` (str), ``value`` (str).
        """
        path = filepath or (self.raw_dir / "ons_series.json")
        if not path.exists():
            raise FileNotFoundError(f"ONS series data not found at {path}. Run download() first.")
        logger.info("Loading ONS series from %s", path)

        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)

        rows: list[dict] = []
        for entry in raw:
            series_id = entry.get("series", "")
            data = entry.get("data", {})
            for period_key in ("months", "quarters", "years"):
                for obs in data.get(period_key, []):
                    rows.append(
                        {
                            "series": series_id,
                            "period": period_key,
                            "raw_date": obs.get("date", ""),
                            "value": obs.get("value", ""),
                        }
                    )

        if not rows:
            return pl.LazyFrame(
                schema={
                    "series": pl.String,
                    "period": pl.String,
                    "raw_date": pl.String,
                    "value": pl.String,
                }
            )

        return pl.DataFrame(rows).lazy()

    def clean(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Parse ONS date strings and cast values to a tidy monthly LazyFrame.

        Date formats handled:

        - Monthly: ``"YYYY MMM"`` → ``"2020 Jan"`` → ``2020-01-01``
        - Quarterly: ``"YYYY QN"`` → ``"2020 Q1"`` → ``2020-01-01``
        - Annual: ``"YYYY"`` → ``2020`` → ``2020-01-01``

        Quarterly dates are mapped to the first month of the quarter.

        Returns:
            LazyFrame with columns: ``date`` (pl.Date), ``series`` (pl.String),
            ``value`` (pl.Float64).
        """
        logger.info("Building cleaning plan for ONS series data")

        # Map quarter abbreviations to month numbers
        quarter_map = {"Q1": "01", "Q2": "04", "Q3": "07", "Q4": "10"}

        # Collect to apply Python-side date parsing (ONS date strings are heterogeneous)
        df = lf.collect()

        if df.is_empty():
            return pl.LazyFrame(schema={"date": pl.Date, "series": pl.String, "value": pl.Float64})

        parsed_dates: list[str | None] = []
        for row in df.iter_rows(named=True):
            raw = (row.get("raw_date") or "").strip()
            period = row.get("period", "")
            parsed: str | None = None
            try:
                if period == "months" and len(raw) == 8:
                    # "2020 Jan" → "2020-01-01"
                    from datetime import datetime

                    dt = datetime.strptime(raw, "%Y %b")
                    parsed = f"{dt.year}-{dt.month:02d}-01"
                elif period == "quarters" and " Q" in raw:
                    year, q = raw.split(" ")
                    month = quarter_map.get(q, "01")
                    parsed = f"{year}-{month}-01"
                elif period == "years" and len(raw) == 4:
                    parsed = f"{raw}-01-01"
            except (ValueError, AttributeError):
                pass
            parsed_dates.append(parsed)

        df = df.with_columns(pl.Series("date_str", parsed_dates, dtype=pl.String))
        df = df.with_columns(
            pl.col("date_str").str.to_date(format="%Y-%m-%d", strict=False).alias("date")
        )
        df = df.filter(pl.col("date").is_not_null())
        df = df.with_columns(pl.col("value").cast(pl.Float64, strict=False))
        df = df.select(["date", "series", "value"]).sort(["series", "date"])

        return df.lazy()
