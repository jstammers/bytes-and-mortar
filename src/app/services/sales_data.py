"""SalesDataService: queries the processed parquet for real property data.

Bugs fixed vs. the initial sketch:
- Property ID is a stable MD5 hash of (paon, saon, street, postcode), not the
  transaction_id.  The parquet has one row per *transaction*; using transaction_id
  means the "property" changes identity every sale.
- search_properties now filters *before* collecting so only matching rows are
  loaded — the full parquet is never pulled into memory.
- Transactions are deduplicated to one row per unique property (latest sale).
- get_price_history filters on (postcode, paon, saon, street) address key, not
  on transaction_id.
- Dates are formatted as YYYY-MM-DD strings directly in Polars via dt.strftime.
- HPI timeseries is derived from the averageprice column that the pipeline joins
  per-transaction; unique(year, month) gives one point per calendar month.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, cast

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

from src.app.services.base import PropertyDataService
from src.data.config import DURATION_MAP, HPI_REGIONAL_DATA_PATH, PROPERTY_TYPE_MAP

logger = logging.getLogger(__name__)

# Columns needed to display a property card / detail view.
_INDEX_COLS = [
    "price",
    "date_of_transfer",
    "postcode",
    "property_type",
    "duration",
    "paon",
    "saon",
    "street",
    "district",
    "town_city",
    "current_energy_rating",
    "total_floor_area",
    "number_habitable_rooms",
    "current_energy_efficiency",
    "potential_energy_efficiency",
    "number_heated_rooms",
    "co2_emissions_current",
    "co2_emissions_potential",
    "construction_age_band",
]


def _make_property_id(paon: str, saon: str, street: str, postcode: str) -> str:
    """Stable property identifier derived from address components."""
    key = f"{paon}_{saon}_{street}_{postcode}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _format_address(paon: str, saon: str, street: str) -> str:
    """Human-readable address (LR data is uppercase; convert to title case)."""
    paon_t = paon.title() if paon else ""
    saon_t = saon.title() if saon else ""
    street_t = street.title() if street else ""
    if saon_t:
        return f"{saon_t}, {paon_t} {street_t}".strip(", ")
    return f"{paon_t} {street_t}".strip()


def _row_to_prop(row: dict) -> dict:
    """Normalise a raw parquet row dict into a canonical property dict."""
    paon = row.get("paon") or ""
    saon = row.get("saon") or ""
    street = row.get("street") or ""
    postcode = (row.get("postcode") or "").strip()
    district = (row.get("district") or "").title()
    prop_type_code = row.get("property_type") or ""

    def _int(val: object) -> int | None:
        try:
            return int(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    return {
        "id": _make_property_id(paon, saon, street, postcode),
        "address": _format_address(paon, saon, street),
        # Private fields retained in cache for efficient address-keyed parquet lookups
        "_paon": paon,
        "_saon": saon,
        "_street": street,
        "postcode": postcode,
        "district": district,
        "property_type": PROPERTY_TYPE_MAP.get(prop_type_code, "Other"),
        "property_type_code": prop_type_code,
        "tenure": DURATION_MAP.get(row.get("duration") or "", "Unknown"),
        "last_sale_price": int(row.get("price") or 0),
        "last_sale_date": row.get("last_sale_date") or "",
        "energy_rating": row.get("current_energy_rating"),
        "floor_area": row.get("total_floor_area"),
        "bedrooms": _int(row.get("number_habitable_rooms")),
        "heated_rooms": _int(row.get("number_heated_rooms")),
        "current_energy_efficiency": _int(row.get("current_energy_efficiency")),
        "potential_energy_efficiency": _int(row.get("potential_energy_efficiency")),
        "co2_emissions_current": row.get("co2_emissions_current"),
        "co2_emissions_potential": row.get("co2_emissions_potential"),
        "construction_age_band": row.get("construction_age_band"),
    }


class SalesDataService(PropertyDataService):
    """Queries real property sales data from the processed parquet file.

    Search and detail queries use ``pl.scan_parquet`` so only the rows
    matching the filter are loaded into memory.  The service caches recently
    returned property dicts (keyed by property_id) to serve follow-up
    requests (history, similar, area comparison) without re-scanning.

    Properties must be returned by search_properties (or get_similar_properties)
    before get_property / get_price_history / get_area_comparison will work,
    since the property ID hash cannot be reversed to an address for parquet
    filtering.
    """

    def __init__(self, parquet_path: Path, hpi_path: Path | None = None) -> None:
        if not parquet_path.exists():
            raise FileNotFoundError(f"Sales data not found at {parquet_path}")
        self._path = parquet_path
        schema = pl.scan_parquet(parquet_path).collect_schema()
        self._schema: set[str] = set(schema.names())
        self._cache: dict[str, dict] = {}

        # Optional dedicated HPI regional parquet (preferred over embedded HPI columns).
        if hpi_path is not None and hpi_path.exists():
            self._hpi_path: Path | None = hpi_path
        elif HPI_REGIONAL_DATA_PATH.exists():
            self._hpi_path = HPI_REGIONAL_DATA_PATH
        else:
            self._hpi_path = None

        if self._hpi_path:
            self._hpi_schema: set[str] = set(
                pl.scan_parquet(self._hpi_path).collect_schema().names()
            )
        else:
            self._hpi_schema = set()

        logger.info(
            "SalesDataService ready (parquet: %s, hpi: %s)",
            parquet_path,
            self._hpi_path or "none",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _scan(self) -> pl.LazyFrame:
        return pl.scan_parquet(self._path)

    def _cols(self, want: list[str]) -> list[str]:
        return [c for c in want if c in self._schema]

    def _addr_filter(self, paon: str, saon: str, street: str, postcode: str) -> pl.Expr:
        return (
            (pl.col("postcode") == postcode)
            & (pl.col("paon").fill_null("") == paon)
            & (pl.col("saon").fill_null("") == saon)
            & (pl.col("street").fill_null("") == street)
        )

    def _search_df(self, query: str, limit: int) -> pl.DataFrame:
        """Filter, deduplicate, and format a property search result DataFrame."""
        q = query.strip().upper()

        filter_expr = (
            pl.col("postcode").fill_null("").str.contains(q, literal=True)
            | pl.col("paon").fill_null("").str.contains(q, literal=True)
            | pl.col("street").fill_null("").str.contains(q, literal=True)
            | pl.col("district").fill_null("").str.contains(q, literal=True)
            | pl.col("town_city").fill_null("").str.contains(q, literal=True)
        )

        cols = self._cols(_INDEX_COLS)

        return cast(
            "pl.DataFrame",
            self._scan()
            .select(cols)
            .filter(filter_expr)
            .sort("date_of_transfer", descending=True)
            # One row per unique property: latest transaction wins.
            .unique(subset=["postcode", "paon", "saon", "street"], keep="first")
            .limit(limit)
            # Format date as YYYY-MM-DD string in Polars (avoids Python datetime
            # repr issues when calling to_dicts()).
            .with_columns(
                pl.col("date_of_transfer").dt.strftime("%Y-%m-%d").alias("last_sale_date")
            )
            .collect(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_properties(self, query: str) -> list[dict]:
        """Search for properties by postcode, address, street, or district.

        The query is matched case-insensitively as a substring.  Results
        are deduplicated to one row per unique property (latest transaction).
        """
        if not query.strip():
            return []

        df = self._search_df(query, limit=20)
        results = []
        for row in df.to_dicts():
            prop = _row_to_prop(row)
            self._cache[prop["id"]] = prop
            results.append(prop)

        logger.debug("search_properties(%r) → %d results", query, len(results))
        return results

    def get_property(self, property_id: str) -> dict | None:
        """Return a cached property dict, or None if not previously searched."""
        return self._cache.get(property_id)

    def get_price_history(self, property_id: str) -> list[dict]:
        """Return all sale transactions for a property, sorted by date."""
        prop = self._cache.get(property_id)
        if not prop:
            return []

        txn_cols = self._cols(
            ["transaction_id", "price", "date_of_transfer", "postcode", "paon", "saon", "street"]
        )
        df = cast(
            "pl.DataFrame",
            self._scan()
            .select(txn_cols)
            .filter(
                self._addr_filter(prop["_paon"], prop["_saon"], prop["_street"], prop["postcode"])
            )
            .select(
                pl.col("transaction_id"),
                pl.col("price").cast(pl.Int64),
                pl.col("date_of_transfer").dt.strftime("%Y-%m-%d").alias("date"),
            )
            .sort("date")
            .collect(),
        )

        return df.to_dicts()

    def get_area_comparison(self, property_id: str) -> dict:
        """Return the property's sale history plus monthly HPI for its district."""
        prop = self._cache.get(property_id)
        if not prop:
            return {}

        district_upper = prop["district"].upper()

        txn_cols = self._cols(["price", "date_of_transfer", "postcode", "paon", "saon", "street"])
        prop_txns = cast(
            "pl.DataFrame",
            self._scan()
            .select(txn_cols)
            .filter(
                self._addr_filter(prop["_paon"], prop["_saon"], prop["_street"], prop["postcode"])
            )
            .select(
                pl.col("price").cast(pl.Int64),
                pl.col("date_of_transfer").dt.strftime("%Y-%m-%d").alias("date"),
            )
            .sort("date")
            .collect(),
        )

        # HPI timeseries: one averageprice per year-month for the district.
        # Prefer the dedicated regional parquet; fall back to embedded HPI columns.
        hpi_rows: list[dict] = []
        if self._hpi_path:
            hpi_region_col = next(
                (c for c in ["regionname", "region_name"] if c in self._hpi_schema), None
            )
            if hpi_region_col and "averageprice" in self._hpi_schema:
                hpi_df = cast(
                    "pl.DataFrame",
                    pl.scan_parquet(self._hpi_path)
                    .select([hpi_region_col, "year", "month", "averageprice"])
                    .filter(pl.col(hpi_region_col).str.to_uppercase() == district_upper)
                    .filter(pl.col("averageprice").is_not_null())
                    .unique(subset=["year", "month"])
                    .sort(["year", "month"])
                    .collect(),
                )
                hpi_rows = hpi_df.to_dicts()
        elif "averageprice" in self._schema:
            hpi_cols = self._cols(["district", "year", "month", "averageprice"])
            hpi_df = cast(
                "pl.DataFrame",
                self._scan()
                .select(hpi_cols)
                .filter(pl.col("district") == district_upper)
                .filter(pl.col("averageprice").is_not_null())
                .unique(subset=["year", "month"])
                .sort(["year", "month"])
                .collect(),
            )
            hpi_rows = hpi_df.to_dicts()

        return {
            "property_id": property_id,
            "district": prop["district"],
            "sale_dates": [row["date"] for row in prop_txns.to_dicts()],
            "sale_prices": [int(row["price"]) for row in prop_txns.to_dicts()],
            "hpi_dates": [f"{row['year']}-{int(row['month']):02d}-01" for row in hpi_rows],
            "hpi_average_prices": [float(row["averageprice"]) for row in hpi_rows],
        }

    def get_similar_properties(self, property_id: str, n: int = 8) -> list[dict]:
        """Find similar properties in the same district by floor area and type."""
        prop = self._cache.get(property_id)
        if not prop:
            return []

        floor_area = prop.get("floor_area")
        if floor_area is None:
            return []

        district_upper = prop["district"].upper()
        min_area = floor_area * 0.6
        max_area = floor_area * 1.4
        cols = self._cols(_INDEX_COLS)

        df = cast(
            "pl.DataFrame",
            self._scan()
            .select(cols)
            .filter(pl.col("district") == district_upper)
            .filter(pl.col("total_floor_area").is_not_null())
            .filter(pl.col("total_floor_area").is_between(min_area, max_area))
            .sort("date_of_transfer", descending=True)
            .unique(subset=["postcode", "paon", "saon", "street"], keep="first")
            .limit(50)
            .with_columns(
                pl.col("date_of_transfer").dt.strftime("%Y-%m-%d").alias("last_sale_date")
            )
            .collect(),
        )

        prop_type_code = prop["property_type_code"]
        results = []

        for row in df.to_dicts():
            p = _row_to_prop(row)
            if p["id"] == property_id:
                continue

            area = p.get("floor_area") or 0.0
            area_diff = abs(area - floor_area) / floor_area
            type_bonus = 0.2 if p["property_type_code"] == prop_type_code else 0.0
            p["similarity_score"] = round(max(0.0, 1.0 - area_diff) * 0.8 + type_bonus, 3)

            self._cache[p["id"]] = p
            results.append(p)

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:n]

    def get_hpi_regions(self) -> list[str]:
        """Return distinct region names from the HPI regional parquet (or sales districts)."""
        if self._hpi_path:
            region_col = next(
                (c for c in ["regionname", "region_name"] if c in self._hpi_schema), None
            )
            if region_col:
                regions = (
                    cast(
                        "pl.DataFrame",
                        pl.scan_parquet(self._hpi_path).select(region_col).unique().collect(),
                    )[region_col]
                    .drop_nulls()
                    .to_list()
                )
                return sorted(str(r).title() for r in regions)

        # Fallback: use distinct district values from the sales parquet
        if "district" not in self._schema:
            return []
        regions = (
            cast(
                "pl.DataFrame",
                self._scan().select("district").unique().collect(),
            )["district"]
            .drop_nulls()
            .to_list()
        )
        return sorted(str(r).title() for r in regions)

    def get_hpi_series(self, region: str) -> tuple[list[str], list[float]]:
        """Return monthly HPI (average price) series for *region*.

        Reads from the dedicated ``uk_hpi_regional.parquet`` when available;
        falls back to embedded HPI columns in the sales parquet otherwise.
        """
        if self._hpi_path:
            region_col = next(
                (c for c in ["regionname", "region_name"] if c in self._hpi_schema), None
            )
            price_col = next(
                (c for c in ["averageprice", "averageprice_hpi"] if c in self._hpi_schema), None
            )
            if region_col and price_col:
                df = cast(
                    "pl.DataFrame",
                    pl.scan_parquet(self._hpi_path)
                    .filter(pl.col(region_col).str.to_uppercase() == region.upper())
                    .filter(pl.col(price_col).is_not_null())
                    .select(["year", "month", price_col])
                    .unique(subset=["year", "month"])
                    .sort(["year", "month"])
                    .collect(),
                )
                if not df.is_empty():
                    dates = [f"{int(r['year'])}-{int(r['month']):02d}-01" for r in df.to_dicts()]
                    return dates, [float(v) for v in df[price_col].to_list()]

        # Fallback: embedded HPI columns
        for col in ("averageprice_hpi", "averageprice"):
            if col in self._schema:
                df = cast(
                    "pl.DataFrame",
                    self._scan()
                    .filter(pl.col("district") == region.upper())
                    .filter(pl.col(col).is_not_null())
                    .select(["year", "month", col])
                    .unique(subset=["year", "month"])
                    .sort(["year", "month"])
                    .collect(),
                )
                if not df.is_empty():
                    dates = [f"{int(r['year'])}-{int(r['month']):02d}-01" for r in df.to_dicts()]
                    return dates, [float(v) for v in df[col].to_list()]

        return [], []
