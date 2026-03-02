"""Pipeline to link and merge UK property data from multiple sources.

Joins Land Registry Price Paid transactions with EPC property features
on postcode + address matching, and enriches with UK HPI area-level data.
"""

import logging
from pathlib import Path

import polars as pl

from src.data.config import PROCESSED_DIR

logger = logging.getLogger(__name__)


def normalise_address(paon: pl.Series, street: pl.Series) -> pl.Series:
    """Create a normalised address key from PAON + street for matching."""
    addr = (
        paon.fill_null("").cast(pl.String).str.to_uppercase().str.strip_chars()
        + " "
        + street.fill_null("").cast(pl.String).str.to_uppercase().str.strip_chars()
    )
    return addr.str.replace_all(r"\s+", " ").str.strip_chars()


def link_sales_to_epc(
    sales: pl.DataFrame,
    epc: pl.DataFrame,
) -> pl.DataFrame:
    """Link Land Registry sales to EPC records on postcode.

    Strategy:
    1. Exact join on postcode + address (the most reliable keys available in both).
    2. For rows that didn't match on address, fall back to postcode-only.
    3. For each sale, pick the EPC certificate closest in time to the sale date.

    This produces a left join — sales without EPC matches are retained
    with null EPC columns.
    """
    logger.info(
        "Linking %d sales to %d EPC records on postcode",
        len(sales),
        len(epc),
    )

    # Create address key for better matching
    if "paon" in sales.columns and "street" in sales.columns:
        sales = sales.with_columns(
            normalise_address(sales["paon"], sales["street"]).alias("_addr_key")
        )

    if "address1" in epc.columns:
        epc = epc.with_columns(
            pl.col("address1")
            .fill_null("")
            .cast(pl.String)
            .str.to_uppercase()
            .str.strip_chars()
            .str.replace_all(r"\s+", " ")
            .alias("_addr_key")
        )

    # Try postcode + address match first
    if "_addr_key" in sales.columns and "_addr_key" in epc.columns:
        merged = sales.join(epc, on=["postcode", "_addr_key"], how="left", suffix="_epc")

        check_col = "current_energy_rating" if "current_energy_rating" in epc.columns else None
        if check_col and check_col in merged.columns and "transaction_id" in merged.columns:
            unmatched_ids = merged.filter(pl.col(check_col).is_null())["transaction_id"]

            if len(unmatched_ids) > 0:
                unmatched_sales = sales.filter(
                    pl.col("transaction_id").is_in(unmatched_ids.to_list())
                )
                # Postcode-only fallback for unmatched rows
                epc_for_fallback = epc.drop("_addr_key") if "_addr_key" in epc.columns else epc
                fallback = unmatched_sales.drop("_addr_key").join(
                    epc_for_fallback,
                    on="postcode",
                    how="left",
                    suffix="_epc",
                )
                matched = merged.filter(pl.col(check_col).is_not_null())
                if "_addr_key" in matched.columns:
                    matched = matched.drop("_addr_key")
                merged = pl.concat([matched, fallback], how="diagonal_relaxed")
    else:
        # Postcode-only join
        merged = sales.join(epc, on="postcode", how="left", suffix="_epc")

    # Clean up temporary columns
    for col in ["_addr_key", "_addr_key_epc"]:
        if col in merged.columns:
            merged = merged.drop([col])

    # If multiple EPC matches per sale, keep the one closest in time
    if "date_of_transfer" in merged.columns and "inspection_date" in merged.columns:
        merged = merged.with_columns(
            (pl.col("date_of_transfer").cast(pl.Date) - pl.col("inspection_date").cast(pl.Date))
            .dt.total_days()
            .abs()
            .alias("_date_diff")
        )
        dedup_subset = (
            ["transaction_id"] if "transaction_id" in merged.columns else merged.columns[:5]
        )
        merged = (
            merged.sort("_date_diff", nulls_last=True)
            .unique(subset=dedup_subset, keep="first", maintain_order=True)
            .drop("_date_diff")
        )

    # Series.mean() has a broad return type in polars stubs; narrow to float explicitly.
    _mr = (
        merged["current_energy_rating"].is_not_null().mean()
        if "current_energy_rating" in merged.columns
        else None
    )
    match_rate: float = _mr if isinstance(_mr, float) else 0.0
    logger.info(
        "Linked dataset: %d rows, EPC match rate: %.1f%%",
        len(merged),
        match_rate * 100,
    )
    return merged


def enrich_with_hpi(
    df: pl.DataFrame,
    hpi: pl.DataFrame,
) -> pl.DataFrame:
    """Enrich sales+EPC data with area-level UK HPI statistics.

    Joins on district/region and year-month to add average area prices
    and price index values.
    """
    logger.info("Enriching with UK HPI data (%d HPI rows)", len(hpi))

    # Find region column in HPI
    hpi_region_col = None
    for candidate in ["regionname", "region_name"]:
        if candidate in hpi.columns:
            hpi_region_col = candidate
            break

    if hpi_region_col is None or "district" not in df.columns:
        logger.warning(
            "Cannot enrich with HPI: missing region columns. Returning data without HPI enrichment."
        )
        return df

    # Select useful HPI columns
    hpi_cols = [hpi_region_col, "date"] if "date" in hpi.columns else [hpi_region_col]
    for col in hpi.columns:
        if any(kw in col for kw in ["average_price", "index", "percentage_change", "sales_volume"]):
            hpi_cols.append(col)
    hpi_cols = list(dict.fromkeys(hpi_cols))  # deduplicate, preserve order
    hpi_subset = hpi.select(hpi_cols)

    # Create year-month key from date column
    if "date" in hpi_subset.columns:
        hpi_subset = hpi_subset.with_columns(
            [
                pl.col("date").cast(pl.Date).dt.year().alias("_hpi_year"),
                pl.col("date").cast(pl.Date).dt.month().alias("_hpi_month"),
            ]
        ).drop(["date"])

    if "year" in df.columns and "month" in df.columns:
        merged = df.join(
            hpi_subset,
            left_on=["district", "year", "month"],
            right_on=[hpi_region_col, "_hpi_year", "_hpi_month"],
            how="left",
            suffix="_hpi",
        )

        hpi_price_cols = [c for c in merged.columns if "average_price" in c]
        _hm = (
            merged.select(pl.any_horizontal([pl.col(c).is_not_null() for c in hpi_price_cols]))
            .to_series()
            .mean()
            if hpi_price_cols
            else None
        )
        hpi_match_rate: float = _hm if isinstance(_hm, float) else 0.0
        logger.info(
            "HPI enrichment: %d rows, match rate: %.1f%%",
            len(merged),
            hpi_match_rate * 100,
        )
        return merged

    return df


def save_dataset(
    df: pl.DataFrame,
    name: str,
    output_dir: Path | None = None,
    fmt: str = "parquet",
) -> Path:
    """Save a DataFrame to the processed data directory.

    Args:
        df: DataFrame to save.
        name: Base filename (without extension).
        output_dir: Override output directory.
        fmt: Output format — "parquet" (default) or "csv".
    """
    out_dir = output_dir or PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        dest = out_dir / f"{name}.parquet"
        df.write_parquet(dest)
    else:
        dest = out_dir / f"{name}.csv"
        df.write_csv(dest)

    logger.info("Saved %d rows to %s", len(df), dest)
    return dest


def run_pipeline(
    sales: pl.DataFrame,
    epc: pl.DataFrame | None = None,
    hpi: pl.DataFrame | None = None,
    output_name: str = "uk_property_sales",
    output_dir: Path | None = None,
    fmt: str = "parquet",
) -> pl.DataFrame:
    """Run the full data linking pipeline.

    Args:
        sales: Land Registry Price Paid Data (required).
        epc: EPC data (optional — will enrich if provided).
        hpi: UK HPI data (optional — will enrich if provided).
        output_name: Name for the output file.
        output_dir: Directory to save output.
        fmt: Output format.
    """
    logger.info("Running pipeline with %d sales records", len(sales))

    df = sales

    if epc is not None and len(epc) > 0:
        df = link_sales_to_epc(df, epc)

    if hpi is not None and len(hpi) > 0:
        df = enrich_with_hpi(df, hpi)

    save_dataset(df, output_name, output_dir=output_dir, fmt=fmt)
    return df
