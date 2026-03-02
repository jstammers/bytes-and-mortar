"""Pipeline to link and merge UK property data from multiple sources.

Joins Land Registry Price Paid transactions with EPC property features
on postcode + address matching, and enriches with UK HPI area-level data.
"""

import logging
from pathlib import Path

import pandas as pd

from src.data.config import PROCESSED_DIR

logger = logging.getLogger(__name__)


def normalise_address(paon: pd.Series, street: pd.Series) -> pd.Series:
    """Create a normalised address key from PAON + street for matching."""
    addr = (
        paon.fillna("").astype(str).str.upper().str.strip()
        + " "
        + street.fillna("").astype(str).str.upper().str.strip()
    )
    return addr.str.replace(r"\s+", " ", regex=True).str.strip()


def link_sales_to_epc(
    sales: pd.DataFrame,
    epc: pd.DataFrame,
) -> pd.DataFrame:
    """Link Land Registry sales to EPC records on postcode.

    Strategy:
    1. Exact join on postcode (the most reliable key available in both).
    2. For each sale, pick the most recent EPC certificate that predates
       or is closest to the sale date.

    This produces a left join — sales without EPC matches are retained
    with NaN EPC columns.
    """
    logger.info(
        "Linking %d sales to %d EPC records on postcode",
        len(sales),
        len(epc),
    )

    # Ensure matching postcode format
    sales = sales.copy()
    epc = epc.copy()

    # Create address key for better matching
    if "paon" in sales.columns and "street" in sales.columns:
        sales["_addr_key"] = normalise_address(sales["paon"], sales["street"])

    if "address1" in epc.columns:
        epc["_addr_key"] = (
            epc["address1"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    # Try postcode + address match first
    if "_addr_key" in sales.columns and "_addr_key" in epc.columns:
        merged = sales.merge(
            epc,
            on=["postcode", "_addr_key"],
            how="left",
            suffixes=("", "_epc"),
        )
        # For rows that didn't match on address, fall back to postcode-only
        unmatched_mask = (
            merged["current_energy_rating"].isna()
            if "current_energy_rating" in merged.columns
            else pd.Series(True, index=merged.index)
        )
        matched = merged[~unmatched_mask]
        unmatched_sales = sales.loc[
            sales.index.isin(merged.loc[unmatched_mask, "transaction_id"].values)
            if "transaction_id" in sales.columns
            else []
        ]

        if len(unmatched_sales) > 0:
            fallback = unmatched_sales.merge(
                epc,
                on="postcode",
                how="left",
                suffixes=("", "_epc"),
            )
            merged = pd.concat([matched, fallback], ignore_index=True)
    else:
        # Postcode-only join
        merged = sales.merge(
            epc,
            on="postcode",
            how="left",
            suffixes=("", "_epc"),
        )

    # Clean up temporary columns
    for col in ["_addr_key", "_addr_key_epc"]:
        if col in merged.columns:
            merged = merged.drop(columns=[col])

    # If multiple EPC matches per sale, keep the one closest in time
    if "date_of_transfer" in merged.columns and "inspection_date" in merged.columns:
        merged["_date_diff"] = abs((merged["date_of_transfer"] - merged["inspection_date"]).dt.days)
        merged = (
            merged.sort_values("_date_diff")
            .drop_duplicates(
                subset=["transaction_id"]
                if "transaction_id" in merged.columns
                else merged.columns[:5].tolist(),
                keep="first",
            )
            .drop(columns=["_date_diff"])
        )

    match_rate = (
        merged["current_energy_rating"].notna().mean()
        if "current_energy_rating" in merged.columns
        else 0
    )
    logger.info(
        "Linked dataset: %d rows, EPC match rate: %.1f%%",
        len(merged),
        match_rate * 100,
    )
    return merged


def enrich_with_hpi(
    df: pd.DataFrame,
    hpi: pd.DataFrame,
) -> pd.DataFrame:
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
    hpi_cols = [hpi_region_col, "date"]
    for col in hpi.columns:
        if any(kw in col for kw in ["average_price", "index", "percentage_change", "sales_volume"]):
            hpi_cols.append(col)
    hpi_cols = list(set(hpi_cols))
    hpi_subset = hpi[hpi_cols].copy()

    # Create year-month key in both datasets
    if "date" in hpi_subset.columns:
        hpi_subset["_hpi_year"] = pd.to_datetime(hpi_subset["date"], errors="coerce").dt.year
        hpi_subset["_hpi_month"] = pd.to_datetime(hpi_subset["date"], errors="coerce").dt.month
        hpi_subset = hpi_subset.drop(columns=["date"])

    if "year" in df.columns and "month" in df.columns:
        df = df.copy()
        # Attempt to match district name to HPI region
        merged = df.merge(
            hpi_subset,
            left_on=["district", "year", "month"],
            right_on=[hpi_region_col, "_hpi_year", "_hpi_month"],
            how="left",
            suffixes=("", "_hpi"),
        )
        for col in [hpi_region_col, "_hpi_year", "_hpi_month"]:
            if col in merged.columns and col not in df.columns:
                merged = merged.drop(columns=[col])

        hpi_match_rate = (
            merged[[c for c in merged.columns if "average_price" in c]].notna().any(axis=1).mean()
        )
        logger.info(
            "HPI enrichment: %d rows, match rate: %.1f%%",
            len(merged),
            hpi_match_rate * 100,
        )
        return merged

    return df


def save_dataset(
    df: pd.DataFrame,
    name: str,
    output_dir: Path | None = None,
    fmt: str = "parquet",
) -> Path:
    """Save a DataFrame to the processed data directory.

    Args:
        df: DataFrame to save.
        name: Base filename (without extension).
        output_dir: Override output directory.
        fmt: Output format — "parquet" or "csv".
    """
    out_dir = output_dir or PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        dest = out_dir / f"{name}.parquet"
        df.to_parquet(dest, index=False)
    else:
        dest = out_dir / f"{name}.csv"
        df.to_csv(dest, index=False)

    logger.info("Saved %d rows to %s", len(df), dest)
    return dest


def run_pipeline(
    sales: pd.DataFrame,
    epc: pd.DataFrame | None = None,
    hpi: pd.DataFrame | None = None,
    output_name: str = "uk_property_sales",
    output_dir: Path | None = None,
    fmt: str = "parquet",
) -> pd.DataFrame:
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
