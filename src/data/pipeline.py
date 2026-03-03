"""Pipeline to link and merge UK property data from multiple sources.

Joins Land Registry Price Paid transactions with EPC property features
on postcode + address matching, and enriches with UK HPI area-level data.

All pipeline functions accept and return pl.LazyFrame, building a single
lazy query plan that is executed once at the end with streaming=True. This
allows datasets larger than available RAM to be processed efficiently.
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


def _addr_key_expr(paon_col: str = "paon", street_col: str = "street") -> pl.Expr:
    """Return a polars expression that produces a normalised address key from LR columns."""
    return (
        (
            pl.col(paon_col).fill_null("").cast(pl.String).str.to_uppercase().str.strip_chars()
            + " "
            + pl.col(street_col).fill_null("").cast(pl.String).str.to_uppercase().str.strip_chars()
        )
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def link_sales_to_epc(
    sales: pl.DataFrame | pl.LazyFrame,
    epc: pl.DataFrame | pl.LazyFrame,
) -> pl.LazyFrame:
    """Link Land Registry sales to EPC records, returning a lazy plan.

    Strategy:
    1. Normalise addresses in both datasets.
    2. Left join sales to EPC on postcode — this produces all candidate EPC
       records for each sale (a property may have multiple certificates).
    3. Score each candidate:
         0 = postcode AND normalised address match  (best)
         1 = postcode match only                    (fallback)
         2 = no EPC record found for that postcode  (unmatched)
    4. Compute |sale_date - inspection_date| in days (null when unmatched).
    5. Sort by (transaction_id, match_score, date_diff) and keep the first
       row per transaction — i.e. the best temporal match per sale.

    This produces a left join result: sales without any EPC match are retained
    with null EPC columns (match_score = 2).

    The returned LazyFrame is not executed until .collect() or .sink_*() is called.
    """
    sales_lf = sales.lazy() if isinstance(sales, pl.DataFrame) else sales
    epc_lf = epc.lazy() if isinstance(epc, pl.DataFrame) else epc

    sales_cols = sales_lf.collect_schema().names()
    epc_cols = epc_lf.collect_schema().names()

    if "postcode" not in sales_cols or "postcode" not in epc_cols:
        logger.warning("Cannot link sales to EPC: 'postcode' column missing. Returning sales.")
        return sales_lf

    # --- Normalise addresses ---
    if "paon" in sales_cols and "street" in sales_cols:
        sales_lf = sales_lf.with_columns(_addr_key_expr("paon", "street").alias("_addr_key"))

    if "address1" in epc_cols:
        epc_lf = epc_lf.with_columns(
            pl.col("address1")
            .fill_null("")
            .cast(pl.String)
            .str.to_uppercase()
            .str.strip_chars()
            .str.replace_all(r"\s+", " ")
            .alias("_addr_key")
        )

    # --- Join on postcode to get all EPC candidates per sale ---
    # EPC columns that conflict with sales columns (except postcode) will be
    # renamed with the "_epc" suffix. In particular, if both frames have "_addr_key",
    # the EPC version becomes "_addr_key_epc".
    merged = sales_lf.join(epc_lf, on="postcode", how="left", suffix="_epc")
    merged_cols = merged.collect_schema().names()

    # --- Score each candidate row ---
    has_addr_match = "_addr_key" in merged_cols and "_addr_key_epc" in merged_cols
    has_inspection_date = "inspection_date" in merged_cols

    if has_addr_match and has_inspection_date:
        score_expr = (
            pl.when(pl.col("inspection_date").is_null())
            .then(pl.lit(2))
            .when(
                pl.col("_addr_key").is_not_null()
                & pl.col("_addr_key_epc").is_not_null()
                & (pl.col("_addr_key") == pl.col("_addr_key_epc"))
            )
            .then(pl.lit(0))
            .otherwise(pl.lit(1))
        )
    elif has_inspection_date:
        score_expr = (
            pl.when(pl.col("inspection_date").is_null()).then(pl.lit(1)).otherwise(pl.lit(0))
        )
    else:
        score_expr = pl.lit(0)

    merged = merged.with_columns(score_expr.alias("_match_score"))

    # --- Compute date difference ---
    sort_cols = ["_match_score"]
    if "date_of_transfer" in merged_cols and "inspection_date" in merged_cols:
        merged = merged.with_columns(
            (pl.col("date_of_transfer").cast(pl.Date) - pl.col("inspection_date"))
            .dt.total_days()
            .abs()
            .alias("_date_diff")
        )
        sort_cols.append("_date_diff")

    # --- Pick best EPC match per sale ---
    # Determine the deduplication key (prefer transaction_id).
    dedup_cols = (
        ["transaction_id"]
        if "transaction_id" in merged_cols
        else [c for c in sales_cols[:3] if c in merged_cols]
    )

    # Sort so that for each transaction the best candidate comes first,
    # then unique(keep="first") selects it.  maintain_order=False is required
    # for streaming compatibility.
    result = merged.sort(dedup_cols + sort_cols, nulls_last=True).unique(
        subset=dedup_cols, keep="first", maintain_order=False
    )

    # --- Drop working columns ---
    drop_cols = [
        c
        for c in ["_addr_key", "_addr_key_epc", "_match_score", "_date_diff"]
        if c in result.collect_schema().names()
    ]
    if drop_cols:
        result = result.drop(drop_cols)

    logger.info(
        "EPC join plan built (join on postcode → score by address + date → deduplicate per sale)"
    )
    return result


def enrich_with_hpi(
    df: pl.DataFrame | pl.LazyFrame,
    hpi: pl.DataFrame | pl.LazyFrame,
) -> pl.LazyFrame:
    """Enrich sales+EPC data with area-level UK HPI statistics (lazy).

    Joins on district/region and year-month to add average area prices
    and price index values.
    """
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    hpi_lf = hpi.lazy() if isinstance(hpi, pl.DataFrame) else hpi

    hpi_cols = hpi_lf.collect_schema().names()
    lf_cols = lf.collect_schema().names()

    # Find region column in HPI
    hpi_region_col = next((c for c in ["regionname", "region_name"] if c in hpi_cols), None)

    if hpi_region_col is None or "district" not in lf_cols:
        logger.warning(
            "Cannot enrich with HPI: missing region columns. Returning data without HPI enrichment."
        )
        return lf

    # Select useful HPI columns
    keep_hpi_cols = [hpi_region_col]
    if "date" in hpi_cols:
        keep_hpi_cols.append("date")
    for col in hpi_cols:
        # Match against both underscore-separated and camelCase-collapsed column names.
        # clean() lowercases and strips spaces but does NOT add underscores to CamelCase:
        # e.g. AveragePrice → averageprice, SalesVolume → salesvolume, 1m%Change → 1m%change.
        if any(
            kw in col
            for kw in [
                "average_price",
                "averageprice",
                "index",
                "percentage_change",
                "%change",
                "sales_volume",
                "salesvolume",
            ]
        ):
            keep_hpi_cols.append(col)
    keep_hpi_cols = list(dict.fromkeys(keep_hpi_cols))  # deduplicate, preserve order
    hpi_subset = hpi_lf.select(keep_hpi_cols)

    # Create year-month key from date column
    if "date" in keep_hpi_cols:
        hpi_subset = hpi_subset.with_columns(
            [
                pl.col("date").cast(pl.Date).dt.year().alias("_hpi_year"),
                pl.col("date").cast(pl.Date).dt.month().alias("_hpi_month"),
            ]
        ).drop("date")

    if "year" not in lf_cols or "month" not in lf_cols:
        logger.warning("Cannot enrich with HPI: sales data missing 'year'/'month' columns.")
        return lf

    # LR stores district in uppercase (e.g. "MANCHESTER"); HPI uses title case ("Manchester").
    # Normalise both to uppercase so the join key matches.
    lf = lf.with_columns(pl.col("district").str.to_uppercase())
    hpi_subset = hpi_subset.with_columns(pl.col(hpi_region_col).str.to_uppercase())

    enriched = lf.join(
        hpi_subset,
        left_on=["district", "year", "month"],
        right_on=[hpi_region_col, "_hpi_year", "_hpi_month"],
        how="left",
        suffix="_hpi",
    )

    logger.info("HPI enrichment plan built (join on district + year-month)")
    return enriched


def save_dataset(
    df: pl.DataFrame | pl.LazyFrame,
    name: str,
    output_dir: Path | None = None,
    fmt: str = "parquet",
) -> Path:
    """Save a DataFrame or LazyFrame to the processed data directory.

    LazyFrames are written via sink_parquet / sink_csv, which execute the
    full query plan in streaming mode — peak memory is O(batch) not O(dataset).

    Args:
        df: DataFrame or LazyFrame to save.
        name: Base filename (without extension).
        output_dir: Override output directory.
        fmt: Output format — "parquet" (default) or "csv".
    """
    out_dir = output_dir or PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(df, pl.LazyFrame):
        if fmt == "parquet":
            dest = out_dir / f"{name}.parquet"
            df.sink_parquet(dest)
        else:
            dest = out_dir / f"{name}.csv"
            df.sink_csv(dest)
        logger.info("Saved streaming output to %s", dest)
    else:
        if fmt == "parquet":
            dest = out_dir / f"{name}.parquet"
            df.write_parquet(dest)
        else:
            dest = out_dir / f"{name}.csv"
            df.write_csv(dest)
        logger.info("Saved %d rows to %s", len(df), dest)

    return dest


def run_pipeline(
    sales: pl.DataFrame | pl.LazyFrame,
    epc: pl.DataFrame | pl.LazyFrame | None = None,
    hpi: pl.DataFrame | pl.LazyFrame | None = None,
    output_name: str = "uk_property_sales",
    output_dir: Path | None = None,
    fmt: str = "parquet",
) -> pl.DataFrame:
    """Run the full data linking pipeline.

    Builds a lazy query plan from all sources, then executes it once using
    collect(streaming=True).  Data is processed in batches so peak memory is
    proportional to batch size rather than dataset size.

    Args:
        sales: Land Registry Price Paid Data (required).
        epc: EPC data (optional — all certificates retained for temporal matching).
        hpi: UK HPI data (optional — will enrich if provided).
        output_name: Name for the output file.
        output_dir: Directory to save output.
        fmt: Output format.

    Returns:
        The collected DataFrame after streaming execution.
    """
    lf = sales.lazy() if isinstance(sales, pl.DataFrame) else sales
    logger.info("Building pipeline plan")

    if epc is not None:
        epc_lf = epc.lazy() if isinstance(epc, pl.DataFrame) else epc
        lf = link_sales_to_epc(lf, epc_lf)

    if hpi is not None:
        hpi_lf = hpi.lazy() if isinstance(hpi, pl.DataFrame) else hpi
        lf = enrich_with_hpi(lf, hpi_lf)

    logger.info("Executing pipeline with streaming engine")
    # polars stubs annotate collect() as InProcessQuery | DataFrame; the streaming
    # engine always returns a DataFrame synchronously, so cast to satisfy ty.
    df: pl.DataFrame = lf.collect(engine="streaming")  # type: ignore[assignment]

    logger.info("Pipeline complete: %d rows, %d columns", len(df), len(df.columns))
    save_dataset(df, output_name, output_dir=output_dir, fmt=fmt)
    return df
