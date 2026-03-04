"""Pipeline to link and merge UK property data from multiple sources.

Joins Land Registry Price Paid transactions with EPC property features
on postcode + address matching, and enriches with UK HPI area-level data.

All pipeline functions accept and return pl.LazyFrame, building a single
lazy query plan that is executed once at the end with streaming=True. This
allows datasets larger than available RAM to be processed efficiently.
"""

import logging
import tempfile
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
    _phase1_sink: Path | None = None,
) -> pl.LazyFrame:
    """Link Land Registry sales to EPC records, returning a lazy plan.

    Strategy:
    1. Normalise addresses in both datasets.
    2. Phase 1 — left join sales to a *slim* EPC frame (postcode + scoring columns
       only) on postcode.  This produces all candidate EPC records per sale but
       keeps each intermediate row narrow, reducing the memory footprint of the
       subsequent sort+unique step by ~20x vs. joining the full feature set.
    3. Score each candidate:
         0 = postcode AND normalised address match  (best)
         1 = postcode match only                    (fallback)
         2 = no EPC record found for that postcode  (unmatched)
    4. Compute |sale_date - inspection_date| in days (null when unmatched).
    5. Sort by (transaction_id, match_score, date_diff) and keep the first
       row per transaction — i.e. the best temporal match per sale.
    6. Phase 2 — 1:1 join the deduplicated (sales-sized) frame back to the full
       EPC feature set on lmk_key.  No fan-out; all wide EPC columns are attached
       only after the result is already one-row-per-sale.

    This produces a left join result: sales without any EPC match are retained
    with null EPC columns (match_score = 2).

    When *_phase1_sink* is a ``Path``, the Phase-1 result (after deduplication)
    is materialised to that file via ``sink_parquet`` before Phase-2 begins.
    This breaks the lazy plan in two, allowing the streaming engine to fully
    flush the large postcode fan-out before the 1:1 feature join, which keeps
    peak memory proportional to the streaming batch size rather than the full
    cross-product of sales × EPC records per postcode.

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

    # Resolve the EPC schema once after address normalisation (epc_lf may have
    # gained _addr_key above).
    epc_cols_after_norm = epc_lf.collect_schema().names()

    # lmk_key uniquely identifies each certificate and is the pivot for Phase 2.
    epc_id_col = "lmk_key" if "lmk_key" in epc_cols_after_norm else None

    # Pre-filter EPC to only postcodes that appear in sales.  This eliminates
    # certificates for postcodes with no corresponding sales transactions and
    # reduces the Phase-1 fan-out — a semi-join has no effect on column schema.
    epc_lf = epc_lf.join(sales_lf.select("postcode").unique(), on="postcode", how="semi")

    # --- Phase 1: join on postcode using only the columns needed for scoring ---
    # The postcode join fans out to (sales x EPC candidates per postcode) rows.
    # Restricting EPC to a slim key frame means each of those rows carries only
    # 3-4 columns rather than 80+, so the sort+unique step processes far less data.
    if epc_id_col is not None:
        key_cols = [
            c
            for c in ["postcode", "_addr_key", "inspection_date", epc_id_col]
            if c in epc_cols_after_norm
        ]
        join_target = epc_lf.select(key_cols)
    else:
        # No unique certificate key available — fall back to joining the full frame.
        join_target = epc_lf

    # EPC columns that conflict with sales columns (except the join key) are
    # renamed with the "_epc" suffix; e.g. _addr_key → _addr_key_epc.
    merged = sales_lf.join(join_target, on="postcode", how="left", suffix="_epc")
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

    # --- Drop working columns used for scoring ---
    drop_cols = [
        c
        for c in ["_addr_key", "_addr_key_epc", "_match_score", "_date_diff"]
        if c in result.collect_schema().names()
    ]
    if drop_cols:
        result = result.drop(drop_cols)

    # --- Materialise Phase-1 result to disk (optional) ---
    # Sinking here breaks the lazy plan so the streaming engine can fully flush
    # the large postcode fan-out before the 1:1 Phase-2 join.  Without this,
    # both the fan-out intermediate and the full EPC feature set must coexist in
    # the plan, driving peak memory to O(sales × EPC-per-postcode × full-width).
    if epc_id_col is not None and _phase1_sink is not None:
        _phase1_sink.parent.mkdir(parents=True, exist_ok=True)
        result.sink_parquet(_phase1_sink)
        result = pl.scan_parquet(_phase1_sink)
        logger.info("Phase-1 intermediate written to %s; resuming Phase-2", _phase1_sink)

    # --- Phase 2: join winning certificate back to the full EPC feature set ---
    # result is now sales-sized (one row per transaction).  This 1:1 join on
    # lmk_key attaches all EPC feature columns with no fan-out.
    # We exclude from the EPC side columns already present in result to avoid
    # spurious _epc-suffixed duplicates:
    #   postcode        - already in sales
    #   _addr_key       - working column not wanted in output
    #   inspection_date - retained from the Phase-1 key join (identical value)
    if epc_id_col is not None:
        epc_feature_drop = [
            c for c in ["postcode", "_addr_key", "inspection_date"] if c in epc_cols_after_norm
        ]
        epc_features = epc_lf.drop(epc_feature_drop)
        result = result.join(epc_features, on=epc_id_col, how="left", suffix="_epc")
        logger.info(
            "EPC join plan built (two-phase: slim key join → deduplicate → full feature join)"
        )
    else:
        logger.info("EPC join plan built (single-phase: full join → deduplicate per sale)")

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
    partition_by: list[str] | None = None,
) -> Path:
    """Save a DataFrame or LazyFrame to the processed data directory.

    LazyFrames are written via sink_parquet / sink_csv, which execute the
    full query plan in streaming mode — peak memory is O(batch) not O(dataset).

    When *partition_by* is provided the output is written as a hive-partitioned
    parquet dataset (e.g. ``name/year=2024/part-0.parquet``).  The return value
    is the dataset directory rather than a single file.  CSV format does not
    support partitioning; *partition_by* is silently ignored when fmt="csv".

    Args:
        df: DataFrame or LazyFrame to save.
        name: Base filename (without extension), or directory name when partitioning.
        output_dir: Override output directory.
        fmt: Output format — "parquet" (default) or "csv".
        partition_by: Column(s) to partition by (e.g. ["year"] or ["year", "district"]).
            Only applies to parquet output; uses Polars hive-style partitioning.
    """
    out_dir = output_dir or PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if partition_by and fmt == "parquet":
        dest = out_dir / name
        dest.mkdir(parents=True, exist_ok=True)
        if isinstance(df, pl.LazyFrame):
            # PartitionBy was renamed from PartitionByKey between Polars 1.30 and 1.38.
            if hasattr(pl, "PartitionBy"):
                scheme = pl.PartitionBy(dest, key=partition_by)
            else:
                scheme = pl.PartitionByKey(dest, by=partition_by)
            df.sink_parquet(scheme, mkdir=True)
        else:
            df.write_parquet(dest, partition_by=partition_by)
        logger.info(
            "Saved partitioned parquet dataset to %s (partition keys: %s)", dest, partition_by
        )
        return dest

    if partition_by and fmt != "parquet":
        logger.warning("partition_by is only supported for parquet; ignoring for fmt=%s", fmt)

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


def _to_lazy(src: pl.DataFrame | pl.LazyFrame | Path) -> pl.LazyFrame:
    """Coerce a DataFrame, LazyFrame, or Path to a LazyFrame.

    Path inputs must point to a ``.parquet`` or ``.csv`` file.  Parquet is
    scanned with ``pl.scan_parquet``; CSV is scanned with ``pl.scan_csv``
    using default options (suitable for files that have already been cleaned
    by a source's ``load()`` method).
    """
    if isinstance(src, Path):
        if not src.exists():
            raise FileNotFoundError(f"Data file not found: {src}")
        return pl.scan_parquet(src) if src.suffix == ".parquet" else pl.scan_csv(src)
    return src.lazy() if isinstance(src, pl.DataFrame) else src


def run_pipeline(
    sales: pl.DataFrame | pl.LazyFrame | Path,
    epc: pl.DataFrame | pl.LazyFrame | Path | None = None,
    hpi: pl.DataFrame | pl.LazyFrame | Path | None = None,
    output_name: str = "uk_property_sales",
    output_dir: Path | None = None,
    fmt: str = "parquet",
    partition_by: list[str] | None = None,
) -> pl.DataFrame:
    """Run the full data linking pipeline.

    When EPC data is provided the join runs in two streaming passes:

    1. Phase 1 — postcode fan-out join → score → sort → unique → sink to a
       temporary parquet file.  Peak memory is proportional to a single
       streaming batch, not the full cross-product of sales × EPC records per
       postcode.
    2. Phase 2 — 1:1 join of the (sales-sized) Phase-1 result to the full EPC
       feature set → sink directly to the output destination.

    The temporary Phase-1 file lives inside a ``tempfile.TemporaryDirectory``
    that is cleaned up once Phase-2 finishes writing the final output.

    Args:
        sales: Land Registry Price Paid Data (required).  Accepts a
            LazyFrame/DataFrame or a ``Path`` to a ``.parquet`` or ``.csv``
            file.  Parquet is preferred — pass a pre-built parquet to avoid
            the CSV brace-stripping overhead on repeat runs.
        epc: EPC data (optional — all certificates retained for temporal
            matching).  Same type options as *sales*.
        hpi: UK HPI data (optional — will enrich if provided).  Same type
            options as *sales*.
        output_name: Name for the output file or partition directory.
        output_dir: Directory to save output.
        fmt: Output format.
        partition_by: Column(s) to partition the parquet output by (e.g. ["year"]).

    Returns:
        The collected DataFrame read back from the saved output.  The pipeline
        itself runs via streaming sinks (O(batch) peak memory).  For datasets
        that are too large to fit in RAM, use ``pl.scan_parquet(dest)`` on the
        saved file instead of consuming the return value.
    """
    lf = _to_lazy(sales)
    logger.info("Building pipeline plan")

    if epc is not None:
        epc_lf = _to_lazy(epc)
        # The EPC join is split into two streaming passes via a temporary file
        # so the streaming engine can flush the large Phase-1 postcode fan-out
        # before the 1:1 Phase-2 join.  The TemporaryDirectory context keeps
        # the Phase-1 file alive until save_dataset finishes writing the output.
        with tempfile.TemporaryDirectory(prefix="bytes_mortar_") as _tmp:
            phase1_path = Path(_tmp) / "phase1_dedup.parquet"
            lf = link_sales_to_epc(lf, epc_lf, _phase1_sink=phase1_path)
            if hpi is not None:
                hpi_lf = _to_lazy(hpi)
                lf = enrich_with_hpi(lf, hpi_lf)
            logger.info("Executing pipeline (streaming two-pass EPC join)")
            dest = save_dataset(
                lf, output_name, output_dir=output_dir, fmt=fmt, partition_by=partition_by
            )
    else:
        if hpi is not None:
            hpi_lf = _to_lazy(hpi)
            lf = enrich_with_hpi(lf, hpi_lf)
        logger.info("Executing pipeline with streaming engine")
        dest = save_dataset(
            lf, output_name, output_dir=output_dir, fmt=fmt, partition_by=partition_by
        )

    # Re-read from the saved file to return a DataFrame for API callers.
    # The heavy lifting above ran via streaming sinks; this collect just reads
    # the already-persisted result.  For very large outputs, prefer
    # pl.scan_parquet(dest) over consuming this return value.
    saved_lf = pl.scan_parquet(dest) if fmt == "parquet" else pl.scan_csv(dest)
    df: pl.DataFrame = saved_lf.collect()
    logger.info("Pipeline complete: %d rows, %d columns", len(df), len(df.columns))
    return df
