"""Pipeline to link and merge UK property data from multiple sources.

Joins Land Registry Price Paid transactions with EPC property features
on postcode + address matching using join_asof for O(n log m) complexity
with no intermediate fan-out, and enriches with UK HPI area-level data.

All pipeline functions accept and return pl.LazyFrame, building lazy
query plans that are executed via streaming sinks (sink_parquet / sink_csv)
to keep peak memory proportional to the streaming batch size.
"""

import logging
import tempfile
import warnings
from pathlib import Path

import polars as pl

from src.data.config import PROCESSED_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UK address abbreviation expansions applied before exact-address matching.
# Expands abbreviated street types to canonical long-form so that
# "VICTORIA RD" and "VICTORIA ROAD" produce the same _addr_key.
#
# NOTE: "ST" (Saint vs Street) is intentionally omitted — context-free
# expansion corrupts names like "ST JAMES STREET" → "STREET JAMES STREET".
# Cases where abbreviations survive normalisation (e.g. ST vs STREET) will
# remain unmatched; a fuzzy-matching Phase-1b fallback can be added later.
# ---------------------------------------------------------------------------
_UK_ADDR_ABBREVS: list[tuple[str, str]] = [
    (r"\bRD\b", "ROAD"),
    (r"\bAVE\b", "AVENUE"),
    (r"\bAV\b", "AVENUE"),
    (r"\bLN\b", "LANE"),
    (r"\bDR\b", "DRIVE"),
    (r"\bGDNS\b", "GARDENS"),
    (r"\bGDN\b", "GARDEN"),
    (r"\bCRES\b", "CRESCENT"),
    (r"\bCL\b", "CLOSE"),
    (r"\bCT\b", "COURT"),
    (r"\bPL\b", "PLACE"),
    (r"\bGR\b", "GROVE"),
    (r"\bBVD\b", "BOULEVARD"),
]


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


def _expand_abbreviations(expr: pl.Expr) -> pl.Expr:
    """Expand common UK street-type abbreviations to canonical long-form.

    Applied to _addr_key on both the sales and EPC sides before exact
    matching, so that e.g. "VICTORIA RD" and "VICTORIA ROAD" resolve to
    the same key.  Also normalises number+letter suffixes such as "1-A"
    or "1 A" to "1A".

    Only a conservative set of unambiguous abbreviations is expanded here.
    Residual structural differences (e.g. ST vs STREET, FLAT 1 vs 1)
    are left unmatched and can be handled by a fuzzy-matching fallback
    in a future extension.
    """
    # Normalise number-letter suffixes: "1-A" or "1 A" → "1A"
    result = expr.str.replace_all(r"(\d)[\s\-]+([A-Z])\b", r"${1}${2}")
    for abbrev, expansion in _UK_ADDR_ABBREVS:
        result = result.str.replace_all(abbrev, expansion)
    return result


def link_sales_to_epc(
    sales: pl.DataFrame | pl.LazyFrame,
    epc: pl.DataFrame | pl.LazyFrame,
    _phase1_sink: Path | None = None,
) -> pl.LazyFrame:
    """Link Land Registry sales to EPC records using join_asof on exact addresses.

    Strategy
    --------
    1. **Address normalisation** — build ``_addr_key`` (PAON + street /
       address1, uppercased, whitespace-collapsed) and expand common
       street-type abbreviations (RD → ROAD, AVE → AVENUE, etc.) so that
       minor formatting differences don't block an exact match.

    2. **EPC postcode pre-filter** — semi-join EPC to only postcodes present
       in sales, eliminating irrelevant certificates before any joins.

    3. **Phase 1 — join_asof (postcode + address, backward)** — for each sale,
       find the EPC certificate in the *same postcode AND normalised address*
       whose ``inspection_date`` is the most recent date **on or before** the
       sale date.  ``strategy="backward"`` prevents data leakage from
       certificates issued after the transaction (which would be unknowable at
       the time of sale).  Result is always sales-sized — no fan-out.
       Sales with no EPC certificate for their exact address retain null EPC
       columns; the pipeline does **not** assume every sale has an EPC record.

    4. **Phase 2 — 1:1 feature join on lmk_key** — attach the full EPC feature
       set to the sales-sized Phase-1 result.  No fan-out; wide columns are
       deferred to this step to keep Phase-1 intermediates narrow.

    When *_phase1_sink* is a ``Path``, the Phase-1 result is materialised to
    disk before Phase-2 begins, bounding peak memory for the feature join.

    .. note::
        Address matching is exact after normalisation.  Residual structural
        differences (e.g. "ST" vs "STREET", "Flat 1" vs "1A") will leave
        sales unmatched.  A fuzzy-matching fallback is planned as a future
        extension.

    Args:
        sales: Land Registry Price Paid data (required).
        epc: EPC certificate data (required for linking).
        _phase1_sink: Optional path to materialise the Phase-1 result as
            parquet before Phase-2.

    Returns:
        LazyFrame: one row per sale with EPC columns joined where an exact
        address match was found, null EPC columns otherwise.
    """
    sales_lf = sales.lazy() if isinstance(sales, pl.DataFrame) else sales
    epc_lf = epc.lazy() if isinstance(epc, pl.DataFrame) else epc

    sales_cols = sales_lf.collect_schema().names()
    epc_cols = epc_lf.collect_schema().names()

    if "postcode" not in sales_cols or "postcode" not in epc_cols:
        logger.warning("Cannot link sales to EPC: 'postcode' column missing. Returning sales.")
        return sales_lf

    # --- Normalise addresses (with abbreviation expansion) ---
    if "paon" in sales_cols and "street" in sales_cols:
        sales_lf = sales_lf.with_columns(
            _expand_abbreviations(_addr_key_expr("paon", "street")).alias("_addr_key")
        )

    if "address1" in epc_cols:
        epc_lf = epc_lf.with_columns(
            _expand_abbreviations(
                pl.col("address1")
                .fill_null("")
                .cast(pl.String)
                .str.to_uppercase()
                .str.strip_chars()
                .str.replace_all(r"\s+", " ")
            ).alias("_addr_key")
        )

    # Resolve the EPC schema once after address normalisation.
    epc_cols_after_norm = epc_lf.collect_schema().names()

    # lmk_key uniquely identifies each certificate and is the pivot for Phase 2.
    epc_id_col = "lmk_key" if "lmk_key" in epc_cols_after_norm else None

    # Pre-filter EPC to only postcodes that appear in sales.
    epc_lf = epc_lf.join(sales_lf.select("postcode").unique(), on="postcode", how="semi")

    # --- Build slim EPC key frame for the Phase-1 join ---
    # Only include the columns needed for the asof join + Phase-2 pivot key.
    # Full EPC features are attached in Phase 2 to avoid wide intermediates.
    if epc_id_col is not None:
        key_cols = [
            c
            for c in ["postcode", "_addr_key", "inspection_date", epc_id_col]
            if c in epc_cols_after_norm
        ]
    else:
        key_cols = epc_cols_after_norm  # no unique key — carry all columns through Phase 1

    # Right frame for join_asof must be sorted on the asof key.
    # Exclude null inspection_dates — they must never win a backward match.
    epc_key = (
        epc_lf.select(key_cols)
        .filter(pl.col("inspection_date").is_not_null())
        .sort("inspection_date")
    )

    # Cast date_of_transfer to pl.Date for type compatibility with inspection_date.
    # (date_of_transfer is pl.Datetime after load(); inspection_date is pl.Date after clean().)
    sales_lf = sales_lf.with_columns(pl.col("date_of_transfer").cast(pl.Date).alias("_sale_date"))
    sales_sorted = sales_lf.sort("_sale_date", nulls_last=True)

    can_addr_match = (
        "_addr_key" in sales_sorted.collect_schema().names() and "_addr_key" in epc_cols_after_norm
    )

    if can_addr_match:
        # --- Phase 1: exact (postcode + address) match, most recent pre-sale cert ---
        # strategy="backward" picks the newest inspection_date ≤ _sale_date so that
        # we never use EPC data that didn't exist at the time of the transaction.
        # Sales with no matching (postcode, _addr_key) pair in EPC get null EPC columns.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="Sortedness")
            result = sales_sorted.join_asof(
                epc_key,
                left_on="_sale_date",
                right_on="inspection_date",
                by=["postcode", "_addr_key"],
                strategy="backward",
                suffix="_epc",
            )
        logger.info("EPC Phase-1 plan built (join_asof: exact address + postcode, backward)")
    else:
        # Address key not available on one or both sides.
        # Skip the EPC join rather than attempt an imprecise postcode-only match.
        logger.warning(
            "Cannot match EPC by address (_addr_key missing from %s). "
            "Returning sales without EPC columns.",
            ("sales" if "_addr_key" not in sales_sorted.collect_schema().names() else "EPC"),
        )
        return sales_sorted.drop("_sale_date")

    # Drop working columns introduced during normalisation and date casting.
    drop_cols = [
        c
        for c in ["_addr_key", "_addr_key_epc", "_sale_date"]
        if c in result.collect_schema().names()
    ]
    if drop_cols:
        result = result.drop(drop_cols)

    # --- Optionally materialise Phase-1 result to disk ---
    # Sinking here isolates Phase-2's 1:1 feature join from the asof-join graph,
    # bounding peak memory to the Phase-1 output size (~sales rows x key columns).
    if epc_id_col is not None and _phase1_sink is not None:
        _phase1_sink.parent.mkdir(parents=True, exist_ok=True)
        result.sink_parquet(_phase1_sink)
        result = pl.scan_parquet(_phase1_sink)
        logger.info("Phase-1 intermediate written to %s; resuming Phase-2", _phase1_sink)

    # --- Phase 2: attach full EPC features on lmk_key (1:1, no fan-out) ---
    if epc_id_col is not None:
        epc_feature_drop = [
            c for c in ["postcode", "_addr_key", "inspection_date"] if c in epc_cols_after_norm
        ]
        epc_features = epc_lf.drop(epc_feature_drop)
        result = result.join(epc_features, on=epc_id_col, how="left", suffix="_epc")
        logger.info("EPC Phase-2 plan built (1:1 feature join on %s)", epc_id_col)
    else:
        logger.info("EPC join complete (all features from Phase-1; no lmk_key for Phase-2)")

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
            "Saved partitioned parquet dataset to %s (partition keys: %s)",
            dest,
            partition_by,
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

    When EPC data is provided the join uses ``join_asof`` with
    ``strategy="backward"`` (most recent pre-sale cert, exact address match)
    and runs in two passes:

    1. Phase 1 — join_asof on (postcode, normalised address), backward.
       Result is always sales-sized.  Sunk to a temporary parquet file so
       Phase-2 sees a clean, small input.
    2. Phase 2 — 1:1 join of the Phase-1 result to the full EPC feature set
       on lmk_key → sink to the output destination via streaming.

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
        partition_by: Column(s) to partition the parquet output by.

    Returns:
        The collected DataFrame read back from the saved output.
    """
    lf = _to_lazy(sales)
    logger.info("Building pipeline plan")

    if epc is not None:
        epc_lf = _to_lazy(epc)
        with tempfile.TemporaryDirectory(prefix="bytes_mortar_") as _tmp:
            phase1_path = Path(_tmp) / "phase1_dedup.parquet"
            lf = link_sales_to_epc(lf, epc_lf, _phase1_sink=phase1_path)
            if hpi is not None:
                hpi_lf = _to_lazy(hpi)
                lf = enrich_with_hpi(lf, hpi_lf)
            logger.info("Executing pipeline (two-pass EPC join)")
            dest = save_dataset(
                lf,
                output_name,
                output_dir=output_dir,
                fmt=fmt,
                partition_by=partition_by,
            )
    else:
        if hpi is not None:
            hpi_lf = _to_lazy(hpi)
            lf = enrich_with_hpi(lf, hpi_lf)
        logger.info("Executing pipeline with streaming engine")
        dest = save_dataset(
            lf, output_name, output_dir=output_dir, fmt=fmt, partition_by=partition_by
        )

    saved_lf = pl.scan_parquet(dest) if fmt == "parquet" else pl.scan_csv(dest)
    df = pl.DataFrame(saved_lf.collect())
    logger.info("Pipeline complete: %d rows, %d columns", len(df), len(df.columns))
    return df
