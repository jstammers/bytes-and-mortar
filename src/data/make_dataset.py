"""CLI for the UK property data ingestion pipeline.

Usage:
    # Download and process Land Registry data for a single year:
    bytes-and-mortar download-land-registry --year 2024

    # Download entire EPC dataset (bulk):
    bytes-and-mortar download-epc

    # Download EPC data for a specific local authority (search API):
    bytes-and-mortar download-epc --use-search --local-authority E09000033

    # Download EPC data for postcodes (search API):
    bytes-and-mortar download-epc --use-search --postcodes "SW1A 1AA,E1 6AN"

    # Download UK HPI data:
    bytes-and-mortar download-hpi

    # Run the full pipeline (download + link + save):
    bytes-and-mortar run --year 2024

    # Run pipeline on already-downloaded data:
    bytes-and-mortar run --skip-download
"""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path  # noqa: TC003 - typer needs Path at runtime
from typing import Annotated

import typer
from dotenv import find_dotenv, load_dotenv

from src.data.config import PROCESSED_DIR, RAW_DIR
from src.data.pipeline import run_pipeline
from src.data.sources.epc import EPCData
from src.data.sources.land_registry import LandRegistryPricePaid
from src.data.sources.uk_hpi import UKHousePriceIndex
from src.models.train_model import train_app

app = typer.Typer(help="UK property data ingestion pipeline.")
app.add_typer(train_app, name="train")


class OutputFormat(StrEnum):
    parquet = "parquet"
    csv = "csv"


@app.command()
def download_land_registry(
    year: Annotated[
        int | None,
        typer.Option(help="Download a single year (e.g. 2024). Omit for complete history."),
    ] = None,
    output_dir: Annotated[
        Path | None, typer.Option(help="Override raw data output directory.")
    ] = None,
) -> None:
    """Download HM Land Registry Price Paid Data."""
    logger = logging.getLogger(__name__)
    raw = output_dir if output_dir else RAW_DIR
    source = LandRegistryPricePaid(raw_dir=raw)

    filepath = source.download(year=year)
    logger.info("Downloaded Land Registry data to %s", filepath)


@app.command()
def download_epc(
    use_search: Annotated[
        bool, typer.Option(help="Use search API instead of bulk download.")
    ] = False,
    bulk_file: Annotated[
        str | None,
        typer.Option(
            help="Specific bulk file to download (e.g., 'all-domestic-certificates.zip'). "
            "Only used when use_search=False."
        ),
    ] = None,
    local_authority: Annotated[
        str | None,
        typer.Option(
            help="Local authority code (e.g. E09000033 for Westminster). "
            "Only used when use_search=True."
        ),
    ] = None,
    postcodes: Annotated[
        str | None,
        typer.Option(help="Comma-separated postcodes to query. Only used when use_search=True."),
    ] = None,
    from_year: Annotated[
        int | None,
        typer.Option(help="Fetch EPCs from this year onwards. Only used when use_search=True."),
    ] = None,
    max_pages: Annotated[
        int, typer.Option(help="Maximum number of pages to fetch (search API only).")
    ] = 10,
    output_dir: Annotated[
        Path | None, typer.Option(help="Override raw data output directory.")
    ] = None,
) -> None:
    """Download EPC (Energy Performance Certificate) data.

    By default, downloads the entire dataset via bulk file.
    Use --use-search to download filtered data instead.
    """
    logger = logging.getLogger(__name__)
    raw = output_dir if output_dir else RAW_DIR
    source = EPCData(raw_dir=raw)

    postcode_list = None
    if postcodes:
        postcode_list = [p.strip() for p in postcodes.split(",")]

    filepath = source.download(
        use_search=use_search,
        bulk_file=bulk_file,
        postcodes=postcode_list,
        local_authority=local_authority,
        from_year=from_year,
        max_pages=max_pages,
    )
    logger.info("Downloaded EPC data to %s", filepath)


@app.command()
def download_hpi(
    url: Annotated[str | None, typer.Option(help="Override UK HPI download URL.")] = None,
    output_dir: Annotated[
        Path | None, typer.Option(help="Override raw data output directory.")
    ] = None,
) -> None:
    """Download UK House Price Index data."""
    logger = logging.getLogger(__name__)
    raw = output_dir if output_dir else RAW_DIR
    source = UKHousePriceIndex(raw_dir=raw)

    filepath = source.download(url=url)
    logger.info("Downloaded UK HPI data to %s", filepath)


@app.command()
def run(
    year: Annotated[
        int | None,
        typer.Option(help="Land Registry year to process. Omit for complete history."),
    ] = None,
    nrows: Annotated[
        int | None, typer.Option(help="Limit rows loaded (useful for testing).")
    ] = None,
    skip_download: Annotated[
        bool, typer.Option(help="Skip downloading; use existing raw files.")
    ] = False,
    skip_epc: Annotated[bool, typer.Option(help="Skip EPC data (e.g. if no API token).")] = False,
    skip_hpi: Annotated[bool, typer.Option(help="Skip UK HPI enrichment.")] = False,
    output_name: Annotated[
        str, typer.Option(help="Output filename (without extension).")
    ] = "uk_property_sales",
    fmt: Annotated[OutputFormat, typer.Option(help="Output format.")] = OutputFormat.parquet,
    partition_by: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated column(s) to partition the parquet output by "
            "(e.g. 'year' or 'year,district'). Only applies to parquet format."
        ),
    ] = None,
    output_dir: Annotated[
        Path | None, typer.Option(help="Override processed data output directory.")
    ] = None,
) -> None:
    """Run the full ingestion pipeline: download, clean, link, save."""
    logger = logging.getLogger(__name__)

    out_dir = output_dir if output_dir else PROCESSED_DIR

    # --- Land Registry ---
    lr = LandRegistryPricePaid()
    if not skip_download:
        lr.download(year=year)
    # load() and clean() return LazyFrames — no data is read yet.
    sales_lf = lr.clean(lr.load(nrows=nrows))

    # --- EPC ---
    epc_lf = None
    if not skip_epc:
        try:
            epc = EPCData()
            if not skip_download:
                epc.download()
            epc_lf = epc.clean(epc.load())
        except (ValueError, FileNotFoundError) as e:
            logger.warning("Skipping EPC data: %s", e)

    # --- UK HPI ---
    hpi_lf = None
    if not skip_hpi:
        try:
            hpi = UKHousePriceIndex()
            if not skip_download:
                hpi.download()
            hpi_lf = hpi.clean(hpi.load())
        except FileNotFoundError as e:
            logger.warning("Skipping UK HPI data: %s", e)

    # --- Pipeline ---
    # run_pipeline executes the full lazy plan with streaming=True.
    partition_cols = [c.strip() for c in partition_by.split(",")] if partition_by else None
    df = run_pipeline(
        sales=sales_lf,
        epc=epc_lf,
        hpi=hpi_lf,
        output_name=output_name,
        output_dir=out_dir,
        fmt=fmt.value,
        partition_by=partition_cols,
    )

    typer.echo(
        f"Pipeline complete: {len(df)} rows, {len(df.columns)} columns "
        f"saved to {out_dir / output_name}.{fmt.value}"
    )


@app.callback()
def main() -> None:
    """UK property data ingestion pipeline."""
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())


if __name__ == "__main__":
    app()
