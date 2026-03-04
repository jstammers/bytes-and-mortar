"""Base class for data source ingestion."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl
import requests
from tqdm import tqdm

from src.data.config import DOWNLOAD_CHUNK_SIZE, RAW_DIR

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract base class for all data sources."""

    name: str = "base"

    def __init__(self, raw_dir: Path | None = None):
        self.raw_dir = raw_dir or RAW_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def download(self, **kwargs) -> Path:
        """Download raw data from the source. Returns path to downloaded file."""

    @abstractmethod
    def load(self, filepath: Path | None = None, **kwargs) -> pl.LazyFrame:
        """Load and parse the raw data into a LazyFrame (no data read yet)."""

    @abstractmethod
    def clean(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Apply source-specific cleaning and standardisation."""

    def ingest(self, **kwargs) -> pl.LazyFrame:
        """Full pipeline: download, load, and clean. Returns a lazy plan."""
        logger.info("Building ingestion plan for %s", self.name)
        filepath = self.download(**kwargs)
        lf = self.load(filepath, **kwargs)
        return self.clean(lf)

    def _download_file(self, url: str, dest: Path, desc: str = "") -> Path:
        """Download a file from a URL with progress bar."""
        if dest.exists():
            logger.info("File already exists, skipping download: %s", dest)
            return dest

        logger.info("Downloading %s from %s", desc or dest.name, url)
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)

        with (
            open(dest, "wb") as f,
            tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=desc or dest.name,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)
                pbar.update(len(chunk))

        logger.info("Downloaded %s to %s", desc or dest.name, dest)
        return dest
