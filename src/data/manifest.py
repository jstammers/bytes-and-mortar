"""Ingest manifest tracking last-updated state per data source.

Stored at `data/raw/.ingest_manifest.json` (gitignored). Each source writes a
small JSON blob recording when it last ran and how far through the upstream
publication calendar it has caught up — used by incremental update paths to
decide which window to fetch.

Schema::

    {
      "<source_name>": {
        "last_run_utc": "2026-05-14T09:30:00Z",
        "last_data_through": "2026-04-30",   # YYYY-MM-DD or null
        "source_url": "https://...",
        "row_count": 30412557
      },
      ...
    }
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

from src.data.config import RAW_DIR

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = ".ingest_manifest.json"


def manifest_path(raw_dir: Path | None = None) -> Path:
    return (raw_dir or RAW_DIR) / MANIFEST_FILENAME


def read_manifest(raw_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Return the manifest dict; empty dict if the file is absent or malformed."""
    path = manifest_path(raw_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("Manifest at %s is malformed; treating as empty", path)
        return {}


def write_manifest(
    source: str,
    *,
    raw_dir: Path | None = None,
    last_data_through: date | datetime | str | None = None,
    source_url: str | None = None,
    row_count: int | None = None,
) -> dict[str, Any]:
    """Update the manifest entry for *source* atomically and return the new entry.

    Unspecified fields are preserved from any prior entry. ``last_run_utc`` is
    always refreshed to the current UTC time.
    """
    raw = raw_dir or RAW_DIR
    raw.mkdir(parents=True, exist_ok=True)
    path = manifest_path(raw)
    data = read_manifest(raw)
    entry = dict(data.get(source, {}))

    entry["last_run_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    if last_data_through is not None:
        entry["last_data_through"] = _coerce_date(last_data_through)
    if source_url is not None:
        entry["source_url"] = source_url
    if row_count is not None:
        entry["row_count"] = int(row_count)

    data[source] = entry
    _atomic_write_json(path, data)
    return entry


def get_last_through(source: str, raw_dir: Path | None = None) -> date | None:
    """Parse and return the ``last_data_through`` date for a source, or None."""
    entry = read_manifest(raw_dir).get(source, {})
    value = entry.get("last_data_through")
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _coerce_date(value: date | datetime | str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return date.fromisoformat(str(value)[:10]).isoformat()


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".manifest-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp_name, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise
