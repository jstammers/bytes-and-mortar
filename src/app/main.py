from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.app.routers import hpi_forecast, properties, valuation
from src.app.services.mock_data import MockDataService
from src.app.services.sales_data import SalesDataService
from src.data.config import PROCESSED_DATA_PATH

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


def _build_service() -> MockDataService | SalesDataService:
    """Select the data service based on environment variables.

    Resolution order:
    1. ``APP_ENV=dev`` → MockDataService.
    2. ``UK_SALES_DATA_PATH`` env var → SalesDataService from that path.
    3. Default processed parquet exists → SalesDataService auto-detected.
    4. Fallback → MockDataService with a warning.
    """
    if os.getenv("APP_ENV", "").lower() == "dev":
        logger.info("APP_ENV=dev: using MockDataService")
        return MockDataService()

    sales_path_env = os.getenv("UK_SALES_DATA_PATH")
    if sales_path_env:
        path = Path(sales_path_env)
        logger.info("UK_SALES_DATA_PATH: loading SalesDataService from %s", path)
        return SalesDataService(path)

    if PROCESSED_DATA_PATH.exists():
        logger.info("Auto-detected parquet at %s: using SalesDataService", PROCESSED_DATA_PATH)
        return SalesDataService(PROCESSED_DATA_PATH)

    logger.warning("No parquet found at %s — falling back to MockDataService", PROCESSED_DATA_PATH)
    return MockDataService()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.data_service = _build_service()
    yield


app = FastAPI(
    title="Property Insights API",
    description="UK property valuation and data API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,  # ty:ignore[invalid-argument-type]
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(properties.router, prefix="/api/properties", tags=["properties"])
app.include_router(valuation.router, prefix="/api/valuation", tags=["valuation"])
app.include_router(hpi_forecast.router, prefix="/api/hpi-forecast", tags=["hpi-forecast"])


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok"}
