from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.app.routers import properties, valuation
from src.app.services.mock_data import MockDataService

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.mock_data = MockDataService()
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


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok"}
