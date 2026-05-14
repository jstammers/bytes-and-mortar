"""HPI forecasting router.

Exposes:
  GET  /api/hpi-forecast/regions  — list of available region names
  POST /api/hpi-forecast           — run a probabilistic HPI forecast
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from src.app.models import (
    HPIForecastPoint,
    HPIForecastRequest,
    HPIForecastResponse,
    HPIHistoricalPoint,
)
from src.app.services.base import PropertyDataService, get_data_service

logger = logging.getLogger(__name__)

router = APIRouter()

_VALID_METHODS = {"ets", "sarima", "trend", "ensemble"}
# Show only the last N months of history in the response to keep payloads
# manageable; the full series is used for fitting.
_HISTORY_DISPLAY_MONTHS = 60


@router.get("/regions", response_model=list[str])
async def get_regions(
    data_service: Annotated[PropertyDataService, Depends(get_data_service)],
) -> list[str]:
    """Return the list of region/district names available for forecasting."""
    return sorted(data_service.get_hpi_regions())


@router.post("", response_model=HPIForecastResponse)
async def forecast_hpi(
    request: HPIForecastRequest,
    data_service: Annotated[PropertyDataService, Depends(get_data_service)],
) -> HPIForecastResponse:
    """Fit an HPI forecaster on historical data and return probabilistic forecasts.

    The response includes the last ``_HISTORY_DISPLAY_MONTHS`` months of historical
    data (used for chart context) and ``steps`` months of forward forecasts with
    lower/upper prediction interval bounds.
    """
    if request.method not in _VALID_METHODS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid method '{request.method}'. Choose from: {sorted(_VALID_METHODS)}",
        )
    if not (1 <= request.steps <= 60):
        raise HTTPException(status_code=422, detail="steps must be between 1 and 60")
    if not (0.01 <= request.alpha <= 0.50):
        raise HTTPException(status_code=422, detail="alpha must be between 0.01 and 0.50")

    iso_dates, values = data_service.get_hpi_series(request.region)
    if not iso_dates:
        raise HTTPException(
            status_code=404,
            detail=f"No HPI data found for region '{request.region}'",
        )

    if len(values) < 24:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Region '{request.region}' has only {len(values)} data points. "
                "At least 24 months of history are required for forecasting."
            ),
        )

    dates = [date.fromisoformat(d) for d in iso_dates]
    import numpy as np

    values_arr = np.array(values, dtype=float)

    forecaster = _build_forecaster(request.method)
    try:
        forecaster.fit(values_arr, dates)
        result = forecaster.forecast(steps=request.steps, alpha=request.alpha)
    except Exception as exc:
        logger.exception("Forecaster %s failed for region %s", request.method, request.region)
        raise HTTPException(status_code=500, detail=f"Forecasting failed: {exc}") from exc

    # Trim history for display
    hist_start = max(0, len(iso_dates) - _HISTORY_DISPLAY_MONTHS)
    historical = [
        HPIHistoricalPoint(date=iso_dates[i], value=round(values[i], 2))
        for i in range(hist_start, len(iso_dates))
    ]

    forecast_points = [
        HPIForecastPoint(
            date=result.dates[i].isoformat(),
            point=round(float(result.point[i]), 2),
            lower=round(float(result.lower[i]), 2),
            upper=round(float(result.upper[i]), 2),
        )
        for i in range(result.horizon)
    ]

    return HPIForecastResponse(
        region=request.region,
        method=result.method,
        alpha=request.alpha,
        historical=historical,
        forecast=forecast_points,
    )


def _build_forecaster(method: str):  # type: ignore[return]
    """Instantiate the requested HPIForecaster."""
    from src.models.hpi_forecast import (
        EnsembleForecaster,
        ETSForecaster,
        SARIMAForecaster,
        TrendForecaster,
    )

    if method == "ets":
        return ETSForecaster()
    if method == "sarima":
        return SARIMAForecaster()
    if method == "trend":
        return TrendForecaster()
    # default: ensemble
    return EnsembleForecaster()
