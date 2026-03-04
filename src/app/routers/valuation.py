from __future__ import annotations

from fastapi import APIRouter, Request

from src.app.config import SENSITIVITY_CONFIGS
from src.app.models import (
    SensitivityPoint,
    SensitivityResult,
    ValuationInput,
    ValuationPrediction,
    ValuationSensitivityResponse,
)
from src.app.services.valuation_model import AREA_BASE_PRICES, ValuationModel

router = APIRouter()

_model = ValuationModel()


@router.post("/predict", response_model=ValuationPrediction)
async def predict_valuation(
    request: Request,
    body: ValuationInput,
) -> ValuationPrediction:
    estimated, lower, upper = _model.predict(
        property_type=body.property_type,
        bedrooms=body.bedrooms,
        floor_area=body.floor_area,
        energy_efficiency_score=body.energy_efficiency_score,
        postcode_prefix=body.postcode_prefix,
    )
    return ValuationPrediction(
        estimated_price=estimated,
        confidence_lower=lower,
        confidence_upper=upper,
    )


@router.post("/sensitivity", response_model=ValuationSensitivityResponse)
async def valuation_sensitivity(
    request: Request,
    body: ValuationInput,
) -> ValuationSensitivityResponse:
    estimated, lower, upper = _model.predict(
        property_type=body.property_type,
        bedrooms=body.bedrooms,
        floor_area=body.floor_area,
        energy_efficiency_score=body.energy_efficiency_score,
        postcode_prefix=body.postcode_prefix,
    )
    base_prediction = ValuationPrediction(
        estimated_price=estimated,
        confidence_lower=lower,
        confidence_upper=upper,
    )

    sensitivities: list[SensitivityResult] = []
    for config in SENSITIVITY_CONFIGS:
        attr = config.attribute
        pairs = _model.sensitivity(
            base_property_type=body.property_type,
            base_bedrooms=body.bedrooms,
            base_floor_area=body.floor_area,
            base_energy_score=body.energy_efficiency_score,
            base_postcode_prefix=body.postcode_prefix,
            attribute=attr,
            values=[float(v) for v in config.values],
        )
        current_value: float
        if attr == "bedrooms":
            current_value = float(body.bedrooms)
        elif attr == "floor_area":
            current_value = float(body.floor_area)
        else:
            current_value = float(body.energy_efficiency_score)

        points = [
            SensitivityPoint(value=v, price=p, label=label)
            for (v, p), label in zip(pairs, config.labels, strict=False)
        ]
        sensitivities.append(
            SensitivityResult(
                attribute=attr,
                attribute_label=config.attribute_label,
                current_value=current_value,
                points=points,
            )
        )

    return ValuationSensitivityResponse(
        base_prediction=base_prediction,
        sensitivities=sensitivities,
    )


@router.get("/postcode-prefixes")
async def get_postcode_prefixes() -> list[str]:
    return sorted(AREA_BASE_PRICES.keys())
