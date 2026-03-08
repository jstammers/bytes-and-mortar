from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from src.app.models import (
    AreaComparisonData,
    PriceHistoryPoint,
    PropertyDetail,
    PropertyHistory,
    PropertySearchResult,
    SimilarProperty,
)
from src.app.services.base import PropertyDataService, get_data_service

router = APIRouter()


@router.get("/search", response_model=list[PropertySearchResult])
async def search_properties(
    data_service: Annotated[PropertyDataService, Depends(get_data_service)],
    q: str = Query(..., min_length=1, description="Postcode prefix or address substring"),
) -> list[PropertySearchResult]:
    results = data_service.search_properties(q)
    return [
        PropertySearchResult(
            id=p["id"],
            address=p["address"],
            postcode=p["postcode"],
            district=p["district"],
            property_type=p["property_type"],
            property_type_code=p["property_type_code"],
            tenure=p.get("tenure", "Unknown"),
            last_sale_price=p["last_sale_price"],
            last_sale_date=p["last_sale_date"],
            energy_rating=p.get("energy_rating"),
            floor_area=p.get("floor_area"),
            bedrooms=p.get("bedrooms"),
        )
        for p in results
    ]


@router.get("/{property_id}", response_model=PropertyDetail)
async def get_property(
    data_service: Annotated[PropertyDataService, Depends(get_data_service)],
    property_id: str,
) -> PropertyDetail:
    prop = data_service.get_property(property_id)
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")
    return PropertyDetail(
        id=prop["id"],
        address=prop["address"],
        postcode=prop["postcode"],
        district=prop["district"],
        property_type=prop["property_type"],
        property_type_code=prop["property_type_code"],
        tenure=prop.get("tenure", "Unknown"),
        last_sale_price=prop["last_sale_price"],
        last_sale_date=prop["last_sale_date"],
        energy_rating=prop.get("energy_rating"),
        current_energy_efficiency=prop.get("current_energy_efficiency"),
        potential_energy_efficiency=prop.get("potential_energy_efficiency"),
        floor_area=prop.get("floor_area"),
        bedrooms=prop.get("bedrooms"),
        heated_rooms=prop.get("heated_rooms"),
        co2_emissions_current=prop.get("co2_emissions_current"),
        co2_emissions_potential=prop.get("co2_emissions_potential"),
        construction_age_band=prop.get("construction_age_band"),
    )


@router.get("/{property_id}/history", response_model=PropertyHistory)
async def get_price_history(
    data_service: Annotated[PropertyDataService, Depends(get_data_service)],
    property_id: str,
) -> PropertyHistory:
    prop = data_service.get_property(property_id)
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")
    transactions = data_service.get_price_history(property_id)
    return PropertyHistory(
        property_id=property_id,
        transactions=[
            PriceHistoryPoint(
                date=t["date"],
                price=t["price"],
                transaction_id=t["transaction_id"],
            )
            for t in transactions
        ],
    )


@router.get("/{property_id}/area-comparison", response_model=AreaComparisonData)
async def get_area_comparison(
    data_service: Annotated[PropertyDataService, Depends(get_data_service)],
    property_id: str,
) -> AreaComparisonData:
    prop = data_service.get_property(property_id)
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")
    data = data_service.get_area_comparison(property_id)
    return AreaComparisonData(
        property_id=data["property_id"],
        district=data["district"],
        sale_dates=data["sale_dates"],
        sale_prices=data["sale_prices"],
        hpi_dates=data["hpi_dates"],
        hpi_average_prices=data["hpi_average_prices"],
    )


@router.get("/{property_id}/similar", response_model=list[SimilarProperty])
async def get_similar_properties(
    data_service: Annotated[PropertyDataService, Depends(get_data_service)],
    property_id: str,
    n: int = Query(default=8, ge=1, le=20),
) -> list[SimilarProperty]:
    prop = data_service.get_property(property_id)
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")
    similar = data_service.get_similar_properties(property_id, n=n)
    return [
        SimilarProperty(
            id=s["id"],
            address=s["address"],
            postcode=s["postcode"],
            property_type=s["property_type"],
            floor_area=s.get("floor_area"),
            bedrooms=s.get("bedrooms"),
            energy_rating=s.get("energy_rating"),
            last_sale_price=s["last_sale_price"],
            last_sale_date=s["last_sale_date"],
            similarity_score=s["similarity_score"],
        )
        for s in similar
    ]
