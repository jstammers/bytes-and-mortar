from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from src.app.models import (
    AreaComparisonData,
    PriceHistoryPoint,
    PropertyDetail,
    PropertyHistory,
    PropertySearchResult,
    SimilarProperty,
)

router = APIRouter()


@router.get("/search", response_model=list[PropertySearchResult])
async def search_properties(
    request: Request,
    q: str = Query(..., min_length=1, description="Postcode prefix or address substring"),
) -> list[PropertySearchResult]:
    service = request.app.state.mock_data
    results = service.search_properties(q)
    return [
        PropertySearchResult(
            id=p["id"],
            address=p["address"],
            postcode=p["postcode"],
            district=p["district"],
            property_type=p["property_type"],
            property_type_code=p["property_type_code"],
            last_sale_price=p["last_sale_price"],
            last_sale_date=p["last_sale_date"],
            energy_rating=p.get("energy_rating"),
            floor_area=p.get("floor_area"),
            bedrooms=p.get("bedrooms"),
        )
        for p in results
    ]


@router.get("/{property_id}", response_model=PropertyDetail)
async def get_property(request: Request, property_id: str) -> PropertyDetail:
    service = request.app.state.mock_data
    prop = service.get_property(property_id)
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")
    return PropertyDetail(
        id=prop["id"],
        address=prop["address"],
        postcode=prop["postcode"],
        district=prop["district"],
        property_type=prop["property_type"],
        property_type_code=prop["property_type_code"],
        tenure=prop["tenure"],
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
async def get_price_history(request: Request, property_id: str) -> PropertyHistory:
    service = request.app.state.mock_data
    prop = service.get_property(property_id)
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")
    transactions = service.get_price_history(property_id)
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
async def get_area_comparison(request: Request, property_id: str) -> AreaComparisonData:
    service = request.app.state.mock_data
    prop = service.get_property(property_id)
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")
    data = service.get_area_comparison(property_id)
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
    request: Request,
    property_id: str,
    n: int = Query(default=8, ge=1, le=20),
) -> list[SimilarProperty]:
    service = request.app.state.mock_data
    prop = service.get_property(property_id)
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")
    similar = service.get_similar_properties(property_id, n=n)
    return [
        SimilarProperty(
            id=s["id"],
            address=s["address"],
            postcode=s["postcode"],
            property_type=s["property_type"],
            floor_area=s["floor_area"],
            bedrooms=s["bedrooms"],
            energy_rating=s["energy_rating"],
            last_sale_price=s["last_sale_price"],
            last_sale_date=s["last_sale_date"],
            similarity_score=s["similarity_score"],
        )
        for s in similar
    ]
