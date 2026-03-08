from __future__ import annotations

from abc import ABC, abstractmethod

from fastapi import HTTPException, Request


class PropertyDataService(ABC):
    """Abstract interface for property data services."""

    @abstractmethod
    def search_properties(self, query: str) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def get_property(self, property_id: str) -> dict | None:
        raise NotImplementedError

    @abstractmethod
    def get_price_history(self, property_id: str) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def get_area_comparison(self, property_id: str) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_similar_properties(self, property_id: str, n: int = 8) -> list[dict]:
        raise NotImplementedError


def get_data_service(request: Request) -> PropertyDataService:
    service = getattr(request.app.state, "data_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Property data service is not configured.",
        )
    return service
