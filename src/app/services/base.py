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

    @abstractmethod
    def get_hpi_series(self, region: str) -> tuple[list[str], list[float]]:
        """Return (iso_dates, average_prices) for a named region's HPI series."""
        raise NotImplementedError

    @abstractmethod
    def get_hpi_regions(self) -> list[str]:
        """Return the list of available region/district names."""
        raise NotImplementedError


def get_data_service(request: Request) -> PropertyDataService:
    service = getattr(request.app.state, "data_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Property data service is not configured.",
        )
    return service
