from __future__ import annotations

AREA_BASE_PRICES: dict[str, int] = {
    "W1A": 1_400_000,
    "W1B": 1_300_000,
    "SW1A": 1_200_000,
    "SW1V": 900_000,
    "E1": 550_000,
    "E14": 600_000,
    "E3": 420_000,
    "LS1": 250_000,
    "LS2": 230_000,
    "LS7": 210_000,
    "LS16": 290_000,
    "M1": 280_000,
    "M4": 260_000,
    "M14": 230_000,
    "M20": 310_000,
    "B1": 220_000,
    "B15": 350_000,
    "B16": 270_000,
    "B29": 250_000,
}

PROPERTY_TYPE_MULTIPLIER: dict[str, float] = {
    "D": 1.4,
    "S": 1.15,
    "T": 1.0,
    "F": 0.8,
}

CONFIDENCE_MARGIN = 0.15


class ValuationModel:
    """Mock log-linear valuation model for UK property price estimation."""

    def _compute_price(
        self,
        property_type: str,
        bedrooms: int,
        floor_area: float,
        energy_efficiency_score: int,
        postcode_prefix: str,
    ) -> float:
        base = AREA_BASE_PRICES.get(postcode_prefix.upper(), 300_000)
        type_mult = PROPERTY_TYPE_MULTIPLIER.get(property_type.upper(), 1.0)
        area_factor = (floor_area / 80) ** 0.7
        bedroom_factor = 1 + 0.08 * (bedrooms - 1)
        energy_factor = 1 + 0.003 * (energy_efficiency_score - 55)

        return base * type_mult * area_factor * bedroom_factor * energy_factor

    def predict(
        self,
        property_type: str,
        bedrooms: int,
        floor_area: float,
        energy_efficiency_score: int,
        postcode_prefix: str,
    ) -> tuple[int, int, int]:
        """Return (estimated_price, confidence_lower, confidence_upper)."""
        price = self._compute_price(
            property_type, bedrooms, floor_area, energy_efficiency_score, postcode_prefix
        )
        lower = int(round(price * (1 - CONFIDENCE_MARGIN), -2))
        upper = int(round(price * (1 + CONFIDENCE_MARGIN), -2))
        estimated = int(round(price, -2))
        return estimated, lower, upper

    def sensitivity(
        self,
        base_property_type: str,
        base_bedrooms: int,
        base_floor_area: float,
        base_energy_score: int,
        base_postcode_prefix: str,
        attribute: str,
        values: list[float],
    ) -> list[tuple[float, int]]:
        """Compute sensitivity of price to varying a single attribute."""
        results = []
        for v in values:
            if attribute == "bedrooms":
                price = self._compute_price(
                    base_property_type,
                    int(v),
                    base_floor_area,
                    base_energy_score,
                    base_postcode_prefix,
                )
            elif attribute == "floor_area":
                price = self._compute_price(
                    base_property_type,
                    base_bedrooms,
                    v,
                    base_energy_score,
                    base_postcode_prefix,
                )
            elif attribute == "energy_efficiency_score":
                price = self._compute_price(
                    base_property_type,
                    base_bedrooms,
                    base_floor_area,
                    int(v),
                    base_postcode_prefix,
                )
            else:
                price = self._compute_price(
                    base_property_type,
                    base_bedrooms,
                    base_floor_area,
                    base_energy_score,
                    base_postcode_prefix,
                )
            results.append((v, int(round(price, -2))))
        return results
