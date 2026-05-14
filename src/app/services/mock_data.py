from __future__ import annotations

import hashlib
import string
from datetime import date, timedelta

import numpy as np

DISTRICTS = [
    {
        "name": "City of Westminster",
        "area_code": "E09000033",
        "postcode_prefixes": ["W1A", "W1B", "SW1A", "SW1V"],
        "base_price_2000": 550_000,
        "annual_growth": 0.056,
        "volatility": 0.08,
        "dominant_types": ["F", "F", "F", "S"],
    },
    {
        "name": "Tower Hamlets",
        "area_code": "E09000030",
        "postcode_prefixes": ["E1", "E14", "E3"],
        "base_price_2000": 185_000,
        "annual_growth": 0.072,
        "volatility": 0.10,
        "dominant_types": ["F", "F", "T", "S"],
    },
    {
        "name": "Leeds",
        "area_code": "E08000035",
        "postcode_prefixes": ["LS1", "LS2", "LS7", "LS16"],
        "base_price_2000": 88_000,
        "annual_growth": 0.045,
        "volatility": 0.07,
        "dominant_types": ["T", "T", "S", "D"],
    },
    {
        "name": "Manchester",
        "area_code": "E08000003",
        "postcode_prefixes": ["M1", "M4", "M14", "M20"],
        "base_price_2000": 92_000,
        "annual_growth": 0.050,
        "volatility": 0.075,
        "dominant_types": ["F", "T", "S", "D"],
    },
    {
        "name": "Birmingham",
        "area_code": "E08000025",
        "postcode_prefixes": ["B1", "B15", "B16", "B29"],
        "base_price_2000": 84_000,
        "annual_growth": 0.042,
        "volatility": 0.07,
        "dominant_types": ["S", "D", "T", "F"],
    },
]

STREET_NAMES = [
    "High Street",
    "Church Lane",
    "Manor Road",
    "Victoria Road",
    "Station Road",
    "Park Avenue",
    "Kings Road",
    "Queens Drive",
    "Elm Street",
    "Oak Close",
    "Maple Avenue",
    "Rose Hill",
    "Brook Lane",
    "Mill Road",
    "Castle Street",
    "Bridge Road",
    "Green Lane",
    "Meadow View",
    "Riverside Walk",
    "Chestnut Drive",
]

ENERGY_RATINGS = ["A", "B", "C", "C", "C", "D", "D", "D", "E", "E", "F", "G"]
ENERGY_RATING_WEIGHTS = [0.02, 0.05, 0.18, 0.18, 0.18, 0.17, 0.17, 0.17, 0.08, 0.08, 0.04, 0.02]
ENERGY_RATINGS_UNIQUE = ["A", "B", "C", "D", "E", "F", "G"]
ENERGY_RATING_WEIGHTS_UNIQUE = [0.02, 0.05, 0.20, 0.28, 0.25, 0.12, 0.08]

CONSTRUCTION_AGE_BANDS = [
    "England and Wales: before 1900",
    "England and Wales: 1900-1929",
    "England and Wales: 1930-1949",
    "England and Wales: 1950-1966",
    "England and Wales: 1967-1975",
    "England and Wales: 1976-1982",
    "England and Wales: 1983-1990",
    "England and Wales: 1991-1995",
    "England and Wales: 1996-2002",
    "England and Wales: 2003-2006",
    "England and Wales: 2007-2011",
    "England and Wales: 2012 onwards",
]

PROPERTY_TYPE_NAMES = {
    "D": "Detached",
    "S": "Semi-Detached",
    "T": "Terraced",
    "F": "Flat/Maisonette",
}

PROPERTY_TYPE_MULTIPLIERS = {
    "D": 1.4,
    "S": 1.15,
    "T": 1.0,
    "F": 0.8,
}

FLOOR_AREA_RANGES = {
    "D": (100, 300),
    "S": (70, 150),
    "T": (60, 130),
    "F": (40, 100),
}

BEDROOM_RANGES = {
    "D": (3, 6),
    "S": (3, 5),
    "T": (2, 4),
    "F": (1, 3),
}


def _make_property_id(address: str, postcode: str) -> str:
    return hashlib.md5(f"{address}_{postcode}".encode()).hexdigest()[:12]


def _random_postcode_suffix(rng: np.random.Generator) -> str:
    digit = str(rng.integers(1, 9))
    letters = "".join(rng.choice(list(string.ascii_uppercase), size=2))
    return f"{digit}{letters}"


def _apply_2008_crash(price: float, year: int, month: int) -> float:
    """Apply 2008 financial crisis effect."""
    # Crisis period: mid-2007 to end of 2009
    if year == 2007 and month >= 8:
        factor = 1.0 - 0.02 * (month - 7)
        return price * max(factor, 0.85)
    elif year == 2008:
        return price * (0.82 + 0.005 * month)
    elif year == 2009:
        return price * (0.84 + 0.01 * month)
    return price


class MockDataService:
    def __init__(self) -> None:
        self._rng = np.random.default_rng(42)
        self._properties: dict[str, dict] = {}
        self._price_histories: dict[str, list[dict]] = {}
        self._hpi_series: dict[str, list[dict]] = {}

        self._generate_all()

    def _generate_all(self) -> None:
        for district in DISTRICTS:
            self._generate_hpi_series(district)
            for _ in range(10):
                self._generate_property(district)

    def _generate_hpi_series(self, district: dict) -> None:
        base_price = float(district["base_price_2000"])
        annual_growth = district["annual_growth"]
        volatility = district["volatility"]
        monthly_growth = annual_growth / 12
        monthly_vol = volatility / (12**0.5)

        series = []
        price = base_price
        start_date = date(2000, 1, 1)

        for i in range(300):  # Jan 2000 to Dec 2024
            month_offset = i
            year = 2000 + month_offset // 12
            month = (month_offset % 12) + 1
            current_date = start_date + timedelta(days=month_offset * 30)
            # Use the actual date
            d = date(year, month, 1)

            noise = self._rng.normal(0, monthly_vol)
            growth = monthly_growth + noise

            # 2008 crash: months 91-108 (Aug 2007 to Dec 2008)
            if 91 <= i <= 108:
                growth -= 0.020
            # Recovery 2009-2010: months 109-132
            elif 109 <= i <= 132:
                growth += 0.005

            # COVID dip: months 243-246 (Apr-Jul 2020)
            if 243 <= i <= 246:
                growth -= 0.015
            # Post-COVID boom: months 247-264
            elif 247 <= i <= 264:
                growth += 0.010

            price = price * (1 + growth)
            price = max(price, base_price * 0.5)

            series.append(
                {
                    "date": d.isoformat(),
                    "average_price": round(price, 2),
                    "district": district["name"],
                }
            )

        _ = current_date  # suppress unused warning
        self._hpi_series[district["name"]] = series

    def _generate_property(self, district: dict) -> None:
        prefix = self._rng.choice(district["postcode_prefixes"])
        suffix = _random_postcode_suffix(self._rng)
        postcode = f"{prefix} {suffix}"

        street = self._rng.choice(STREET_NAMES)
        house_number = int(self._rng.integers(1, 201))
        address = f"{house_number} {street}"

        prop_type_idx = int(self._rng.integers(0, len(district["dominant_types"])))
        prop_type = district["dominant_types"][prop_type_idx]

        fa_min, fa_max = FLOOR_AREA_RANGES[prop_type]
        floor_area = round(float(self._rng.uniform(fa_min, fa_max)), 1)

        bed_min, bed_max = BEDROOM_RANGES[prop_type]
        bedrooms = int(self._rng.integers(bed_min, bed_max + 1))

        heated_rooms = bedrooms + int(self._rng.integers(1, 3))

        energy_rating = str(self._rng.choice(ENERGY_RATINGS_UNIQUE, p=ENERGY_RATING_WEIGHTS_UNIQUE))

        # Energy efficiency score based on rating
        rating_to_score = {
            "A": (92, 100),
            "B": (81, 91),
            "C": (69, 80),
            "D": (55, 68),
            "E": (39, 54),
            "F": (21, 38),
            "G": (1, 20),
        }
        score_min, score_max = rating_to_score[energy_rating]
        current_efficiency = int(self._rng.integers(score_min, score_max + 1))
        potential_efficiency = min(100, current_efficiency + int(self._rng.integers(5, 25)))

        # CO2 emissions (inversely related to efficiency)
        co2_current = round(float(self._rng.uniform(1.5, 8.0) * (100 - current_efficiency) / 50), 2)
        co2_potential = round(co2_current * 0.7, 2)

        age_band = str(self._rng.choice(CONSTRUCTION_AGE_BANDS))

        # Tenure: Flats usually leasehold
        if prop_type == "F":
            tenure = "Leasehold" if self._rng.random() > 0.1 else "Freehold"
        else:
            tenure = "Freehold" if self._rng.random() > 0.1 else "Leasehold"

        prop_id = _make_property_id(address, postcode)

        # Generate price history
        base_price = (
            district["base_price_2000"]
            * PROPERTY_TYPE_MULTIPLIERS[prop_type]
            * ((floor_area / 80) ** 0.7)
        )

        num_sales = int(self._rng.integers(3, 9))
        start_year = int(self._rng.integers(2000, 2011))

        transactions = []
        price = base_price * self._rng.uniform(0.85, 1.15)
        sale_year = start_year

        for t in range(num_sales):
            years_gap = int(self._rng.integers(2, 7))
            sale_year = sale_year + years_gap
            if sale_year > 2024:
                break

            sale_month = int(self._rng.integers(1, 13))
            sale_day = int(self._rng.integers(1, 28))
            sale_date = date(sale_year, sale_month, sale_day)

            # Apply growth from previous sale
            years_elapsed = years_gap
            annual = district["annual_growth"]
            noise = self._rng.normal(0, district["volatility"])
            growth_factor = (1 + annual + noise) ** years_elapsed

            # 2008 crash effect
            if 2007 <= sale_year <= 2009:
                crash_factor = 0.80 if sale_year == 2008 else 0.90
                growth_factor *= crash_factor

            price = price * growth_factor
            price = max(price, base_price * 0.4)

            tx_id = hashlib.md5(f"{prop_id}_{sale_date}_{t}".encode()).hexdigest()[:16]
            transactions.append(
                {
                    "date": sale_date.isoformat(),
                    "price": int(round(price, -2)),
                    "transaction_id": tx_id,
                }
            )

        if not transactions:
            # Ensure at least one transaction
            sale_date = date(2020, 6, 15)
            tx_id = hashlib.md5(f"{prop_id}_fallback".encode()).hexdigest()[:16]
            transactions.append(
                {
                    "date": sale_date.isoformat(),
                    "price": int(round(base_price * 2.0, -2)),
                    "transaction_id": tx_id,
                }
            )

        transactions.sort(key=lambda x: x["date"])
        last_tx = transactions[-1]

        prop = {
            "id": prop_id,
            "address": address,
            "postcode": postcode,
            "postcode_prefix": str(prefix),
            "district": district["name"],
            "property_type": PROPERTY_TYPE_NAMES[prop_type],
            "property_type_code": prop_type,
            "tenure": tenure,
            "energy_rating": energy_rating,
            "current_energy_efficiency": current_efficiency,
            "potential_energy_efficiency": potential_efficiency,
            "floor_area": floor_area,
            "bedrooms": bedrooms,
            "heated_rooms": heated_rooms,
            "co2_emissions_current": co2_current,
            "co2_emissions_potential": co2_potential,
            "construction_age_band": age_band,
            "last_sale_price": last_tx["price"],
            "last_sale_date": last_tx["date"],
        }

        self._properties[prop_id] = prop
        self._price_histories[prop_id] = transactions

    def search_properties(self, query: str) -> list[dict]:
        query_lower = query.lower().strip()
        results = []
        for prop in self._properties.values():
            if (
                query_lower in prop["postcode"].lower()
                or query_lower in prop["address"].lower()
                or query_lower in prop["district"].lower()
                or query_lower in prop["postcode_prefix"].lower()
            ):
                results.append(prop)
        return results[:10]

    def get_property(self, property_id: str) -> dict | None:
        return self._properties.get(property_id)

    def get_price_history(self, property_id: str) -> list[dict]:
        return self._price_histories.get(property_id, [])

    def get_area_comparison(self, property_id: str) -> dict:
        prop = self._properties.get(property_id)
        if not prop:
            return {}

        district_name = prop["district"]
        hpi = self._hpi_series.get(district_name, [])
        transactions = self._price_histories.get(property_id, [])

        return {
            "property_id": property_id,
            "district": district_name,
            "sale_dates": [t["date"] for t in transactions],
            "sale_prices": [t["price"] for t in transactions],
            "hpi_dates": [h["date"] for h in hpi],
            "hpi_average_prices": [h["average_price"] for h in hpi],
        }

    def get_hpi_regions(self) -> list[str]:
        return list(self._hpi_series.keys())

    def get_hpi_series(self, region: str) -> tuple[list[str], list[float]]:
        series = self._hpi_series.get(region, [])
        return (
            [entry["date"] for entry in series],
            [entry["average_price"] for entry in series],
        )

    def get_similar_properties(self, property_id: str, n: int = 8) -> list[dict]:
        prop = self._properties.get(property_id)
        if not prop:
            return []

        target_area = prop["floor_area"]
        target_district = prop["district"]

        candidates = []
        for pid, p in self._properties.items():
            if pid == property_id:
                continue
            if p["district"] != target_district:
                continue

            area = p["floor_area"]
            if area < target_area * 0.6 or area > target_area * 1.4:
                continue

            # Compute similarity score (1.0 = identical floor area)
            area_diff = abs(area - target_area) / target_area
            type_bonus = 0.2 if p["property_type_code"] == prop["property_type_code"] else 0.0
            similarity = max(0.0, 1.0 - area_diff) * 0.8 + type_bonus

            candidates.append(
                {
                    "id": pid,
                    "address": p["address"],
                    "postcode": p["postcode"],
                    "property_type": p["property_type"],
                    "floor_area": p["floor_area"],
                    "bedrooms": p["bedrooms"],
                    "energy_rating": p["energy_rating"] or "D",
                    "last_sale_price": p["last_sale_price"],
                    "last_sale_date": p["last_sale_date"],
                    "similarity_score": round(similarity, 3),
                }
            )

        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        return candidates[:n]
