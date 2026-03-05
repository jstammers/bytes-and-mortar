from dataclasses import dataclass


@dataclass(frozen=True)
class SensitivityConfig:
    """Configuration for a sensitivity analysis attribute."""

    attribute: str
    attribute_label: str
    values: list[float]
    labels: list[str]


SENSITIVITY_CONFIGS = [
    SensitivityConfig(
        attribute="bedrooms",
        attribute_label="Bedrooms",
        values=[1, 2, 3, 4, 5, 6],
        labels=["1 bed", "2 bed", "3 bed", "4 bed", "5 bed", "6 bed"],
    ),
    SensitivityConfig(
        attribute="floor_area",
        attribute_label="Floor Area (sqm)",
        values=[40, 60, 80, 100, 120, 150, 200, 250, 300, 350],
        labels=[
            "40 sqm",
            "60 sqm",
            "80 sqm",
            "100 sqm",
            "120 sqm",
            "150 sqm",
            "200 sqm",
            "250 sqm",
            "300 sqm",
            "350 sqm",
        ],
    ),
    SensitivityConfig(
        attribute="energy_efficiency_score",
        attribute_label="Energy Efficiency Score",
        values=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        labels=[
            "10 (G)",
            "20 (G)",
            "30 (F)",
            "40 (E)",
            "50 (E)",
            "60 (D)",
            "70 (C)",
            "80 (B)",
            "90 (B)",
            "100 (A)",
        ],
    ),
]
