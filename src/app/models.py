from pydantic import BaseModel


class PropertySearchResult(BaseModel):
    id: str
    address: str
    postcode: str
    district: str
    property_type: str  # full name e.g. "Flat/Maisonette"
    property_type_code: str  # D/S/T/F
    tenure: str  # Freehold / Leasehold
    last_sale_price: int
    last_sale_date: str
    energy_rating: str | None
    floor_area: float | None
    bedrooms: int | None


class PropertyDetail(BaseModel):
    id: str
    address: str
    postcode: str
    district: str
    property_type: str
    property_type_code: str
    tenure: str
    last_sale_price: int
    last_sale_date: str
    energy_rating: str | None
    current_energy_efficiency: int | None
    potential_energy_efficiency: int | None
    floor_area: float | None
    bedrooms: int | None
    heated_rooms: int | None
    co2_emissions_current: float | None
    co2_emissions_potential: float | None
    construction_age_band: str | None


class PriceHistoryPoint(BaseModel):
    date: str
    price: int
    transaction_id: str


class PropertyHistory(BaseModel):
    property_id: str
    transactions: list[PriceHistoryPoint]


class AreaComparisonData(BaseModel):
    property_id: str
    district: str
    sale_dates: list[str]
    sale_prices: list[int]
    hpi_dates: list[str]
    hpi_average_prices: list[float]


class SimilarProperty(BaseModel):
    id: str
    address: str
    postcode: str
    property_type: str
    floor_area: float | None
    bedrooms: int | None
    energy_rating: str | None
    last_sale_price: int
    last_sale_date: str
    similarity_score: float


class ValuationInput(BaseModel):
    property_type: str  # D/S/T/F
    bedrooms: int
    floor_area: float
    energy_efficiency_score: int  # 1-100
    postcode_prefix: str  # e.g. "SW1A", "LS1"


class ValuationPrediction(BaseModel):
    estimated_price: int
    confidence_lower: int
    confidence_upper: int


class SensitivityPoint(BaseModel):
    value: float
    price: int
    label: str


class SensitivityResult(BaseModel):
    attribute: str
    attribute_label: str
    current_value: float
    points: list[SensitivityPoint]


class ValuationSensitivityResponse(BaseModel):
    base_prediction: ValuationPrediction
    sensitivities: list[SensitivityResult]


class HPIForecastRequest(BaseModel):
    region: str
    method: str = "ensemble"  # "ets" | "sarima" | "trend" | "ensemble"
    steps: int = 12  # 1-60
    alpha: float = 0.05  # significance level; 1-alpha = PI coverage


class HPIHistoricalPoint(BaseModel):
    date: str
    value: float


class HPIForecastPoint(BaseModel):
    date: str
    point: float
    lower: float
    upper: float


class HPIForecastResponse(BaseModel):
    region: str
    method: str
    alpha: float
    historical: list[HPIHistoricalPoint]
    forecast: list[HPIForecastPoint]
