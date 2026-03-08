export interface PropertySearchResult {
  id: string
  address: string
  postcode: string
  district: string
  property_type: string
  property_type_code: string
  tenure: string
  last_sale_price: number
  last_sale_date: string
  energy_rating: string | null
  floor_area: number | null
  bedrooms: number | null
}

export interface PropertyDetail {
  id: string
  address: string
  postcode: string
  district: string
  property_type: string
  property_type_code: string
  tenure: string
  last_sale_price: number
  last_sale_date: string
  energy_rating: string | null
  current_energy_efficiency: number | null
  potential_energy_efficiency: number | null
  floor_area: number | null
  bedrooms: number | null
  heated_rooms: number | null
  co2_emissions_current: number | null
  co2_emissions_potential: number | null
  construction_age_band: string | null
}

export interface PriceHistoryPoint {
  date: string
  price: number
  transaction_id: string
}

export interface PropertyHistory {
  property_id: string
  transactions: PriceHistoryPoint[]
}

export interface AreaComparisonData {
  property_id: string
  district: string
  sale_dates: string[]
  sale_prices: number[]
  hpi_dates: string[]
  hpi_average_prices: number[]
}

export interface SimilarProperty {
  id: string
  address: string
  postcode: string
  property_type: string
  floor_area: number | null
  bedrooms: number | null
  energy_rating: string | null
  last_sale_price: number
  last_sale_date: string
  similarity_score: number
}

export interface ValuationInput {
  property_type: string
  bedrooms: number
  floor_area: number
  energy_efficiency_score: number
  postcode_prefix: string
}

export interface ValuationPrediction {
  estimated_price: number
  confidence_lower: number
  confidence_upper: number
}

export interface SensitivityPoint {
  value: number
  price: number
  label: string
}

export interface SensitivityResult {
  attribute: string
  attribute_label: string
  current_value: number
  points: SensitivityPoint[]
}

export interface ValuationSensitivityResponse {
  base_prediction: ValuationPrediction
  sensitivities: SensitivityResult[]
}
