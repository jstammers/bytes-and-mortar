import type {
  AreaComparisonData,
  PropertyDetail,
  PropertyHistory,
  PropertySearchResult,
  SimilarProperty,
  ValuationInput,
  ValuationPrediction,
  ValuationSensitivityResponse,
} from './types'

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options)
  if (!response.ok) {
    const text = await response.text()
    throw new Error(`API error ${response.status}: ${text}`)
  }
  return response.json() as Promise<T>
}

export async function searchProperties(query: string): Promise<PropertySearchResult[]> {
  return fetchJson<PropertySearchResult[]>(
    `/api/properties/search?q=${encodeURIComponent(query)}`
  )
}

export async function getProperty(id: string): Promise<PropertyDetail> {
  return fetchJson<PropertyDetail>(`/api/properties/${id}`)
}

export async function getPriceHistory(id: string): Promise<PropertyHistory> {
  return fetchJson<PropertyHistory>(`/api/properties/${id}/history`)
}

export async function getAreaComparison(id: string): Promise<AreaComparisonData> {
  return fetchJson<AreaComparisonData>(`/api/properties/${id}/area-comparison`)
}

export async function getSimilarProperties(
  id: string,
  n: number = 8
): Promise<SimilarProperty[]> {
  return fetchJson<SimilarProperty[]>(`/api/properties/${id}/similar?n=${n}`)
}

export async function predictValuation(input: ValuationInput): Promise<ValuationPrediction> {
  return fetchJson<ValuationPrediction>('/api/valuation/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  })
}

export async function getValuationSensitivity(
  input: ValuationInput
): Promise<ValuationSensitivityResponse> {
  return fetchJson<ValuationSensitivityResponse>('/api/valuation/sensitivity', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  })
}
