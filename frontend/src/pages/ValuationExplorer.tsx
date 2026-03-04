import { useState, useEffect, useRef, useCallback } from 'react'
import { getValuationSensitivity } from '../api/client'
import type { ValuationInput, ValuationSensitivityResponse } from '../api/types'
import ValuationSensitivityChart from '../components/charts/ValuationSensitivityChart'

const POSTCODE_PREFIXES = [
  'W1A', 'W1B', 'SW1A', 'SW1V',
  'E1', 'E14', 'E3',
  'LS1', 'LS2', 'LS7', 'LS16',
  'M1', 'M4', 'M14', 'M20',
  'B1', 'B15', 'B16', 'B29',
].sort()

const PROPERTY_TYPES = [
  { code: 'F', label: 'Flat/Maisonette' },
  { code: 'T', label: 'Terraced' },
  { code: 'S', label: 'Semi-Detached' },
  { code: 'D', label: 'Detached' },
]

function energyScoreToRating(score: number): string {
  if (score >= 92) return 'A'
  if (score >= 81) return 'B'
  if (score >= 69) return 'C'
  if (score >= 55) return 'D'
  if (score >= 39) return 'E'
  if (score >= 21) return 'F'
  return 'G'
}

function ratingColor(rating: string): string {
  if (rating === 'A' || rating === 'B') return 'text-green-700 bg-green-100'
  if (rating === 'C' || rating === 'D') return 'text-yellow-700 bg-yellow-100'
  if (rating === 'E' || rating === 'F') return 'text-orange-700 bg-orange-100'
  return 'text-red-700 bg-red-100'
}

function formatPrice(price: number): string {
  return '£' + price.toLocaleString('en-GB')
}

interface SliderProps {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (v: number) => void
  displayValue?: React.ReactNode
}

function Slider({ label, value, min, max, step = 1, onChange, displayValue }: SliderProps) {
  return (
    <div className="mb-5">
      <div className="flex justify-between items-center mb-1">
        <label className="text-xs font-medium text-gray-700">{label}</label>
        <span className="text-xs font-semibold text-blue-700">{displayValue ?? value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
      />
      <div className="flex justify-between text-xs text-gray-300 mt-0.5">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  )
}

export default function ValuationExplorer() {
  const [propertyType, setPropertyType] = useState('F')
  const [bedrooms, setBedrooms] = useState(2)
  const [floorArea, setFloorArea] = useState(80)
  const [energyScore, setEnergyScore] = useState(65)
  const [postcode, setPostcode] = useState('E14')

  const [result, setResult] = useState<ValuationSensitivityResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const debounceTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const fetchValuation = useCallback(async (input: ValuationInput) => {
    setLoading(true)
    setError(null)
    try {
      const data = await getValuationSensitivity(input)
      setResult(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Valuation failed')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    const input: ValuationInput = {
      property_type: propertyType,
      bedrooms,
      floor_area: floorArea,
      energy_efficiency_score: energyScore,
      postcode_prefix: postcode,
    }
    if (debounceTimer.current) clearTimeout(debounceTimer.current)
    debounceTimer.current = setTimeout(() => {
      fetchValuation(input)
    }, 300)
    return () => {
      if (debounceTimer.current) clearTimeout(debounceTimer.current)
    }
  }, [propertyType, bedrooms, floorArea, energyScore, postcode, fetchValuation])

  const rating = energyScoreToRating(energyScore)

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <h1 className="text-2xl font-bold text-gray-900 mb-2">Valuation Explorer</h1>
      <p className="text-sm text-gray-500 mb-8">
        Adjust property attributes to explore estimated market values in real time.
      </p>

      <div className="flex gap-6 flex-col lg:flex-row">
        {/* Controls */}
        <div className="lg:w-80 flex-shrink-0">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-5">
            <h2 className="text-sm font-semibold text-gray-700 mb-4">Property Attributes</h2>

            {/* Property Type */}
            <div className="mb-5">
              <label className="text-xs font-medium text-gray-700 block mb-2">Property Type</label>
              <div className="grid grid-cols-2 gap-1.5">
                {PROPERTY_TYPES.map((t) => (
                  <button
                    key={t.code}
                    onClick={() => setPropertyType(t.code)}
                    className={`text-xs px-3 py-2 rounded border transition-colors ${
                      propertyType === t.code
                        ? 'bg-blue-600 text-white border-blue-600'
                        : 'bg-white text-gray-700 border-gray-200 hover:border-blue-300'
                    }`}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Location */}
            <div className="mb-5">
              <label className="text-xs font-medium text-gray-700 block mb-1">Location (Postcode Prefix)</label>
              <select
                value={postcode}
                onChange={(e) => setPostcode(e.target.value)}
                className="w-full text-xs border border-gray-200 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {POSTCODE_PREFIXES.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </div>

            <Slider
              label="Bedrooms"
              value={bedrooms}
              min={1}
              max={6}
              onChange={setBedrooms}
              displayValue={`${bedrooms} bed`}
            />

            <Slider
              label="Floor Area"
              value={floorArea}
              min={40}
              max={350}
              step={5}
              onChange={setFloorArea}
              displayValue={`${floorArea} sqm`}
            />

            <Slider
              label="Energy Efficiency Score"
              value={energyScore}
              min={1}
              max={100}
              onChange={setEnergyScore}
              displayValue={
                <span className={`px-1.5 py-0.5 rounded text-xs font-bold ${ratingColor(rating)}`}>
                  {energyScore} ({rating})
                </span>
              }
            />
          </div>
        </div>

        {/* Results */}
        <div className="flex-1 min-w-0">
          {/* Price display */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-4">
            {loading && (
              <div className="text-center text-gray-400 py-4 animate-pulse text-sm">
                Calculating...
              </div>
            )}
            {error && (
              <div className="text-center text-red-500 py-4 text-sm">{error}</div>
            )}
            {!loading && !error && result && (
              <>
                <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">Estimated Value</p>
                <p className="text-4xl font-bold text-gray-900 mb-2">
                  {formatPrice(result.base_prediction.estimated_price)}
                </p>
                <p className="text-sm text-gray-500">
                  Confidence interval:{' '}
                  <span className="font-medium text-gray-700">
                    {formatPrice(result.base_prediction.confidence_lower)}
                  </span>{' '}
                  &ndash;{' '}
                  <span className="font-medium text-gray-700">
                    {formatPrice(result.base_prediction.confidence_upper)}
                  </span>
                </p>
                <div className="mt-3 flex flex-wrap gap-2 text-xs text-gray-500">
                  <span className="bg-gray-50 px-2 py-1 rounded border border-gray-100">
                    {PROPERTY_TYPES.find((t) => t.code === propertyType)?.label}
                  </span>
                  <span className="bg-gray-50 px-2 py-1 rounded border border-gray-100">{postcode}</span>
                  <span className="bg-gray-50 px-2 py-1 rounded border border-gray-100">{bedrooms} bed</span>
                  <span className="bg-gray-50 px-2 py-1 rounded border border-gray-100">{floorArea} sqm</span>
                  <span className={`px-2 py-1 rounded border ${ratingColor(rating)}`}>EPC {rating}</span>
                </div>
              </>
            )}
          </div>

          {/* Sensitivity charts */}
          {!loading && !error && result && result.sensitivities.length > 0 && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-5">
              <h2 className="text-sm font-semibold text-gray-700 mb-1">Sensitivity Analysis</h2>
              <p className="text-xs text-gray-400 mb-4">
                How the estimated price changes as each attribute varies (dashed line = current value)
              </p>
              <ValuationSensitivityChart sensitivities={result.sensitivities} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
