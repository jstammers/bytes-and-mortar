import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { PropertyDetail, SimilarProperty } from '../../api/types'

interface Props {
  similar: SimilarProperty[]
  currentProperty: PropertyDetail
}

function formatPrice(value: number): string {
  if (value >= 1_000_000) return `£${(value / 1_000_000).toFixed(1)}m`
  if (value >= 1_000) return `£${Math.round(value / 1_000)}k`
  return `£${value}`
}

function energyRatingColor(rating: string): string {
  const r = rating.toUpperCase()
  if (r === 'A' || r === 'B') return '#16a34a'
  if (r === 'C') return '#84cc16'
  if (r === 'D') return '#eab308'
  if (r === 'E' || r === 'F') return '#f97316'
  return '#dc2626'
}

interface TooltipProps {
  active?: boolean
  payload?: Array<{ payload: { address: string; last_sale_price: number; floor_area: number; energy_rating: string } }>
}

function CustomTooltip({ active, payload }: TooltipProps) {
  if (!active || !payload || payload.length === 0) return null
  const d = payload[0].payload
  return (
    <div className="bg-white border border-gray-200 rounded shadow-lg p-3 text-xs max-w-[200px]">
      <p className="font-semibold text-gray-800 mb-1">{d.address}</p>
      <p className="text-gray-600">{formatPrice(d.last_sale_price)}</p>
      <p className="text-gray-500">{d.floor_area} sqm</p>
      <p className="text-gray-500">EPC {d.energy_rating}</p>
    </div>
  )
}

export default function SimilarPropertiesChart({ similar, currentProperty }: Props) {
  // Only include similar properties that have floor_area and last_sale_price for the chart
  const plottable = similar.filter(
    (s): s is SimilarProperty & { floor_area: number } => s.floor_area != null && s.last_sale_price > 0
  )

  // Group by energy rating for colour coding; fall back to 'Unknown' when null
  const ratingGroups: Record<string, typeof plottable> = {}
  for (const s of plottable) {
    const r = s.energy_rating ?? 'Unknown'
    if (!ratingGroups[r]) ratingGroups[r] = []
    ratingGroups[r].push(s)
  }

  const currentPoint = currentProperty.floor_area != null && currentProperty.last_sale_price > 0
    ? {
        floor_area: currentProperty.floor_area,
        last_sale_price: currentProperty.last_sale_price,
        address: currentProperty.address,
        energy_rating: currentProperty.energy_rating ?? 'Unknown',
      }
    : null

  if (plottable.length === 0 && !currentPoint) {
    return (
      <p className="text-xs text-gray-400 italic py-4 text-center">
        Floor area data unavailable — chart cannot be shown.
      </p>
    )
  }

  return (
    <div className="w-full h-72">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            type="number"
            dataKey="floor_area"
            name="Floor Area"
            unit=" sqm"
            tick={{ fontSize: 11, fill: '#6b7280' }}
            tickLine={false}
            axisLine={{ stroke: '#e5e7eb' }}
            label={{ value: 'Floor Area (sqm)', position: 'insideBottom', offset: -5, fontSize: 11, fill: '#9ca3af' }}
          />
          <YAxis
            type="number"
            dataKey="last_sale_price"
            name="Price"
            tickFormatter={formatPrice}
            tick={{ fontSize: 11, fill: '#6b7280' }}
            tickLine={false}
            axisLine={false}
            width={70}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          {Object.entries(ratingGroups).sort().map(([rating, props]) => (
            <Scatter
              key={rating}
              name={rating === 'Unknown' ? 'No EPC' : `EPC ${rating}`}
              data={props}
              fill={rating === 'Unknown' ? '#9ca3af' : energyRatingColor(rating)}
              opacity={0.75}
            />
          ))}
          {currentPoint && (
            <Scatter
              name="This Property"
              data={[currentPoint]}
              fill="#2563eb"
              shape="star"
            />
          )}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}
