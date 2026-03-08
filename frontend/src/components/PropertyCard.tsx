import { useNavigate } from 'react-router-dom'
import type { PropertySearchResult } from '../api/types'

interface Props {
  property: PropertySearchResult
}

function energyRatingColor(rating: string | null): string {
  if (!rating) return 'bg-gray-100 text-gray-700'
  const r = rating.toUpperCase()
  if (r === 'A' || r === 'B') return 'bg-green-100 text-green-800'
  if (r === 'C' || r === 'D') return 'bg-yellow-100 text-yellow-800'
  if (r === 'E' || r === 'F') return 'bg-orange-100 text-orange-800'
  return 'bg-red-100 text-red-800'
}

function propertyTypeIcon(code: string): string {
  switch (code) {
    case 'D': return 'D'
    case 'S': return 'S'
    case 'T': return 'T'
    case 'F': return 'F'
    default: return '?'
  }
}

function formatPrice(price: number): string {
  if (!price) return 'Price unavailable'
  return '£' + price.toLocaleString('en-GB')
}

function formatDate(dateStr: string): string {
  if (!dateStr) return 'Unknown date'
  const d = new Date(dateStr)
  if (isNaN(d.getTime())) return 'Unknown date'
  return d.toLocaleDateString('en-GB', { month: 'short', year: 'numeric' })
}

export default function PropertyCard({ property }: Props) {
  const navigate = useNavigate()

  return (
    <div
      className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 cursor-pointer hover:shadow-md hover:border-blue-300 transition-all"
      onClick={() => navigate(`/property/${property.id}`)}
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <h3 className="font-semibold text-gray-900 text-sm leading-tight">{property.address}</h3>
        <span className="flex-shrink-0 inline-flex items-center justify-center w-7 h-7 rounded bg-slate-100 text-slate-700 text-xs font-bold">
          {propertyTypeIcon(property.property_type_code)}
        </span>
      </div>

      <p className="text-xs text-gray-500 mb-2">
        {property.postcode} &middot; {property.district}
      </p>

      <p className="text-xs text-gray-600 mb-3">
        {property.property_type}
        {property.tenure && (
          <span className="text-gray-400"> &middot; {property.tenure}</span>
        )}
      </p>

      <div className="flex items-center gap-2 mb-3 flex-wrap">
        {property.energy_rating && (
          <span className={`text-xs font-semibold px-2 py-0.5 rounded ${energyRatingColor(property.energy_rating)}`}>
            EPC {property.energy_rating}
          </span>
        )}
        {property.floor_area && (
          <span className="text-xs text-gray-500">{property.floor_area} sqm</span>
        )}
        {property.bedrooms && (
          <span className="text-xs text-gray-500">{property.bedrooms} bed</span>
        )}
        {!property.energy_rating && !property.floor_area && !property.bedrooms && (
          <span className="text-xs text-gray-400 italic">No EPC data</span>
        )}
      </div>

      <div className="border-t border-gray-100 pt-3">
        <p className="text-lg font-bold text-gray-900">{formatPrice(property.last_sale_price)}</p>
        <p className="text-xs text-gray-400 mt-0.5">Sold {formatDate(property.last_sale_date)}</p>
      </div>
    </div>
  )
}
