import type { PropertyDetail } from '../api/types'

interface Props {
  property: PropertyDetail
}

function energyRatingColor(rating: string | null): string {
  if (!rating) return 'bg-gray-100 text-gray-600'
  const r = rating.toUpperCase()
  if (r === 'A' || r === 'B') return 'bg-green-100 text-green-800'
  if (r === 'C' || r === 'D') return 'bg-yellow-100 text-yellow-800'
  if (r === 'E' || r === 'F') return 'bg-orange-100 text-orange-800'
  return 'bg-red-100 text-red-800'
}

interface RowProps {
  label: string
  value: React.ReactNode
}

function Row({ label, value }: RowProps) {
  return (
    <div className="flex justify-between items-start py-2 border-b border-gray-100 last:border-0">
      <span className="text-xs text-gray-500 font-medium flex-shrink-0 mr-2">{label}</span>
      <span className="text-xs text-gray-800 text-right">{value ?? '—'}</span>
    </div>
  )
}

export default function PropertyAttributes({ property }: Props) {
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      <h2 className="text-sm font-semibold text-gray-700 mb-3">Property Details</h2>

      <Row label="Type" value={property.property_type} />
      <Row label="Tenure" value={property.tenure} />
      <Row
        label="Floor Area"
        value={property.floor_area ? `${property.floor_area} sqm` : null}
      />
      <Row
        label="Bedrooms"
        value={property.bedrooms}
      />
      <Row
        label="Heated Rooms"
        value={property.heated_rooms}
      />
      <Row
        label="Energy Rating"
        value={
          property.energy_rating ? (
            <span
              className={`inline-block px-2 py-0.5 rounded text-xs font-bold ${energyRatingColor(property.energy_rating)}`}
            >
              {property.energy_rating}
            </span>
          ) : null
        }
      />
      <Row
        label="Current EPC Score"
        value={property.current_energy_efficiency}
      />
      <Row
        label="Potential EPC Score"
        value={property.potential_energy_efficiency}
      />
      <Row
        label="CO2 Current"
        value={property.co2_emissions_current != null ? `${property.co2_emissions_current} t/year` : null}
      />
      <Row
        label="CO2 Potential"
        value={property.co2_emissions_potential != null ? `${property.co2_emissions_potential} t/year` : null}
      />
      <Row
        label="Age Band"
        value={property.construction_age_band}
      />
    </div>
  )
}
