import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  getProperty,
  getPriceHistory,
  getAreaComparison,
  getSimilarProperties,
} from '../api/client'
import type {
  AreaComparisonData,
  PropertyDetail as PropertyDetailType,
  PropertyHistory,
  SimilarProperty,
} from '../api/types'
import PropertyAttributes from '../components/PropertyAttributes'
import PriceHistoryChart from '../components/charts/PriceHistoryChart'
import AreaComparisonChart from '../components/charts/AreaComparisonChart'
import SimilarPropertiesChart from '../components/charts/SimilarPropertiesChart'

type TabId = 'history' | 'area' | 'similar'

function energyRatingColor(rating: string | null): string {
  if (!rating) return 'bg-gray-100 text-gray-600'
  const r = rating.toUpperCase()
  if (r === 'A' || r === 'B') return 'bg-green-100 text-green-800'
  if (r === 'C' || r === 'D') return 'bg-yellow-100 text-yellow-800'
  if (r === 'E' || r === 'F') return 'bg-orange-100 text-orange-800'
  return 'bg-red-100 text-red-800'
}

function formatPrice(price: number): string {
  return '£' + price.toLocaleString('en-GB')
}

export default function PropertyDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  const [property, setProperty] = useState<PropertyDetailType | null>(null)
  const [history, setHistory] = useState<PropertyHistory | null>(null)
  const [areaData, setAreaData] = useState<AreaComparisonData | null>(null)
  const [similar, setSimilar] = useState<SimilarProperty[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<TabId>('history')

  useEffect(() => {
    if (!id) return
    setLoading(true)
    setError(null)
    Promise.all([
      getProperty(id),
      getPriceHistory(id),
      getAreaComparison(id),
      getSimilarProperties(id),
    ])
      .then(([prop, hist, area, sim]) => {
        setProperty(prop)
        setHistory(hist)
        setAreaData(area)
        setSimilar(sim)
      })
      .catch((e: unknown) => {
        setError(e instanceof Error ? e.message : 'Failed to load property')
      })
      .finally(() => setLoading(false))
  }, [id])

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-10 text-center text-gray-400">
        <div className="animate-pulse">Loading property data...</div>
      </div>
    )
  }

  if (error || !property) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-10 text-center">
        <p className="text-red-600 mb-4">{error ?? 'Property not found'}</p>
        <button onClick={() => navigate('/')} className="text-blue-600 hover:underline text-sm">
          Back to search
        </button>
      </div>
    )
  }

  const tabs: { id: TabId; label: string }[] = [
    { id: 'history', label: 'Price History' },
    { id: 'area', label: 'Area Comparison' },
    { id: 'similar', label: 'Similar Properties' },
  ]

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <button
        onClick={() => navigate(-1)}
        className="text-sm text-blue-600 hover:underline mb-6 flex items-center gap-1"
      >
        &larr; Back
      </button>

      {/* Header */}
      <div className="mb-6">
        <div className="flex flex-wrap items-start gap-3 mb-2">
          <h1 className="text-2xl font-bold text-gray-900">{property.address}</h1>
          <span className="inline-block px-3 py-1 rounded-full text-xs font-semibold bg-slate-100 text-slate-700">
            {property.property_type}
          </span>
          {property.energy_rating && (
            <span
              className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${energyRatingColor(property.energy_rating)}`}
            >
              EPC {property.energy_rating}
            </span>
          )}
        </div>
        <p className="text-gray-500 text-sm mb-1">
          {property.postcode} &middot; {property.district}
        </p>
        {history && history.transactions.length > 0 && (
          <p className="text-xl font-bold text-gray-800 mt-2">
            {formatPrice(history.transactions[history.transactions.length - 1].price)}
            <span className="text-sm font-normal text-gray-400 ml-2">last sale</span>
          </p>
        )}
      </div>

      {/* Main content + sidebar */}
      <div className="flex gap-6 flex-col lg:flex-row">
        {/* Main content */}
        <div className="flex-1 min-w-0">
          {/* Tabs */}
          <div className="flex border-b border-gray-200 mb-4 gap-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-px ${
                  activeTab === tab.id
                    ? 'border-blue-600 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            {activeTab === 'history' && history && (
              <div>
                <h2 className="text-sm font-semibold text-gray-700 mb-4">Transaction History</h2>
                {history.transactions.length === 0 ? (
                  <p className="text-gray-400 text-sm">No transaction history available.</p>
                ) : (
                  <>
                    <PriceHistoryChart transactions={history.transactions} />
                    <div className="mt-4">
                      <table className="w-full text-xs text-gray-600">
                        <thead>
                          <tr className="border-b border-gray-100">
                            <th className="text-left py-2 font-medium text-gray-500">Date</th>
                            <th className="text-right py-2 font-medium text-gray-500">Price</th>
                          </tr>
                        </thead>
                        <tbody>
                          {[...history.transactions].reverse().map((tx) => (
                            <tr key={tx.transaction_id} className="border-b border-gray-50">
                              <td className="py-2">{new Date(tx.date).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })}</td>
                              <td className="py-2 text-right font-semibold">{formatPrice(tx.price)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}
              </div>
            )}

            {activeTab === 'area' && areaData && (
              <div>
                <h2 className="text-sm font-semibold text-gray-700 mb-1">
                  Area Comparison: {areaData.district}
                </h2>
                <p className="text-xs text-gray-400 mb-4">
                  This property's sales vs. district average price (Jan 2000 – Dec 2024)
                </p>
                <AreaComparisonChart data={areaData} />
              </div>
            )}

            {activeTab === 'similar' && (
              <div>
                <h2 className="text-sm font-semibold text-gray-700 mb-1">Similar Properties</h2>
                <p className="text-xs text-gray-400 mb-4">
                  Properties in the same district with similar floor area
                </p>
                {similar.length === 0 ? (
                  <p className="text-gray-400 text-sm">No similar properties found.</p>
                ) : (
                  <>
                    <SimilarPropertiesChart similar={similar} currentProperty={property} />
                    <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-2">
                      {similar.map((s) => (
                        <div
                          key={s.id}
                          className="p-3 rounded border border-gray-100 hover:border-blue-200 hover:bg-blue-50 cursor-pointer transition-colors"
                          onClick={() => navigate(`/property/${s.id}`)}
                        >
                          <p className="text-xs font-semibold text-gray-800">{s.address}</p>
                          <p className="text-xs text-gray-400">{s.postcode}</p>
                          <div className="flex justify-between mt-1">
                            <span className="text-xs text-gray-500">{s.floor_area} sqm &middot; {s.bedrooms} bed</span>
                            <span className="text-xs font-semibold text-gray-800">{formatPrice(s.last_sale_price)}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Sidebar */}
        <div className="lg:w-72 flex-shrink-0">
          <PropertyAttributes property={property} />
        </div>
      </div>
    </div>
  )
}
