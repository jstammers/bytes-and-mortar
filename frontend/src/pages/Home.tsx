import { useState, useCallback } from 'react'
import PropertyCard from '../components/PropertyCard'
import PropertySearch from '../components/PropertySearch'
import type { PropertySearchResult } from '../api/types'

export default function Home() {
  const [results, setResults] = useState<PropertySearchResult[]>([])
  const [searched, setSearched] = useState(false)
  const [lastQuery, setLastQuery] = useState('')

  const handleResults = useCallback((data: PropertySearchResult[], query: string) => {
    setResults(data)
    setLastQuery(query)
    setSearched(true)
  }, [])

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      <div className="max-w-2xl mx-auto text-center mb-10">
        <h1 className="text-3xl font-bold text-gray-900 mb-3">UK Property Insights</h1>
        <p className="text-gray-500 mb-6 text-sm">
          Search for any UK property by postcode or address to view price history, area comparisons, and valuations.
        </p>

        <PropertySearch onResults={handleResults} />
      </div>

      {!searched && (
        <div className="text-center text-gray-400 mt-16">
          <div className="text-5xl mb-4">🏠</div>
          <p className="text-sm">Enter a postcode or address above to find properties.</p>
          <p className="text-xs mt-2 text-gray-300">Try: SW1A 1AA or Church Lane, Westminster</p>
        </div>
      )}

      {searched && results.length === 0 && (
        <div className="text-center text-gray-400 mt-16">
          <p className="text-sm">No results found for "{lastQuery}".</p>
          <p className="text-xs mt-1 text-gray-300">Try a different postcode or address.</p>
        </div>
      )}

      {results.length > 0 && (
        <div>
          <p className="text-xs text-gray-400 mb-4">
            {results.length} result{results.length !== 1 ? 's' : ''} for "{lastQuery}"
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {results.map((property) => (
              <PropertyCard key={property.id} property={property} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
