import { useState, useCallback } from 'react'
import PropertyCard from '../components/PropertyCard'
import { searchProperties } from '../api/client'
import type { PropertySearchResult } from '../api/types'

export default function Home() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<PropertySearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [searched, setSearched] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSearch = useCallback(async () => {
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    setSearched(true)
    try {
      const data = await searchProperties(query.trim())
      setResults(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Search failed')
      setResults([])
    } finally {
      setLoading(false)
    }
  }, [query])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') handleSearch()
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      <div className="max-w-2xl mx-auto text-center mb-10">
        <h1 className="text-3xl font-bold text-gray-900 mb-3">UK Property Insights</h1>
        <p className="text-gray-500 mb-6 text-sm">
          Search for any UK property by postcode or address to view price history, area comparisons, and valuations.
        </p>

        <div className="flex gap-2 shadow-sm">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search by postcode (e.g. SW1A) or address..."
            className="flex-1 px-4 py-3 border border-gray-300 rounded-l-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <button
            onClick={handleSearch}
            disabled={loading}
            className="px-6 py-3 bg-blue-600 text-white text-sm font-medium rounded-r-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>
      </div>

      {error && (
        <div className="max-w-2xl mx-auto mb-6 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
          {error}
        </div>
      )}

      {!searched && !loading && (
        <div className="text-center text-gray-400 mt-16">
          <div className="text-5xl mb-4">🏠</div>
          <p className="text-sm">Enter a postcode or address above to find properties.</p>
          <p className="text-xs mt-2 text-gray-300">Try: SW1A, Leeds, E14, Manchester</p>
        </div>
      )}

      {searched && !loading && results.length === 0 && !error && (
        <div className="text-center text-gray-400 mt-16">
          <p className="text-sm">No results found for "{query}".</p>
          <p className="text-xs mt-1 text-gray-300">Try a different postcode or address.</p>
        </div>
      )}

      {results.length > 0 && (
        <div>
          <p className="text-xs text-gray-400 mb-4">
            {results.length} result{results.length !== 1 ? 's' : ''} for "{query}"
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
