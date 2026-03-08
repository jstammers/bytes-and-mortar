import { useState, useCallback, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { searchProperties } from '../api/client'
import type { PropertySearchResult } from '../api/types'

function useDebounce<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState<T>(value)
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delay)
    return () => clearTimeout(id)
  }, [value, delay])
  return debounced
}

interface PropertySearchProps {
  onResults: (results: PropertySearchResult[], query: string) => void
}

export default function PropertySearch({ onResults }: PropertySearchProps) {
  const [query, setQuery] = useState('')
  const [suggestions, setSuggestions] = useState<PropertySearchResult[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [loading, setLoading] = useState(false)
  const [activeIndex, setActiveIndex] = useState(-1)
  const navigate = useNavigate()
  const containerRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const debouncedQuery = useDebounce(query, 300)

  // Fetch suggestions as user types
  useEffect(() => {
    if (debouncedQuery.trim().length < 2) {
      setSuggestions([])
      setShowSuggestions(false)
      return
    }

    let cancelled = false
    searchProperties(debouncedQuery.trim())
      .then((results) => {
        if (!cancelled) {
          setSuggestions(results.slice(0, 6))
          setShowSuggestions(results.length > 0)
          setActiveIndex(-1)
        }
      })
      .catch(() => {
        if (!cancelled) setSuggestions([])
      })

    return () => {
      cancelled = true
    }
  }, [debouncedQuery])

  // Close suggestions on outside click
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setShowSuggestions(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  const handleSearch = useCallback(async () => {
    const q = query.trim()
    if (!q) return
    setShowSuggestions(false)
    setLoading(true)
    try {
      const results = await searchProperties(q)
      onResults(results, q)
    } finally {
      setLoading(false)
    }
  }, [query, onResults])

  const handleSelectSuggestion = (prop: PropertySearchResult) => {
    setShowSuggestions(false)
    navigate(`/property/${prop.id}`)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      if (activeIndex >= 0 && suggestions[activeIndex]) {
        handleSelectSuggestion(suggestions[activeIndex])
      } else {
        handleSearch()
      }
      return
    }
    if (e.key === 'Escape') {
      setShowSuggestions(false)
      return
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActiveIndex((i) => Math.min(i + 1, suggestions.length - 1))
      return
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveIndex((i) => Math.max(i - 1, -1))
    }
  }

  return (
    <div ref={containerRef} className="relative w-full">
      <div className="flex gap-2 shadow-sm">
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => suggestions.length > 0 && setShowSuggestions(true)}
          placeholder="Search by postcode or street..."
          className="flex-1 px-4 py-3 border border-gray-300 rounded-l-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          aria-autocomplete="list"
          aria-expanded={showSuggestions}
          aria-haspopup="listbox"
          role="combobox"
        />
        <button
          onClick={handleSearch}
          disabled={loading}
          className="px-6 py-3 bg-blue-600 text-white text-sm font-medium rounded-r-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? 'Searching…' : 'Search'}
        </button>
      </div>

      {showSuggestions && suggestions.length > 0 && (
        <ul
          role="listbox"
          className="absolute z-20 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg overflow-hidden"
        >
          {suggestions.map((prop, i) => (
            <li key={prop.id} role="option" aria-selected={i === activeIndex}>
              <button
                className={`w-full px-4 py-3 text-left border-b border-gray-100 last:border-b-0 transition-colors ${
                  i === activeIndex ? 'bg-blue-50' : 'hover:bg-gray-50'
                }`}
                onMouseDown={(e) => e.preventDefault()} // prevent input blur before click
                onClick={() => handleSelectSuggestion(prop)}
              >
                <div className="text-sm font-medium text-gray-900 truncate">{prop.address}</div>
                <div className="text-xs text-gray-500 mt-0.5">
                  {prop.postcode}
                  {prop.district ? ` · ${prop.district}` : ''}
                  {prop.property_type ? ` · ${prop.property_type}` : ''}
                  {prop.last_sale_price
                    ? ` · £${prop.last_sale_price.toLocaleString()}`
                    : ''}
                </div>
              </button>
            </li>
          ))}
          {suggestions.length >= 6 && (
            <li>
              <button
                className="w-full px-4 py-2.5 text-center text-xs text-blue-600 hover:bg-blue-50 transition-colors"
                onMouseDown={(e) => e.preventDefault()}
                onClick={handleSearch}
              >
                Show all results for "{query}"
              </button>
            </li>
          )}
        </ul>
      )}
    </div>
  )
}
