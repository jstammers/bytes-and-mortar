import { useState, useEffect, useCallback, useRef } from 'react'
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { getHPIForecastRegions, getHPIForecast } from '../api/client'
import type { HPIForecastResponse } from '../api/types'

// ---------------------------------------------------------------------------
// Types for chart data
// ---------------------------------------------------------------------------

interface ChartPoint {
  date: string
  historical?: number
  point?: number
  band?: [number, number]  // [lower, upper] for area chart
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const METHODS = [
  { id: 'ensemble', label: 'Ensemble', description: 'Weighted combination — most reliable' },
  { id: 'ets', label: 'ETS', description: 'Exponential smoothing — best univariate baseline' },
  { id: 'sarima', label: 'SARIMA', description: 'Seasonal ARIMA — handles seasonality explicitly' },
  { id: 'trend', label: 'Trend', description: 'Polynomial trend + bootstrap intervals' },
]

const ALPHA_OPTIONS = [
  { value: 0.05, label: '95% interval' },
  { value: 0.10, label: '90% interval' },
  { value: 0.20, label: '80% interval' },
]

const STEP_OPTIONS = [6, 12, 18, 24]

function formatPrice(v: number): string {
  if (v >= 1_000_000) return `£${(v / 1_000_000).toFixed(2)}m`
  if (v >= 1_000) return `£${(v / 1_000).toFixed(0)}k`
  return `£${v.toFixed(0)}`
}

function formatDateTick(dateStr: string): string {
  const d = new Date(dateStr)
  return d.toLocaleDateString('en-GB', { month: 'short', year: '2-digit' })
}

// ---------------------------------------------------------------------------
// Custom tooltip
// ---------------------------------------------------------------------------

interface TooltipPayload {
  name: string
  value: number | [number, number]
  color: string
}

interface CustomTooltipProps {
  active?: boolean
  payload?: TooltipPayload[]
  label?: string
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload || !label) return null

  const d = new Date(label)
  const dateLabel = d.toLocaleDateString('en-GB', { month: 'long', year: 'numeric' })

  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-lg p-3 text-sm">
      <p className="font-semibold text-gray-800 mb-2">{dateLabel}</p>
      {payload.map((entry) => {
        if (entry.name === 'Prediction interval') {
          const [lo, hi] = entry.value as [number, number]
          return (
            <p key="band" className="text-blue-500">
              PI: {formatPrice(lo)} – {formatPrice(hi)}
            </p>
          )
        }
        if (entry.name === 'Historical') {
          return (
            <p key="hist" className="text-gray-700">
              Historical: {formatPrice(entry.value as number)}
            </p>
          )
        }
        if (entry.name === 'Forecast') {
          return (
            <p key="fc" className="text-blue-700 font-medium">
              Forecast: {formatPrice(entry.value as number)}
            </p>
          )
        }
        return null
      })}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function HPIForecast() {
  const [regions, setRegions] = useState<string[]>([])
  const [region, setRegion] = useState<string>('')
  const [method, setMethod] = useState<string>('ensemble')
  const [steps, setSteps] = useState<number>(12)
  const [alpha, setAlpha] = useState<number>(0.05)

  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<HPIForecastResponse | null>(null)

  const abortRef = useRef<AbortController | null>(null)

  // Load regions on mount
  useEffect(() => {
    getHPIForecastRegions()
      .then((r) => {
        setRegions(r)
        if (r.length > 0) setRegion(r[0])
      })
      .catch(() => setError('Failed to load regions'))
  }, [])

  const runForecast = useCallback(() => {
    if (!region) return
    abortRef.current?.abort()
    abortRef.current = new AbortController()

    setLoading(true)
    setError(null)

    getHPIForecast({ region, method, steps, alpha })
      .then((r) => {
        setResult(r)
        setLoading(false)
      })
      .catch((err: Error) => {
        if (err.name !== 'AbortError') {
          setError(err.message || 'Forecast failed')
          setLoading(false)
        }
      })
  }, [region, method, steps, alpha])

  // Run forecast whenever inputs change (after initial region load)
  useEffect(() => {
    if (region) runForecast()
  }, [region, method, steps, alpha, runForecast])

  // Build chart data from result
  const chartData: ChartPoint[] = (() => {
    if (!result) return []
    const points: ChartPoint[] = result.historical.map((h) => ({
      date: h.date,
      historical: h.value,
    }))
    // Add a bridge point so lines connect cleanly
    const lastHist = result.historical[result.historical.length - 1]
    if (lastHist && result.forecast.length > 0) {
      // Overwrite last historical point to also carry forecast values
      points[points.length - 1] = {
        ...points[points.length - 1],
        point: lastHist.value,
        band: [lastHist.value, lastHist.value],
      }
    }
    result.forecast.forEach((f) => {
      points.push({
        date: f.date,
        point: f.point,
        band: [f.lower, f.upper],
      })
    })
    return points
  })()

  const splitDate = result?.historical[result.historical.length - 1]?.date

  // Summary stats
  const lastHistValue = result?.historical[result.historical.length - 1]?.value
  const lastForecastValue = result?.forecast[result.forecast.length - 1]?.point
  const pctChange =
    lastHistValue && lastForecastValue
      ? ((lastForecastValue - lastHistValue) / lastHistValue) * 100
      : null
  const finalInterval = result?.forecast[result.forecast.length - 1]
  const coveragePct = result ? Math.round((1 - result.alpha) * 100) : 95

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">HPI Forecast</h1>
        <p className="mt-1 text-sm text-gray-500">
          Probabilistic forecasts of regional average house prices using time-series models
          fitted on historical UK HPI data.
        </p>
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        {/* Controls sidebar */}
        <aside className="lg:w-72 flex-shrink-0 space-y-5">
          {/* Region */}
          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
              Region
            </h2>
            <select
              value={region}
              onChange={(e) => setRegion(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {regions.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </div>

          {/* Method */}
          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
              Method
            </h2>
            <div className="space-y-2">
              {METHODS.map((m) => (
                <button
                  key={m.id}
                  onClick={() => setMethod(m.id)}
                  className={`w-full text-left px-3 py-2 rounded-lg border text-sm transition-colors ${
                    method === m.id
                      ? 'bg-blue-50 border-blue-400 text-blue-800'
                      : 'border-gray-200 text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <span className="font-medium">{m.label}</span>
                  <span className="block text-xs text-gray-500 mt-0.5">{m.description}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Horizon */}
          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
              Horizon
            </h2>
            <div className="grid grid-cols-4 gap-1">
              {STEP_OPTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => setSteps(s)}
                  className={`py-1.5 rounded-lg text-xs font-medium transition-colors ${
                    steps === s
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  {s}m
                </button>
              ))}
            </div>
          </div>

          {/* Confidence */}
          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
              Prediction Interval
            </h2>
            <div className="space-y-1">
              {ALPHA_OPTIONS.map((a) => (
                <button
                  key={a.value}
                  onClick={() => setAlpha(a.value)}
                  className={`w-full text-left px-3 py-1.5 rounded-lg border text-sm transition-colors ${
                    alpha === a.value
                      ? 'bg-blue-50 border-blue-400 text-blue-800 font-medium'
                      : 'border-gray-200 text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  {a.label}
                </button>
              ))}
            </div>
          </div>

          {/* Methodology note */}
          <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-xs text-amber-800 leading-relaxed">
            <p className="font-semibold mb-1">Methodology note</p>
            <p>
              Forecasts are based on historical price patterns only. They do not
              incorporate interest rate expectations, policy changes, or macro
              conditions. Treat longer-horizon forecasts as indicative only.
            </p>
          </div>
        </aside>

        {/* Main content */}
        <div className="flex-1 min-w-0 space-y-4">
          {/* Summary cards */}
          {result && !loading && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="bg-white rounded-xl border border-gray-200 p-4">
                <p className="text-xs text-gray-500 mb-1">Current avg price</p>
                <p className="text-lg font-bold text-gray-900">
                  {lastHistValue ? formatPrice(lastHistValue) : '—'}
                </p>
              </div>
              <div className="bg-white rounded-xl border border-gray-200 p-4">
                <p className="text-xs text-gray-500 mb-1">Forecast in {steps}m</p>
                <p className="text-lg font-bold text-blue-700">
                  {lastForecastValue ? formatPrice(lastForecastValue) : '—'}
                </p>
              </div>
              <div className="bg-white rounded-xl border border-gray-200 p-4">
                <p className="text-xs text-gray-500 mb-1">Expected change</p>
                <p
                  className={`text-lg font-bold ${
                    pctChange === null
                      ? 'text-gray-400'
                      : pctChange >= 0
                        ? 'text-green-700'
                        : 'text-red-700'
                  }`}
                >
                  {pctChange !== null ? `${pctChange >= 0 ? '+' : ''}${pctChange.toFixed(1)}%` : '—'}
                </p>
              </div>
              <div className="bg-white rounded-xl border border-gray-200 p-4">
                <p className="text-xs text-gray-500 mb-1">{coveragePct}% interval width</p>
                <p className="text-sm font-semibold text-gray-700">
                  {finalInterval
                    ? `${formatPrice(finalInterval.lower)} – ${formatPrice(finalInterval.upper)}`
                    : '—'}
                </p>
              </div>
            </div>
          )}

          {/* Chart */}
          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold text-gray-700">
                {region || 'Loading…'} — average house price
              </h2>
              {loading && (
                <span className="text-xs text-gray-400 animate-pulse">Forecasting…</span>
              )}
            </div>

            {error && (
              <div className="flex items-center justify-center h-64 text-red-600 text-sm">
                {error}
              </div>
            )}

            {!error && chartData.length === 0 && !loading && (
              <div className="flex items-center justify-center h-64 text-gray-400 text-sm">
                Select a region to view the forecast
              </div>
            )}

            {!error && (chartData.length > 0 || loading) && (
              <ResponsiveContainer width="100%" height={380}>
                <ComposedChart data={chartData} margin={{ top: 4, right: 16, bottom: 4, left: 16 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis
                    dataKey="date"
                    tickFormatter={formatDateTick}
                    tick={{ fontSize: 11, fill: '#6b7280' }}
                    interval="preserveStartEnd"
                    minTickGap={60}
                  />
                  <YAxis
                    tickFormatter={formatPrice}
                    tick={{ fontSize: 11, fill: '#6b7280' }}
                    width={72}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend
                    iconType="line"
                    wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
                  />

                  {/* Prediction interval band */}
                  <Area
                    type="monotone"
                    dataKey="band"
                    name="Prediction interval"
                    stroke="none"
                    fill="#3b82f6"
                    fillOpacity={0.12}
                    legendType="rect"
                    dot={false}
                    activeDot={false}
                    connectNulls={false}
                  />

                  {/* Dashed vertical line at forecast start */}
                  {splitDate && (
                    <ReferenceLine
                      x={splitDate}
                      stroke="#9ca3af"
                      strokeDasharray="4 4"
                      label={{ value: 'Forecast start', position: 'top', fontSize: 10, fill: '#9ca3af' }}
                    />
                  )}

                  {/* Historical line */}
                  <Line
                    type="monotone"
                    dataKey="historical"
                    name="Historical"
                    stroke="#374151"
                    strokeWidth={2}
                    dot={false}
                    connectNulls={false}
                  />

                  {/* Forecast line */}
                  <Line
                    type="monotone"
                    dataKey="point"
                    name="Forecast"
                    stroke="#2563eb"
                    strokeWidth={2}
                    strokeDasharray="6 3"
                    dot={false}
                    connectNulls={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            )}
          </div>

          {/* Forecast table */}
          {result && !loading && result.forecast.length > 0 && (
            <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-100">
                <h2 className="text-sm font-semibold text-gray-700">
                  Forecast detail — {coveragePct}% prediction interval
                </h2>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-gray-50 text-xs text-gray-500 uppercase">
                      <th className="px-4 py-2 text-left font-medium">Date</th>
                      <th className="px-4 py-2 text-right font-medium">Forecast</th>
                      <th className="px-4 py-2 text-right font-medium">Lower bound</th>
                      <th className="px-4 py-2 text-right font-medium">Upper bound</th>
                      <th className="px-4 py-2 text-right font-medium">Interval width</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {result.forecast.map((f, i) => {
                      const d = new Date(f.date)
                      const label = d.toLocaleDateString('en-GB', {
                        month: 'short',
                        year: 'numeric',
                      })
                      const width = f.upper - f.lower
                      const isEven = i % 2 === 0
                      return (
                        <tr key={f.date} className={isEven ? 'bg-white' : 'bg-gray-50/50'}>
                          <td className="px-4 py-2 text-gray-700">{label}</td>
                          <td className="px-4 py-2 text-right font-medium text-blue-700">
                            {formatPrice(f.point)}
                          </td>
                          <td className="px-4 py-2 text-right text-gray-500">
                            {formatPrice(f.lower)}
                          </td>
                          <td className="px-4 py-2 text-right text-gray-500">
                            {formatPrice(f.upper)}
                          </td>
                          <td className="px-4 py-2 text-right text-gray-400">
                            ±{formatPrice(width / 2)}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
