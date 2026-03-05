import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { AreaComparisonData } from '../../api/types'

interface Props {
  data: AreaComparisonData
}

function formatPrice(value: number): string {
  if (value >= 1_000_000) return `£${(value / 1_000_000).toFixed(1)}m`
  if (value >= 1_000) return `£${Math.round(value / 1_000)}k`
  return `£${value}`
}

function formatDateAxis(dateStr: string): string {
  const d = new Date(dateStr)
  return d.toLocaleDateString('en-GB', { month: 'short', year: 'numeric' })
}

export default function AreaComparisonChart({ data }: Props) {
  // Build a merged dataset: every 6th HPI point to avoid clutter
  const hpiData = data.hpi_dates
    .map((date, i) => ({ date, hpi_price: data.hpi_average_prices[i] }))
    .filter((_, i) => i % 6 === 0)

  // Map sale dates for overlay
  const saleLookup: Record<string, number> = {}
  data.sale_dates.forEach((d, i) => {
    saleLookup[d] = data.sale_prices[i]
  })

  const chartData = hpiData.map((point) => ({
    ...point,
    sale_price: saleLookup[point.date] ?? null,
  }))

  // Also add any sale dates not already in chart
  data.sale_dates.forEach((d, i) => {
    if (!chartData.find((p) => p.date === d)) {
      chartData.push({ date: d, hpi_price: 0, sale_price: data.sale_prices[i] })
    }
  })
  chartData.sort((a, b) => a.date.localeCompare(b.date))

  return (
    <div className="w-full h-72">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            dataKey="date"
            tickFormatter={(d: string) => formatDateAxis(d)}
            tick={{ fontSize: 10, fill: '#6b7280' }}
            tickLine={false}
            axisLine={{ stroke: '#e5e7eb' }}
            interval="preserveStartEnd"
          />
          <YAxis
            tickFormatter={formatPrice}
            tick={{ fontSize: 11, fill: '#6b7280' }}
            tickLine={false}
            axisLine={false}
            width={70}
          />
          <Tooltip
            formatter={(value: number, name: string) => [
              formatPrice(value),
              name === 'hpi_price' ? `${data.district} Average` : 'This Property',
            ]}
            labelFormatter={(label: string) => formatDateAxis(label)}
          />
          <Legend
            formatter={(value: string) =>
              value === 'hpi_price' ? `${data.district} Average` : 'This Property Sale'
            }
          />
          <Area
            type="monotone"
            dataKey="hpi_price"
            stroke="#93c5fd"
            fill="#dbeafe"
            strokeWidth={1.5}
            dot={false}
            name="hpi_price"
          />
          <Scatter dataKey="sale_price" fill="#2563eb" name="sale_price" />
          <Line
            type="monotone"
            dataKey="sale_price"
            stroke="#2563eb"
            strokeWidth={2}
            dot={{ r: 6, fill: '#2563eb', stroke: '#fff', strokeWidth: 2 }}
            connectNulls={false}
            name="sale_price_line"
            legendType="none"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
