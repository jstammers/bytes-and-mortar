import {
  CartesianGrid,
  LineChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { PriceHistoryPoint } from '../../api/types'

interface Props {
  transactions: PriceHistoryPoint[]
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

interface TooltipPayload {
  payload?: {
    date: string
    price: number
    transaction_id: string
  }
}

function CustomTooltip({ payload }: TooltipPayload) {
  if (!payload) return null
  const { date, price } = payload
  return (
    <div className="bg-white border border-gray-200 rounded shadow-lg p-3 text-xs">
      <p className="font-semibold text-gray-800">{formatPrice(price)}</p>
      <p className="text-gray-500 mt-1">{formatDateAxis(date)}</p>
    </div>
  )
}

export default function PriceHistoryChart({ transactions }: Props) {
  const data = transactions.map((t) => ({
    date: t.date,
    price: t.price,
    transaction_id: t.transaction_id,
  }))

  return (
    <div className="w-full h-64">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            dataKey="date"
            tickFormatter={formatDateAxis}
            tick={{ fontSize: 11, fill: '#6b7280' }}
            tickLine={false}
            axisLine={{ stroke: '#e5e7eb' }}
          />
          <YAxis
            tickFormatter={formatPrice}
            tick={{ fontSize: 11, fill: '#6b7280' }}
            tickLine={false}
            axisLine={false}
            width={70}
          />
          <Tooltip content={<CustomTooltip />} />
          <Line
            type="monotone"
            dataKey="price"
            stroke="#2563eb"
            strokeWidth={2}
            dot={{ r: 5, fill: '#2563eb', strokeWidth: 2, stroke: '#fff' }}
            activeDot={{ r: 7 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
