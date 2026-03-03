import {
  CartesianGrid,
  LineChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { SensitivityResult } from '../../api/types'

interface Props {
  sensitivities: SensitivityResult[]
}

function formatPrice(value: number): string {
  if (value >= 1_000_000) return `£${(value / 1_000_000).toFixed(1)}m`
  if (value >= 1_000) return `£${Math.round(value / 1_000)}k`
  return `£${value}`
}

interface SingleChartProps {
  result: SensitivityResult
}

function SingleSensitivityChart({ result }: SingleChartProps) {
  const data = result.points.map((p) => ({
    value: p.value,
    price: p.price,
    label: p.label,
  }))

  return (
    <div className="flex-1 min-w-0">
      <h4 className="text-xs font-semibold text-gray-600 mb-2 text-center">
        {result.attribute_label}
      </h4>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              dataKey="value"
              tick={{ fontSize: 9, fill: '#6b7280' }}
              tickLine={false}
              axisLine={{ stroke: '#e5e7eb' }}
              tickFormatter={(v: number) =>
                result.attribute === 'bedrooms' ? `${v}` : `${Math.round(v)}`
              }
            />
            <YAxis
              tickFormatter={formatPrice}
              tick={{ fontSize: 9, fill: '#6b7280' }}
              tickLine={false}
              axisLine={false}
              width={55}
            />
            <Tooltip
              formatter={(value: number) => [formatPrice(value), 'Est. Price']}
              labelFormatter={(label: number) => {
                const point = data.find((p) => p.value === label)
                return point?.label ?? String(label)
              }}
            />
            <ReferenceLine
              x={result.current_value}
              stroke="#2563eb"
              strokeDasharray="4 2"
              strokeWidth={2}
            />
            <Line
              type="monotone"
              dataKey="price"
              stroke="#7c3aed"
              strokeWidth={2}
              dot={{ r: 3, fill: '#7c3aed' }}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export default function ValuationSensitivityChart({ sensitivities }: Props) {
  return (
    <div className="flex gap-4 flex-wrap md:flex-nowrap">
      {sensitivities.map((s) => (
        <SingleSensitivityChart key={s.attribute} result={s} />
      ))}
    </div>
  )
}
