import { useState, useMemo } from 'react'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'

// ─── Types ────────────────────────────────────────────────────────────────────

type Mode = 'payment' | 'equity'

interface PaymentInputs {
  currentEquity: string
  currentMortgage: string
  propertyValue: string
  termYears: string
  interestRate: string
  monthlyIncome: string
  monthlyExpenses: string
  monthlySavings: string
}

interface EquityInputs {
  propertyValue: string
  desiredRepayment: string
  termYears: string
  interestRate: string
}

interface AmortisationRow {
  year: number
  balance: number
  principal: number
  interest: number
}

// ─── Maths ────────────────────────────────────────────────────────────────────

function monthlyPayment(principal: number, annualRate: number, termYears: number): number {
  if (principal <= 0) return 0
  if (annualRate === 0) return principal / (termYears * 12)
  const r = annualRate / 100 / 12
  const n = termYears * 12
  return (principal * r * Math.pow(1 + r, n)) / (Math.pow(1 + r, n) - 1)
}

function maxAffordablePrincipal(
  monthly: number,
  annualRate: number,
  termYears: number,
): number {
  if (monthly <= 0) return 0
  if (annualRate === 0) return monthly * termYears * 12
  const r = annualRate / 100 / 12
  const n = termYears * 12
  return (monthly * (Math.pow(1 + r, n) - 1)) / (r * Math.pow(1 + r, n))
}

function buildAmortisation(
  principal: number,
  annualRate: number,
  termYears: number,
): AmortisationRow[] {
  if (principal <= 0) return []
  const r = annualRate === 0 ? 0 : annualRate / 100 / 12
  const payment = monthlyPayment(principal, annualRate, termYears)
  const rows: AmortisationRow[] = []
  let balance = principal

  for (let year = 1; year <= termYears; year++) {
    let yearPrincipal = 0
    let yearInterest = 0
    for (let m = 0; m < 12; m++) {
      const interestCharge = balance * r
      const principalCharge = Math.min(payment - interestCharge, balance)
      yearInterest += interestCharge
      yearPrincipal += principalCharge
      balance = Math.max(0, balance - principalCharge)
    }
    rows.push({
      year,
      balance: Math.round(balance),
      principal: Math.round(yearPrincipal),
      interest: Math.round(yearInterest),
    })
    if (balance === 0) break
  }
  return rows
}

// ─── Formatting ───────────────────────────────────────────────────────────────

const gbp = new Intl.NumberFormat('en-GB', { style: 'currency', currency: 'GBP', maximumFractionDigits: 0 })
const pct = (n: number) => `${n.toFixed(1)}%`

function fmt(v: number) {
  return gbp.format(v)
}

// ─── Shared sub-components ────────────────────────────────────────────────────

function InputField({
  label,
  value,
  onChange,
  prefix,
  suffix,
  hint,
  step = '1',
  min = '0',
}: {
  label: string
  value: string
  onChange: (v: string) => void
  prefix?: string
  suffix?: string
  hint?: string
  step?: string
  min?: string
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-slate-700 mb-1">{label}</label>
      {hint && <p className="text-xs text-slate-500 mb-1">{hint}</p>}
      <div className="flex rounded-lg shadow-sm border border-slate-300 focus-within:border-slate-500 focus-within:ring-1 focus-within:ring-slate-500 overflow-hidden">
        {prefix && (
          <span className="flex items-center px-3 bg-slate-100 text-slate-500 text-sm border-r border-slate-300">
            {prefix}
          </span>
        )}
        <input
          type="number"
          min={min}
          step={step}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="flex-1 px-3 py-2 text-sm bg-white focus:outline-none"
        />
        {suffix && (
          <span className="flex items-center px-3 bg-slate-100 text-slate-500 text-sm border-l border-slate-300">
            {suffix}
          </span>
        )}
      </div>
    </div>
  )
}

function ResultCard({
  label,
  value,
  sub,
  highlight,
  warn,
}: {
  label: string
  value: string
  sub?: string
  highlight?: boolean
  warn?: boolean
}) {
  return (
    <div
      className={`rounded-xl p-4 border ${
        highlight
          ? 'bg-slate-700 text-white border-slate-600'
          : warn
            ? 'bg-amber-50 border-amber-300 text-slate-800'
            : 'bg-white border-slate-200 text-slate-800'
      }`}
    >
      <p className={`text-xs font-medium uppercase tracking-wide mb-1 ${highlight ? 'text-slate-300' : 'text-slate-500'}`}>
        {label}
      </p>
      <p className={`text-2xl font-bold ${highlight ? 'text-white' : warn ? 'text-amber-700' : 'text-slate-800'}`}>
        {value}
      </p>
      {sub && (
        <p className={`text-xs mt-1 ${highlight ? 'text-slate-300' : 'text-slate-500'}`}>{sub}</p>
      )}
    </div>
  )
}

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-3">
      {children}
    </h3>
  )
}

// ─── Amortisation chart ───────────────────────────────────────────────────────

function AmortisationChart({ rows }: { rows: AmortisationRow[] }) {
  if (rows.length === 0) return null
  const tickFmt = (v: number) =>
    v >= 1_000_000 ? `£${(v / 1_000_000).toFixed(1)}m` : `£${(v / 1_000).toFixed(0)}k`

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-6">
      <h3 className="text-sm font-semibold text-slate-700 mb-4">Amortisation Schedule</h3>
      <ResponsiveContainer width="100%" height={280}>
        <AreaChart data={rows} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="balanceGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#334155" stopOpacity={0.2} />
              <stop offset="95%" stopColor="#334155" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="interestGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis
            dataKey="year"
            tick={{ fontSize: 11 }}
            label={{ value: 'Year', position: 'insideBottom', offset: -2, fontSize: 11 }}
          />
          <YAxis tickFormatter={tickFmt} tick={{ fontSize: 11 }} width={56} />
          <Tooltip
            formatter={(value: number, name: string) => [fmt(value), name]}
            labelFormatter={(l) => `Year ${l}`}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Area
            type="monotone"
            dataKey="balance"
            name="Remaining Balance"
            stroke="#334155"
            fill="url(#balanceGrad)"
            strokeWidth={2}
          />
          <Area
            type="monotone"
            dataKey="interest"
            name="Annual Interest"
            stroke="#f59e0b"
            fill="url(#interestGrad)"
            strokeWidth={2}
          />
          <Area
            type="monotone"
            dataKey="principal"
            name="Annual Principal"
            stroke="#10b981"
            fill="none"
            strokeWidth={2}
            strokeDasharray="4 2"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

// ─── Payment mode ─────────────────────────────────────────────────────────────

const defaultPaymentInputs: PaymentInputs = {
  currentEquity: '200000',
  currentMortgage: '50000',
  propertyValue: '400000',
  termYears: '25',
  interestRate: '4.5',
  monthlyIncome: '5000',
  monthlyExpenses: '2000',
  monthlySavings: '500',
}

function PaymentMode() {
  const [inputs, setInputs] = useState<PaymentInputs>(defaultPaymentInputs)

  const set = (key: keyof PaymentInputs) => (v: string) =>
    setInputs((prev) => ({ ...prev, [key]: v }))

  const results = useMemo(() => {
    const equity = parseFloat(inputs.currentEquity) || 0
    const existingMortgage = parseFloat(inputs.currentMortgage) || 0
    const propValue = parseFloat(inputs.propertyValue) || 0
    const term = parseFloat(inputs.termYears) || 25
    const rate = parseFloat(inputs.interestRate) || 0
    const income = parseFloat(inputs.monthlyIncome) || 0
    const expenses = parseFloat(inputs.monthlyExpenses) || 0
    const savings = parseFloat(inputs.monthlySavings) || 0

    const netEquity = Math.max(0, equity - existingMortgage)
    const deposit = Math.min(netEquity, propValue)
    const mortgageNeeded = Math.max(0, propValue - deposit)
    const ltv = propValue > 0 ? (mortgageNeeded / propValue) * 100 : 0
    const monthly = monthlyPayment(mortgageNeeded, rate, term)
    const totalRepaid = monthly * term * 12
    const totalInterest = totalRepaid - mortgageNeeded
    const availableBudget = income - expenses - savings
    const affordable = availableBudget >= monthly
    const shortfall = monthly - availableBudget

    const amortisation = buildAmortisation(mortgageNeeded, rate, term)

    return {
      netEquity,
      deposit,
      mortgageNeeded,
      ltv,
      monthly,
      totalRepaid,
      totalInterest,
      availableBudget,
      affordable,
      shortfall,
      amortisation,
      term,
      rate,
    }
  }, [inputs])

  return (
    <div className="space-y-6">
      {/* Inputs */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Property */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 space-y-4">
          <SectionHeading>Property</SectionHeading>
          <InputField
            label="Property value"
            value={inputs.propertyValue}
            onChange={set('propertyValue')}
            prefix="£"
            step="1000"
          />
          <InputField
            label="Mortgage term"
            value={inputs.termYears}
            onChange={set('termYears')}
            suffix="years"
            step="1"
            min="1"
          />
          <InputField
            label="Interest rate"
            value={inputs.interestRate}
            onChange={set('interestRate')}
            suffix="%"
            step="0.1"
          />
        </div>

        {/* Your equity */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 space-y-4">
          <SectionHeading>Your Equity</SectionHeading>
          <InputField
            label="Total equity"
            hint="Value of home + savings + investments"
            value={inputs.currentEquity}
            onChange={set('currentEquity')}
            prefix="£"
            step="1000"
          />
          <InputField
            label="Existing mortgage outstanding"
            hint="Set to 0 if no current mortgage"
            value={inputs.currentMortgage}
            onChange={set('currentMortgage')}
            prefix="£"
            step="1000"
          />
        </div>

        {/* Affordability */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 space-y-4">
          <SectionHeading>Monthly Budget</SectionHeading>
          <InputField
            label="Household income"
            value={inputs.monthlyIncome}
            onChange={set('monthlyIncome')}
            prefix="£"
            step="100"
          />
          <InputField
            label="Monthly expenses"
            hint="Bills, food, transport, etc."
            value={inputs.monthlyExpenses}
            onChange={set('monthlyExpenses')}
            prefix="£"
            step="100"
          />
          <InputField
            label="Monthly savings target"
            value={inputs.monthlySavings}
            onChange={set('monthlySavings')}
            prefix="£"
            step="100"
          />
        </div>
      </div>

      {/* Results */}
      <div className="space-y-3">
        <SectionHeading>Results</SectionHeading>

        {/* Affordability banner */}
        {results.mortgageNeeded > 0 && (
          <div
            className={`rounded-xl px-5 py-3 text-sm font-medium border ${
              results.affordable
                ? 'bg-emerald-50 border-emerald-300 text-emerald-800'
                : 'bg-red-50 border-red-300 text-red-800'
            }`}
          >
            {results.affordable ? (
              <>
                ✓ Affordable — your available budget of {fmt(results.availableBudget)}/mo exceeds
                the repayment with {fmt(results.availableBudget - results.monthly)}/mo to spare.
              </>
            ) : (
              <>
                ✗ Affordability gap — you are {fmt(results.shortfall)}/mo short after income,
                expenses, and savings target. Consider a longer term or higher deposit.
              </>
            )}
          </div>
        )}

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <ResultCard
            label="Net Equity"
            value={fmt(results.netEquity)}
            sub="After existing mortgage"
          />
          <ResultCard
            label="Deposit"
            value={fmt(results.deposit)}
            sub={pct((results.deposit / (parseFloat(inputs.propertyValue) || 1)) * 100) + ' of value'}
          />
          <ResultCard
            label="Mortgage Needed"
            value={fmt(results.mortgageNeeded)}
            sub={`LTV ${pct(results.ltv)}`}
            highlight
          />
          <ResultCard
            label="Monthly Repayment"
            value={fmt(results.monthly)}
            sub={`over ${results.term} yrs @ ${results.rate}%`}
            highlight
          />
          <ResultCard
            label="Total Interest"
            value={fmt(results.totalInterest)}
            warn={results.totalInterest > results.mortgageNeeded * 0.5}
          />
          <ResultCard label="Total Repaid" value={fmt(results.totalRepaid)} />
        </div>
      </div>

      {/* Chart */}
      <AmortisationChart rows={results.amortisation} />
    </div>
  )
}

// ─── Equity mode ──────────────────────────────────────────────────────────────

const defaultEquityInputs: EquityInputs = {
  propertyValue: '400000',
  desiredRepayment: '1500',
  termYears: '25',
  interestRate: '4.5',
}

function EquityMode() {
  const [inputs, setInputs] = useState<EquityInputs>(defaultEquityInputs)

  const set = (key: keyof EquityInputs) => (v: string) =>
    setInputs((prev) => ({ ...prev, [key]: v }))

  const results = useMemo(() => {
    const propValue = parseFloat(inputs.propertyValue) || 0
    const desired = parseFloat(inputs.desiredRepayment) || 0
    const term = parseFloat(inputs.termYears) || 25
    const rate = parseFloat(inputs.interestRate) || 0

    const maxMortgage = maxAffordablePrincipal(desired, rate, term)
    const requiredDeposit = Math.max(0, propValue - maxMortgage)
    const ltv = propValue > 0 ? (Math.min(maxMortgage, propValue) / propValue) * 100 : 0
    const depositPct = propValue > 0 ? (requiredDeposit / propValue) * 100 : 0
    const totalRepaid = desired * term * 12
    const actualMortgage = Math.min(maxMortgage, propValue)
    const totalInterest = totalRepaid - actualMortgage

    const amortisation = buildAmortisation(actualMortgage, rate, term)

    return {
      maxMortgage,
      actualMortgage,
      requiredDeposit,
      ltv,
      depositPct,
      totalRepaid,
      totalInterest,
      term,
      rate,
      amortisation,
      propValue,
    }
  }, [inputs])

  const interestRateScenarios = useMemo(() => {
    const propValue = parseFloat(inputs.propertyValue) || 0
    const desired = parseFloat(inputs.desiredRepayment) || 0
    const term = parseFloat(inputs.termYears) || 25
    const baseRate = parseFloat(inputs.interestRate) || 0

    return [-1, -0.5, 0, 0.5, 1, 1.5, 2].map((delta) => {
      const r = Math.max(0, baseRate + delta)
      const maxMort = maxAffordablePrincipal(desired, r, term)
      const deposit = Math.max(0, propValue - maxMort)
      return { rate: r, deposit, maxMortgage: Math.min(maxMort, propValue) }
    })
  }, [inputs])

  return (
    <div className="space-y-6">
      {/* Inputs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white rounded-xl border border-slate-200 p-5 space-y-4">
          <SectionHeading>Property</SectionHeading>
          <InputField
            label="Property value"
            value={inputs.propertyValue}
            onChange={set('propertyValue')}
            prefix="£"
            step="1000"
          />
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-5 space-y-4">
          <SectionHeading>Target</SectionHeading>
          <InputField
            label="Desired monthly repayment"
            value={inputs.desiredRepayment}
            onChange={set('desiredRepayment')}
            prefix="£"
            step="50"
          />
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-5 space-y-4">
          <SectionHeading>Mortgage</SectionHeading>
          <InputField
            label="Term"
            value={inputs.termYears}
            onChange={set('termYears')}
            suffix="years"
            step="1"
            min="1"
          />
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-5 space-y-4">
          <SectionHeading>Rate</SectionHeading>
          <InputField
            label="Interest rate"
            value={inputs.interestRate}
            onChange={set('interestRate')}
            suffix="%"
            step="0.1"
          />
        </div>
      </div>

      {/* Results */}
      <div className="space-y-3">
        <SectionHeading>Required Equity</SectionHeading>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          <ResultCard
            label="Deposit Required"
            value={fmt(results.requiredDeposit)}
            sub={pct(results.depositPct) + ' of property value'}
            highlight
          />
          <ResultCard
            label="Max Mortgage"
            value={fmt(results.actualMortgage)}
            sub={`LTV ${pct(results.ltv)}`}
            highlight
          />
          <ResultCard
            label="Monthly Repayment"
            value={fmt(parseFloat(inputs.desiredRepayment) || 0)}
            sub={`over ${results.term} yrs`}
          />
          <ResultCard label="Total Interest" value={fmt(results.totalInterest)} />
          <ResultCard label="Total Repaid" value={fmt(results.totalRepaid)} />
        </div>
      </div>

      {/* Rate sensitivity table */}
      <div className="bg-white rounded-xl border border-slate-200 p-5">
        <h3 className="text-sm font-semibold text-slate-700 mb-4">
          Interest Rate Sensitivity — Deposit Required
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-100">
                <th className="text-left py-2 pr-4 text-xs text-slate-500 font-medium">Rate</th>
                <th className="text-right py-2 pr-4 text-xs text-slate-500 font-medium">
                  Max Mortgage
                </th>
                <th className="text-right py-2 text-xs text-slate-500 font-medium">
                  Deposit Needed
                </th>
              </tr>
            </thead>
            <tbody>
              {interestRateScenarios.map(({ rate, deposit, maxMortgage }, i) => {
                const isCurrent =
                  Math.abs(rate - (parseFloat(inputs.interestRate) || 0)) < 0.001
                return (
                  <tr
                    key={i}
                    className={`border-b border-slate-50 ${isCurrent ? 'bg-slate-50 font-semibold' : ''}`}
                  >
                    <td className="py-2 pr-4 text-slate-700">
                      {rate.toFixed(1)}%{isCurrent && ' ← current'}
                    </td>
                    <td className="py-2 pr-4 text-right text-slate-700">{fmt(maxMortgage)}</td>
                    <td className="py-2 text-right text-slate-700">{fmt(deposit)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Chart */}
      <AmortisationChart rows={results.amortisation} />
    </div>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function MortgageCalculator() {
  const [mode, setMode] = useState<Mode>('payment')

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-slate-800">Mortgage Calculator</h1>
        <p className="text-sm text-slate-500 mt-1">
          UK mortgage planning tool — figures are illustrative and exclude fees, stamp duty, and
          survey costs.
        </p>
      </div>

      {/* Mode toggle */}
      <div className="flex gap-1 mb-6 bg-slate-100 rounded-xl p-1 w-fit">
        <button
          onClick={() => setMode('payment')}
          className={`px-5 py-2 rounded-lg text-sm font-medium transition-all ${
            mode === 'payment'
              ? 'bg-white text-slate-800 shadow-sm'
              : 'text-slate-500 hover:text-slate-700'
          }`}
        >
          Calculate Repayments
        </button>
        <button
          onClick={() => setMode('equity')}
          className={`px-5 py-2 rounded-lg text-sm font-medium transition-all ${
            mode === 'equity'
              ? 'bg-white text-slate-800 shadow-sm'
              : 'text-slate-500 hover:text-slate-700'
          }`}
        >
          Find Required Equity
        </button>
      </div>

      {mode === 'payment' ? <PaymentMode /> : <EquityMode />}
    </div>
  )
}
