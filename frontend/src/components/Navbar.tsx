import { Link, NavLink } from 'react-router-dom'

export default function Navbar() {
  return (
    <nav className="bg-slate-700 text-white shadow-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="text-xl font-bold tracking-tight hover:text-slate-200 transition-colors">
            Property Insights
          </Link>
          <div className="flex items-center gap-6">
            <NavLink
              to="/valuation"
              className={({ isActive }) =>
                `text-sm font-medium transition-colors ${
                  isActive ? 'text-white underline underline-offset-4' : 'text-slate-300 hover:text-white'
                }`
              }
            >
              Valuation Explorer
            </NavLink>
            <NavLink
              to="/mortgage"
              className={({ isActive }) =>
                `text-sm font-medium transition-colors ${
                  isActive ? 'text-white underline underline-offset-4' : 'text-slate-300 hover:text-white'
                }`
              }
            >
              Mortgage Calculator
            </NavLink>
          </div>
        </div>
      </div>
    </nav>
  )
}
