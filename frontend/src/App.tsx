import { Route, Routes } from 'react-router-dom'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import HPIForecast from './pages/HPIForecast'
import PropertyDetail from './pages/PropertyDetail'
import ValuationExplorer from './pages/ValuationExplorer'

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/property/:id" element={<PropertyDetail />} />
          <Route path="/valuation" element={<ValuationExplorer />} />
          <Route path="/hpi-forecast" element={<HPIForecast />} />
        </Routes>
      </main>
    </div>
  )
}
