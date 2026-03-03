# Property Valuation App — User Guide

The property valuation app is a full-stack web application that lets you explore UK property data, view transaction histories, and get instant price estimates for any combination of property attributes. It runs alongside the existing data pipeline and is entirely self-contained, using synthetic but realistic data seeded from the same area codes as the pipeline.

---

## Running the app

### Prerequisites

- Python 3.11+ with `uv` installed
- Node.js 18+ with `npm` installed
- The project checked out and its Python dependencies installed (`just install`)

### Starting the backend

```bash
just install-app   # install FastAPI + uvicorn into the project venv (once only)
just dev-api       # starts the API at http://localhost:8000
```

The interactive API docs are available at <http://localhost:8000/docs>.

### Starting the frontend

Open a second terminal:

```bash
just install-frontend   # npm install inside frontend/ (once only)
just dev-frontend       # starts Vite dev server at http://localhost:5173
```

The Vite dev server proxies all `/api` requests to `localhost:8000`, so you only need to visit <http://localhost:5173>.

---

## Pages

### Home — property search

The home page lets you search the property dataset by postcode prefix (e.g. `SW1A`, `LS7`) or by address substring. Results appear as cards showing:

- Full address and postcode
- Property type and last recorded sale price
- Floor area
- EPC energy rating badge (colour-coded: green for A/B, yellow for C/D, orange for E/F, red for G)

Click any card to open the property detail view.

### Property detail

The detail page loads three datasets in parallel and presents them in tabs:

#### Price History tab

A line chart of every recorded sale for the property, with individual sale events marked as dots. The Y-axis is automatically formatted as £Xk or £X.Xm depending on price magnitude. Hover over any point to see the exact date and price.

#### Area Comparison tab

A composite chart overlaying two series:

- **District average** (filled area) — the UK House Price Index monthly average for the property's district, normalised to the property's last sale price.
- **Property sales** (scatter + line) — the property's own sale prices over the same timeline.

This shows at a glance whether the property has tracked, outperformed, or underperformed its local market.

#### Similar Properties tab

A scatter plot of up to eight nearby properties with comparable attributes. Each point is coloured by EPC rating (same scale as the search cards). The current property is highlighted in blue. Hovering a point shows the address, price, floor area, and bedrooms.

#### Attributes sidebar

Always visible alongside the chart tabs, the sidebar shows the full EPC record for the property:

| Field | Description |
|---|---|
| Energy rating | Letter grade A–G |
| Current efficiency | SAP score (1–100) |
| Potential efficiency | SAP score if recommended improvements are made |
| CO₂ emissions (current / potential) | Tonnes per year |
| Floor area | Square metres |
| Bedrooms / heated rooms | Room counts |
| Construction age band | e.g. "1967–1975" |

### Valuation Explorer

The valuation explorer lets you build a hypothetical property and get an instant price estimate without needing to find a real listing.

**Inputs:**

| Control | Options |
|---|---|
| Property type | Detached, Semi-detached, Terraced, Flat |
| Postcode area | 19 UK area codes (London, Leeds, Manchester, Birmingham) |
| Bedrooms | Slider 1–6 |
| Floor area | Slider 40–350 sqm |
| Energy efficiency | Slider 1–100 (live EPC letter grade shown) |

The price estimate updates automatically 300 ms after you stop adjusting any slider. The display shows:

- **Estimated price** — large, formatted to the nearest £100
- **Confidence interval** — ±15% range in smaller text below

Beneath the estimate, three small sensitivity charts show how the price changes as each attribute varies independently (holding everything else fixed). A reference line marks your current input value on each chart.

---

## API reference

The backend exposes two groups of endpoints under `/api`.

### Properties — `/api/properties`

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/properties/search?q=<query>` | Search by postcode prefix or address substring |
| `GET` | `/api/properties/{id}` | Full property record including EPC fields |
| `GET` | `/api/properties/{id}/history` | All recorded sale transactions |
| `GET` | `/api/properties/{id}/area-comparison` | HPI series + property sales for charting |
| `GET` | `/api/properties/{id}/similar?n=8` | Up to `n` similar properties (max 20) |

### Valuation — `/api/valuation`

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/valuation/predict` | Single price estimate with confidence interval |
| `POST` | `/api/valuation/sensitivity` | Estimate + sensitivity curves for all three attributes |
| `GET` | `/api/valuation/postcode-prefixes` | List of all 19 supported postcode area codes |

**Request body for `/predict` and `/sensitivity`:**

```json
{
  "property_type": "T",
  "bedrooms": 3,
  "floor_area": 85,
  "energy_efficiency_score": 68,
  "postcode_prefix": "LS7"
}
```

Property type codes: `D` (Detached), `S` (Semi-detached), `T` (Terraced), `F` (Flat).

---

## How the valuation model works

The model uses a log-linear formula:

```
price = base_price × type_multiplier × (floor_area / 80)^0.7 × bedroom_factor × energy_factor
```

Where:

- **base_price** — median price for the postcode area (e.g. £1,400,000 for W1A, £210,000 for LS7)
- **type_multiplier** — 1.4× Detached, 1.15× Semi-detached, 1.0× Terraced, 0.8× Flat
- **floor_area factor** — sub-linear (exponent 0.7), so doubling the area does not double the price
- **bedroom_factor** — `1 + 0.08 × (bedrooms − 1)`, adding ~8% per additional bedroom above one
- **energy_factor** — `1 + 0.003 × (score − 55)`, adding ~0.3% per SAP point above average (55)

The confidence interval is ±15% around the central estimate, rounded to the nearest £100.

This is a **mock model** intended for demonstration. It is deterministic and does not use real training data. To replace it with a trained model, update `src/app/services/valuation_model.py` — the API schema and frontend are already wired up.

---

## Data

The app uses synthetic data generated at startup by `MockDataService` (`src/app/services/mock_data.py`):

- **50 properties** across five districts (W1A, E14, LS7, M1, B15), with realistic addresses, postcodes, EPC attributes, and 2–4 historical sale transactions each.
- **HPI series** — 300 monthly data points per district (January 2000 – December 2024), with a modelled 2008 financial crisis dip and a 2020 COVID pause.

All data is seeded (NumPy seed 42) so results are deterministic across restarts.
