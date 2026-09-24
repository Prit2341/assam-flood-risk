# Assam Rainfall Forecast Dashboard

React + FastAPI full-stack app.

## Quick start

```bat
dashboard-app\start.bat
```

Then open http://localhost:5173

## Manual start

**Backend** (terminal 1):
```bash
cd dashboard-app/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend** (terminal 2):
```bash
cd dashboard-app/frontend
npm install
npm run dev
```

## API endpoints

| Endpoint | Description |
|---|---|
| `GET /api/districts` | All 4 districts with h+1 metrics |
| `GET /api/performance/{district}` | Per-lead corr/rmse/heavy_cov |
| `GET /api/forecast/{district}?date=&hour=` | 6-horizon forecast (real if ≥2024) |
| `GET /api/daycomp/{district}?date=&lead=` | 24-hour observed vs. predicted |
| `GET /api/events` | Flood event windows |
| `GET /docs` | FastAPI auto-docs |

## Data

- Real predictions from `data/rainfall_nowcast/base_rainfall_{district}_predictions.csv` (2024–2026)
- Performance from `data/rainfall_nowcast/stack_cnn_base_{district}_summary.csv`
- Train period (pre-2024): synthetic diurnal patterns seeded by date
