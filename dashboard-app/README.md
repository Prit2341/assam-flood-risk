# Assam Flood Risk Dashboard

React + FastAPI full-stack app. One page per date: the flood susceptibility map for the chosen district and date, then that day's rainfall (predicted vs actual) and the 10 days of rain before it.

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
| `GET /api/rain/evaluation` | Rainfall v1 vs persistence/climatology summary |
| `GET /api/rain/day/{district}?date=` | 24 hourly values: observed, v1 1-h-ahead, persistence |
| `GET /api/susc/status` | Label counts, AHP weights, pre-registered criteria and results |
| `GET /api/susc/dates/{district}` | Dates with a usable Sentinel-1 (GFM) scene |
| `GET /api/susc/map/{district}?date=` | Map bounds, layers available on that date, observed flood %, per-scene AUC and mean probability for AHP/H2/H5/H6 |
| `GET /api/susc/layer/{district}/{name}.png?date=` | Map overlay PNG (layers below) |
| `GET /api/susc/rain_window/{district}?date=` | IMERG daily rain for the 10 days up to the date |
| `GET /docs` | FastAPI auto-docs |

## Susceptibility layers

| Layer | What it is |
|---|---|
| `ml_h6` (default) | H6, the working model for the 4 districts (Amendment 6): H5 inputs plus what the most recent earlier Sentinel-1 scene (up to 30 days back) showed at each cell |
| `ml_h5` | H5 (Amendment 4): H2 plus each cell's own Sentinel-1 flood history |
| `ml_h2` | H2: terrain + rain + soil moisture + GloFAS river flow, no flood history (the model for districts without labels) |
| `ml_h6_rel`, `ml_h5_rel`, `ml_h2_rel` | Rank of each cell within the date: where risk is highest, not how much floods |
| `ahp_dynamic` | AHP index with the 5-day rain factor (no training) |
| `observed` | GFM flooded / dry / not observed |
| static layers | HAND, elevation, slope, TWI, distance to major river, drainage density, NDVI, land cover |

Every ML map for a past date comes from the model trained **without that date's year**, so it is a held-out prediction. The H5/H6 per-pixel history is also built without that year. If a model is missing for a year, the page falls back H6 → H5 → H2.

Probability colour scale (log, same on every date): light blue 0.01%, very pale blue 0.1%, pale yellow 1%, orange 10%, dark red 50% or more.

Model files the backend reads (built by the scripts, not by the app):
- `data/dynamic_susceptibility/models_h2_loyo/` (scripts/171)
- `data/dynamic_susceptibility/models_h5_loyo/` and `h5_history/` (scripts/175)
- `data/dynamic_susceptibility/models_h6_loyo/` (scripts/180); previous-scene features are computed on request from the label rasters

The `.claude/launch.json` backend entry runs without `--reload`, so restart it after editing `backend/*.py`.

## Data

- Rainfall: real out-of-sample predictions from `data/rainfall_nowcast/` (test period 2024-01 to 2026-03). Dates without one show nothing; the earlier synthetic fill for the training period was removed.
- Susceptibility: `data/dynamic_susceptibility/` (static factors, GFM labels, met, GloFAS, models). Nothing is synthetic.
