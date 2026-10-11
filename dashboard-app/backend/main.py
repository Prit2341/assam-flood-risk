"""
Assam Rainfall Forecast — FastAPI backend
Serves real CSV data from data/rainfall_nowcast/
"""
import os
import math
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]           # D:\BISAG-N\India\Assam
DATA = ROOT / "data" / "rainfall_nowcast"

app = FastAPI(title="Assam Rainfall Forecast API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from susceptibility import router as susc_router, rain_router
app.include_router(susc_router)
app.include_router(rain_router)

# ── District metadata ─────────────────────────────────────────────────────────
DISTRICTS = {
    "nagaon":    {"label": "Nagaon",    "lat": 26.3456, "lon": 92.6835, "elev_m": 58,  "river": "Kopili"},
    "karimganj": {"label": "Karimganj", "lat": 24.8672, "lon": 92.3595, "elev_m": 22,  "river": "Kushiyara"},
    "marigaon":  {"label": "Marigaon",  "lat": 26.2301, "lon": 92.0317, "elev_m": 49,  "river": "Brahmaputra trib."},
    "sivasagar": {"label": "Sivasagar", "lat": 26.9876, "lon": 94.6377, "elev_m": 98,  "river": "Dikhow"},
}

FLOOD_EVENTS = [
    {"district": "nagaon",    "start": "2022-06-18", "end": "2022-06-27", "label": "Jun 2022 Kopili flood"},
    {"district": "karimganj", "start": "2022-06-18", "end": "2022-06-27", "label": "Jun 2022 Kushiyara flood"},
    {"district": "marigaon",  "start": "2022-06-18", "end": "2022-06-27", "label": "Jun 2022 Kopili breach"},
    {"district": "karimganj", "start": "2024-06-10", "end": "2024-06-22", "label": "Jun 2024 Karimganj event"},
    {"district": "sivasagar", "start": "2026-07-16", "end": "2026-07-25", "label": "Jul 2026 Sivasagar ⚠"},
]

# ── Cache ──────────────────────────────────────────────────────────────────────
_perf_cache: dict[str, pd.DataFrame] = {}
_pred_cache: dict[str, pd.DataFrame] = {}


def load_perf(district: str) -> pd.DataFrame:
    if district not in _perf_cache:
        p = DATA / f"stack_cnn_base_{district}_summary.csv"
        if not p.exists():
            raise HTTPException(404, f"No summary CSV for {district}")
        df = pd.read_csv(p)
        _perf_cache[district] = df[df["arm"] == "stacked"].copy()
    return _perf_cache[district]


def load_pred(district: str) -> pd.DataFrame:
    """Real out-of-sample predictions only (test period >= 2024), one row per (lead, target time).
    Lead 1 = rainfall v1 (base + CNN stacked) from scripts/120; leads 2-6 = base model from
    scripts/30 (stacked per-row predictions were not stored for leads > 1). Row 'datetime' in the
    horizons file is target - (lead-1) h (scripts/30: tp_mm.shift(-(h-1)))."""
    if district not in _pred_cache:
        frames = []
        hz = DATA / f"forecast_horizons_{district}.csv"
        if hz.exists():
            h = pd.read_csv(hz, parse_dates=["datetime"])
            h = h[h["lead_hour"] > 1]
            h["target_dt"] = h["datetime"] + pd.to_timedelta(h["lead_hour"] - 1, unit="h")
            h["model"] = "base model"
            frames.append(h.rename(columns={"lead_hour": "lead"})[["lead", "target_dt", "tp_mm", "q10_ens", "q50_final", "q90_ens", "model"]])
        v1 = DATA / f"v1_rolling1h_{district}_hourly.csv"
        if v1.exists():
            a = pd.read_csv(v1, parse_dates=["datetime"]).rename(columns={"datetime": "target_dt"})
            a["lead"] = 1
            a["model"] = "v1 (base + CNN)"
            frames.append(a[["lead", "target_dt", "tp_mm", "q10_ens", "q50_final", "q90_ens", "model"]])
        _pred_cache[district] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _pred_cache[district]


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/api/districts")
def get_districts():
    result = []
    for key, meta in DISTRICTS.items():
        perf = load_perf(key)
        h1 = perf[perf["lead"] == 1].iloc[0]
        result.append({
            "id":       key,
            "label":    meta["label"],
            "lat":      meta["lat"],
            "lon":      meta["lon"],
            "elev_m":   meta["elev_m"],
            "river":    meta["river"],
            "h1_corr":  round(float(h1["corr_all"]), 4),
            "h1_heavy_cov": round(float(h1["heavy_cov"]), 4),
        })
    return result


@app.get("/api/performance/{district}")
def get_performance(district: str):
    if district not in DISTRICTS:
        raise HTTPException(404)
    perf = load_perf(district)
    rows = []
    for _, r in perf.iterrows():
        rows.append({
            "lead":       int(r["lead"]),
            "corr":       round(float(r["corr_all"]),  4),
            "corr_rainy": round(float(r["corr_rainy"]),4),
            "rmse":       round(float(r["rmse_all"]),  4),
            "heavy_cov":  round(float(r["heavy_cov"]) * 100, 2),
            "auc":        round(float(r["auc_gate_heavy"]), 4),
            "n_test":     int(r["n_test"]),
        })
    return rows


@app.get("/api/forecast/{district}")
def get_forecast(
    district: str,
    date_str: str = Query(default="2024-06-24", alias="date"),
    hour:     int  = Query(default=6, ge=0, le=23),
):
    if district not in DISTRICTS:
        raise HTTPException(404)

    perf = load_perf(district)
    pred_df = load_pred(district)

    issue_dt = datetime.strptime(f"{date_str} {hour:02d}:00:00", "%Y-%m-%d %H:%M:%S")
    in_test  = date_str >= "2024-01-01"

    horizons = []
    for lead in range(1, 7):
        target_dt = issue_dt + timedelta(hours=lead)
        ph = perf[perf["lead"] == lead].iloc[0]
        rmse = float(ph["rmse_all"])

        # Real out-of-sample predictions only; no value is shown for dates without one.
        q50 = q10 = q90 = None
        model = None
        if not pred_df.empty:
            row = pred_df[(pred_df["lead"] == lead) & (pred_df["target_dt"] == target_dt)]
            if not row.empty:
                q50 = round(float(row.iloc[0]["q50_final"]), 3)
                q10 = round(float(row.iloc[0]["q10_ens"]),   3)
                q90 = round(float(row.iloc[0]["q90_ens"]),   3)
                model = row.iloc[0]["model"]

        horizons.append({
            "lead":     lead,
            "label":    f"h+{lead}",
            "target_dt": target_dt.isoformat(),
            "q10":      q10,
            "q50":      q50,
            "q90":      q90,
            "is_real":  q50 is not None,
            "model":    model,
        })

    # Event label
    event = None
    for ev in FLOOD_EVENTS:
        if ev["district"] == district and ev["start"] <= date_str <= ev["end"]:
            event = ev["label"]
            break

    return {
        "district":  district,
        "issue_dt":  issue_dt.isoformat(),
        "in_test":   in_test,
        "event":     event,
        "horizons":  horizons,
    }


@app.get("/api/daycomp/{district}")
def get_daycomp(
    district: str,
    date_str: str = Query(default="2024-06-24", alias="date"),
    lead:     int  = Query(default=1, ge=1, le=6),
):
    if district not in DISTRICTS:
        raise HTTPException(404)

    perf    = load_perf(district)
    pred_df = load_pred(district)
    in_test = date_str >= "2024-01-01"
    ph      = perf[perf["lead"] == lead].iloc[0]

    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    hours = list(range(24))
    labels = [f"{h:02d}:00" for h in hours]

    actual_vals: list[Optional[float]] = []
    pred_vals:   list[Optional[float]] = []
    q90_vals:    list[Optional[float]] = []

    if not pred_df.empty:
        sub = pred_df[(pred_df["lead"] == lead) & (pred_df["target_dt"].dt.date == target_date)].set_index("target_dt")
        for h in hours:
            obs_dt = datetime.combine(target_date, __import__('datetime').time(h, 0))
            if obs_dt in sub.index:
                r = sub.loc[obs_dt]
                actual_vals.append(round(float(r["tp_mm"]), 3))
                pred_vals.append(round(float(r["q50_final"]), 3))
                q90_vals.append(round(float(r["q90_ens"]), 3))
            else:
                actual_vals.append(None); pred_vals.append(None); q90_vals.append(None)
    else:
        actual_vals = pred_vals = q90_vals = [None] * 24
    # Training-period dates (< 2024) have no out-of-sample prediction: all values stay None.

    # Compute stats
    pairs = [(a, p) for a, p in zip(actual_vals, pred_vals) if a is not None and p is not None]
    rmse_day  = round(math.sqrt(sum((a - p)**2 for a, p in pairs) / len(pairs)), 4) if pairs else None
    bias_day  = round(sum(p - a for a, p in pairs) / len(pairs), 4) if pairs else None
    thresh    = 2.5
    hits      = sum(1 for a, p in pairs if a > thresh and p > thresh)
    misses    = sum(1 for a, p in pairs if a > thresh and p <= thresh)
    false_alm = sum(1 for a, p in pairs if a <= thresh and p > thresh)

    return {
        "district":    district,
        "date":        date_str,
        "lead":        lead,
        "in_test":     in_test,
        "labels":      labels,
        "actual":      actual_vals,
        "predicted":   pred_vals,
        "q90":         q90_vals,
        "model_corr":  round(float(ph["corr_all"]), 4),
        "model_heavy_cov": round(float(ph["heavy_cov"]) * 100, 2),
        "rmse_day":    rmse_day,
        "bias_day":    bias_day,
        "hits":        hits,
        "misses":      misses,
        "false_alarms": false_alm,
        "available":   any(v is not None for v in pred_vals),
        "model":       "v1 (base + CNN)" if lead == 1 else "base model",
    }


@app.get("/api/events")
def get_events():
    return FLOOD_EVENTS


@app.get("/health")
def health():
    return {"status": "ok", "data_root": str(DATA)}
