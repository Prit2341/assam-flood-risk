"""
Model results on heavy-rain days (IMD heavy >= 64.5 mm/day, district-mean IMERG; list from heavy_rain_days_imerg.csv).
For each heavy-rain day: the first usable Sentinel-1 (GFM) scene 0-6 days later; on that scene, observed flood %,
H6, H5 and H2 held-out-year mean flood probability and AUC (same functions as the dashboard).

Run:  python -u scripts/176_heavy_rain_results.py
Out:  data/dynamic_susceptibility/heavy_rain_results.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dashboard-app", "backend"))
import susceptibility as s

MAX_LAG = 6
H = pd.read_csv(os.path.join(s.DYN, "heavy_rain_days_imerg.csv"), parse_dates=["date"])
rows = []
for d, g in H.groupby("district"):
    idx = pd.read_csv(s.DYN / "labels" / d / "labels_index.csv", parse_dates=["date"])
    idx = idx[idx["obs_frac"] >= s.dyn.MIN_OBS_FRAC].sort_values("date")
    for _, r in g.iterrows():
        c = idx[(idx["date"] >= r["date"]) & (idx["date"] <= r["date"] + pd.Timedelta(days=MAX_LAG))]
        out = {"district": d, "rain_date": r["date"].date(), "rain_mm": r["rain_mm_district_mean"],
               "imd_category": r["imd_category"]}
        if c.empty:
            rows.append({**out, "scene_date": None})
            continue
        t = c.iloc[0]["date"].strftime("%Y-%m-%d")
        lab = np.load(s.DYN / "labels" / d / f"{t}.npz")["label"]
        o = lab != 255
        out.update(scene_date=t, lag_days=(c.iloc[0]["date"] - r["date"]).days,
                   observed_pct=round(float((lab[o] == 1).mean() * 100), 2))
        for name, fn in (("h6", s._h6_map), ("h5", s._h5_map), ("h2", s._h2_prob)):
            try:
                p = fn(d, t)
            except HTTPException:
                continue
            ok = o & ~np.isnan(p)
            y = lab[ok]
            out[f"{name}_mean_pct"] = round(float(p[ok].mean() * 100), 2)
            out[f"{name}_auc"] = round(float(roc_auc_score(y, p[ok])), 3) if 0 < y.mean() < 1 else np.nan
        rows.append(out)
        print(out, flush=True)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(s.DYN, "heavy_rain_results.csv"), index=False)
print(R.to_string(index=False))
