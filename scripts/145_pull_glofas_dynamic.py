"""
GloFAS v4 (LISFLOOD, ERA5-forced) daily discharge 2015-2026 for the dynamic susceptibility model.

Extends data/glofas_q90 (2015-2024, tributary gauges) in two ways:
  1. adds 2025-2026 (consolidated where released, else 'intermediate'), so the 2026 Sivasagar
     scenes get a discharge input;
  2. adds a Brahmaputra main-stem cell next to Morigaon, Nagaon and Sivasagar, because these districts
     border the main stem and the tributary gauge alone may miss main-stem-driven inundation. This is a
     hypothesis (no specific paper checked for it); whether the input helps is decided by the
     held-out test, not assumed.
Channel cell = max mean discharge in a small box (same pragmatic rule as pull_glofas_q90.py).
GloFAS v4: Grimaldi et al. 2023 / Harrigan et al. 2020 (ESSD 12:2043, GloFAS-ERA5 reanalysis).

Out: data/glofas_dyn/<point>.csv (date, discharge_m3s), points.json
"""
import json
import os

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

OUT = "data/glofas_dyn"
os.makedirs(OUT, exist_ok=True)
POINTS = [  # tag, lat, lon, box half-width
    ("nagaon_kopili", 26.20, 92.63, 0.15), ("morigaon_kopili", 26.16, 92.35, 0.15),
    ("sivasagar_dikhow", 26.92, 94.73, 0.15), ("karimganj_kushiyara", 24.87, 92.35, 0.15),
    ("morigaon_brahmaputra", 26.40, 92.30, 0.20), ("nagaon_brahmaputra", 26.62, 92.85, 0.20),
    ("sivasagar_brahmaputra", 27.00, 94.40, 0.20),
    # Amendment 2 (2026-10-14): Karimganj had no main-stem input (q_main 100% NaN). Barak main stem upstream
    # of the Surma/Kushiyara split (Badarpur-Silchar reach); the box excludes the Kushiyara gauge cell.
    ("karimganj_barak", 24.85, 92.65, 0.20),
]


def req(lat, lon, half, years, ptype, target):
    if os.path.exists(target):
        return target
    cdsapi.Client().retrieve("cems-glofas-historical", {
        "system_version": ["version_4_0"], "hydrological_model": ["lisflood"], "product_type": [ptype],
        "timespan": ["time_mean"], "variable": ["average_river_discharge_in_the_last_24_hours"],
        "year": [str(y) for y in years], "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)], "data_format": "grib2", "download_format": "unarchived",
        "area": [lat + half, lon - half, lat - half, lon + half]}, target)
    return target


def series(files):
    import cfgrib
    arrs = []
    for f in files:
        for ds in cfgrib.open_datasets(f):
            if "avg_dis" in ds:
                arrs.append(ds["avg_dis"])
    da = xr.concat(arrs, dim="time").load()
    mm = da.mean("time")
    i, j = divmod(int(np.nanargmax(mm.values)), mm.shape[1])
    s = da.isel(latitude=i, longitude=j).to_series()
    s.index = pd.to_datetime(s.index).normalize()
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s, float(mm.latitude[i]), float(mm.longitude[j])


def year_file(tag, lat, lon, half, y):
    """Fetch one point-year (consolidated, falling back to intermediate for recent years)."""
    for ptype in (("consolidated", "intermediate") if y == 2025 else (("intermediate",) if y == 2026 else ("consolidated",))):
        try:
            return req(lat, lon, half, [y], ptype, f"{OUT}/{tag}_{y}_{ptype[:4]}.grib")
        except Exception as e:
            print(tag, y, ptype, "failed:", str(e)[:150], flush=True)


# Submit missing point-years 3 at a time: EWDS rejects more ("Number queued requests for this dataset is
# temporarily limited", 2026-10-12); each request spends ~3-7 min queued and seconds downloading.
from concurrent.futures import ThreadPoolExecutor
jobs = [(tag, lat, lon, half, y) for tag, lat, lon, half in POINTS for y in range(2015, 2027)
        if not (tag.endswith(("kopili", "dikhow", "kushiyara")) and os.path.exists(f"data/glofas_q90/{tag.split('_')[0]}_{y}.grib"))
        and not os.path.exists(f"{OUT}/{tag}_{y}_cons.grib") and not os.path.exists(f"{OUT}/{tag}_{y}_inte.grib")]
print(f"{len(jobs)} point-years to fetch", flush=True)
with ThreadPoolExecutor(int(os.environ.get("GLOFAS_PARALLEL", 3))) as ex:
    list(ex.map(lambda j: year_file(*j), jobs))

meta = {}
for tag, lat, lon, half in POINTS:
    files = []
    q90tag = tag.split("_")[0] if tag.endswith(("kopili", "dikhow", "kushiyara")) else None
    for y in range(2015, 2027):
        old = f"data/glofas_q90/{q90tag}_{y}.grib" if q90tag else None
        if old and os.path.exists(old):          # 2015-2024 tributary years already pulled by pull_glofas_q90.py
            files.append(old)
            continue
        # one year per request: multi-year requests exceed the EWDS cost limit (403, 2026-10-10)
        f = year_file(tag, lat, lon, half, y)
        if f:
            files.append(f)
    s, clat, clon = series([f for f in files if os.path.exists(f)])
    s.rename("discharge_m3s").rename_axis("date").reset_index().to_csv(f"{OUT}/{tag}.csv", index=False)
    meta[tag] = {"chan_lat": clat, "chan_lon": clon, "first": str(s.index.min().date()),
                 "last": str(s.index.max().date()), "n": int(s.notna().sum()), "mean": float(s.mean())}
    print(tag, meta[tag], flush=True)
json.dump(meta, open(f"{OUT}/points.json", "w"), indent=2)
