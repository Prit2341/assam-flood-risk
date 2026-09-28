"""
Pull GloFAS-ERA5 daily discharge for 2015-2024 at all 4 gauge points
and compute Q90 thresholds for Phase 2 dynamic susceptibility features.

Q90 = 90th percentile of daily discharge over the 10-year climatology.
Q_anomaly = (Q_current - Q90) / Q90  -->  dynamic feature for XGBoost Phase 2.

Gauge coordinates from scripts/pull_glofas_hydrograph.py (verified events).
Channel-cell method: argmax of mean discharge over the first pulled year in
a 0.15-deg box around the named gauge -- same pragmatic fallback as hydrograph
script (rigorous upstream-area layer blocked by ECMWF 403).

Outputs:
  data/glofas_q90/timeseries_<gauge>.csv   -- daily Q 2015-2024 per gauge
  data/glofas_q90/q90_thresholds.csv       -- Q90 + Q50 + Q95 per gauge
"""
import os, json
from datetime import date
import cdsapi
import xarray as xr
import pandas as pd
import numpy as np

# ── config ────────────────────────────────────────────────────────────────────
client  = cdsapi.Client()
DATASET = "cems-glofas-historical"
BOX_DEG = 0.15        # half-width of spatial search box (degrees)
YEARS   = list(range(2015, 2025))   # 2015-2024 inclusive; all consolidated

GAUGES = [
    # (tag, district, river, gauge_name, lat, lon)
    ("nagaon",    "Nagaon",    "Kopili",    "Kampur",         26.20, 92.63),
    ("morigaon",  "Morigaon",  "Kopili",    "Dharamtul",      26.16, 92.35),
    ("sivasagar", "Sivasagar", "Dikhow",    "Nazira",         26.92, 94.73),
    ("karimganj", "Karimganj", "Kushiyara", "Karimganj town", 24.87, 92.35),
]

ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]
ALL_DAYS   = [f"{d:02d}" for d in range(1, 32)]   # CDS silently drops Feb 30 etc.

OUT_DIR = "data/glofas_q90"
os.makedirs(OUT_DIR, exist_ok=True)


def pull_year(tag, lat, lon, year):
    """Download one year of daily discharge in a small box. Returns local path."""
    area = [lat + BOX_DEG, lon - BOX_DEG, lat - BOX_DEG, lon + BOX_DEG]
    product_type = "intermediate" if year >= 2026 else "consolidated"
    request = {
        "system_version":     ["version_4_0"],
        "hydrological_model": ["lisflood"],
        "product_type":       [product_type],
        "timespan":           ["time_mean"],
        "variable":           ["average_river_discharge_in_the_last_24_hours"],
        "year":               [str(year)],
        "month":              ALL_MONTHS,
        "day":                ALL_DAYS,
        "data_format":        "grib2",
        "download_format":    "unarchived",
        "area":               area,
    }
    target = f"{OUT_DIR}/{tag}_{year}.grib"
    if os.path.exists(target):
        print(f"  [skip] {target} already exists")
        return target
    print(f"  pulling {year}...", flush=True)
    client.retrieve(DATASET, request, target)
    return target


def extract_series(files, lat, lon, tag):
    """
    Open all GRIB files for one gauge, identify the channel cell (max mean Q),
    return daily timeseries as a pd.Series indexed by date.
    """
    import cfgrib
    # open each file independently (cfgrib.open_datasets handles multi-message GRIBs)
    # then concat along time
    arrays = []
    for f in files:
        dsets = cfgrib.open_datasets(f)
        # first dataset is avg_dis
        arrays.append(dsets[0]["avg_dis"])
    da = xr.concat(arrays, dim="time").load()   # dims: time, latitude, longitude

    # channel cell = cell with highest MEAN discharge across the full period
    mean_map = da.mean(dim="time")
    flat_idx = int(mean_map.values.argmax())
    n_lon = mean_map.shape[1]
    lat_i, lon_i = divmod(flat_idx, n_lon)
    chan_lat = float(mean_map.latitude[lat_i])
    chan_lon = float(mean_map.longitude[lon_i])
    print(f"  channel cell: ({chan_lat:.3f}, {chan_lon:.3f})  "
          f"[named gauge: ({lat:.2f}, {lon:.2f})]")

    series = (
        da.isel(latitude=lat_i, longitude=lon_i)
          .to_series()
          .rename("discharge_m3s")
    )
    series.index = pd.to_datetime(series.index).normalize()
    series = series.sort_index()
    return series, chan_lat, chan_lon


def main():
    q90_rows = []
    channel_cells = {}

    for tag, district, river, gauge, lat, lon in GAUGES:
        print(f"\n{'='*60}")
        print(f"{district} / {river} at {gauge}  ({lat:.2f}N, {lon:.2f}E)")
        print(f"{'='*60}")

        # ── download ──────────────────────────────────────────────────────────
        files = []
        for year in YEARS:
            f = pull_year(tag, lat, lon, year)
            files.append(f)

        # ── extract timeseries ────────────────────────────────────────────────
        series, chan_lat, chan_lon = extract_series(files, lat, lon, tag)
        channel_cells[tag] = {"lat": chan_lat, "lon": chan_lon}

        # ── save daily timeseries ─────────────────────────────────────────────
        ts_df = series.reset_index()
        ts_df.columns = ["date", "discharge_m3s"]
        ts_df["date"] = ts_df["date"].dt.date.astype(str)
        ts_path = f"{OUT_DIR}/timeseries_{tag}.csv"
        ts_df.to_csv(ts_path, index=False)
        print(f"  timeseries saved: {ts_path}  ({len(ts_df)} days)")

        # ── compute thresholds ────────────────────────────────────────────────
        vals = series.dropna().values
        q50 = float(np.percentile(vals, 50))
        q90 = float(np.percentile(vals, 90))
        q95 = float(np.percentile(vals, 95))
        q99 = float(np.percentile(vals, 99))
        print(f"  Q50={q50:.1f}  Q90={q90:.1f}  Q95={q95:.1f}  Q99={q99:.1f}  m³/s")
        print(f"  n_days={len(vals)}  min={vals.min():.1f}  max={vals.max():.1f}")

        q90_rows.append({
            "tag": tag, "district": district, "river": river, "gauge": gauge,
            "channel_lat": chan_lat, "channel_lon": chan_lon,
            "n_days": len(vals),
            "Q_min": round(float(vals.min()), 1),
            "Q_mean": round(float(vals.mean()), 1),
            "Q50": round(q50, 1),
            "Q90": round(q90, 1),
            "Q95": round(q95, 1),
            "Q99": round(q99, 1),
            "Q_max": round(float(vals.max()), 1),
            "period": f"{YEARS[0]}-{YEARS[-1]}",
        })

    # ── save thresholds table ─────────────────────────────────────────────────
    thresh_df = pd.DataFrame(q90_rows)
    thresh_path = f"{OUT_DIR}/q90_thresholds.csv"
    thresh_df.to_csv(thresh_path, index=False)
    print(f"\n{'='*60}")
    print("Q90 THRESHOLDS")
    print(thresh_df[["district", "river", "gauge", "Q50", "Q90", "Q95", "Q_max"]].to_string(index=False))
    print(f"\nSaved: {thresh_path}")

    # save channel-cell coords for downstream scripts
    cells_path = f"{OUT_DIR}/channel_cells.json"
    with open(cells_path, "w") as f:
        json.dump(channel_cells, f, indent=2)
    print(f"Channel cells: {cells_path}")

    print("\nDone. Next: use Q90 in Phase 2 dynamic susceptibility training.")
    print("  Q_anomaly = (Q_current - Q90) / Q90")


if __name__ == "__main__":
    main()
