"""
Production GloFAS-ERA5 discharge HYDROGRAPH pull (not just pre/during
snapshots) for all 4 scope districts, as the real HEC-RAS unsteady inflow
boundary condition -- see manual-0042 (Prit's decision: GloFAS is now the
real boundary, not a plausibility check, since no discharge alternative
survived the heavy search in manual-0039/0040/0041).

CHANNEL-CELL SELECTION -- HONEST LIMITATION, READ BEFORE TRUSTING THIS:
the rigorous method is GloFAS's own upstream-area ancillary layer
(uparea_glofas_v4_0.nc, confirmed to exist, hosted at
confluence.ecmwf.int/download/attachments/242067380/...). It could NOT be
downloaded from this environment -- confirmed via both curl and a real
Playwright browser fetch, both got a genuine ECMWF-wide 403 "Service
forbidden" (nginx WAF), not a timeout or DNS issue. This is a real access
block on ECMWF's side from this environment, not given up on lightly.
FALLBACK USED INSTEAD: the channel cell is the one with the highest MEAN
discharge across the whole pulled multi-day window (more robust than the
single-day-argmax heuristic used in the first pass, since a channel cell
carries persistently higher discharge across many days, not just one).
Still not as rigorous as the real upstream-area layer -- state this
plainly if this discharge series is cited in the thesis.

WINDOW: 14 days before the verified onset date through 7 days after the
verified peak date, per reach (covers the full rise-peak-recession shape
needed for an unsteady HEC-RAS boundary, not just two snapshots).
"""
import os
import cdsapi
import xarray as xr
import pandas as pd
from datetime import date, timedelta

client = cdsapi.Client()
DATASET = "cems-glofas-historical"
BOX_DEG = 0.15

# district, river, gauge, lat, lon, onset_date, peak_date (from TASK.md verified dates)
REACHES = [
    ("Nagaon", "Kopili", "Kampur", 26.20, 92.63, "2022-06-19", "2022-06-30"),
    ("Morigaon", "Kopili", "Dharamtul", 26.16, 92.35, "2022-06-19", "2022-06-30"),
    ("Sivasagar", "Dikhow", "Nazira", 26.92, 94.73, "2026-07-19", "2026-07-20"),
    ("Karimganj", "Kushiyara", "Karimganj town", 24.87, 92.35, "2022-06-19", "2022-06-21"),
    ("Karimganj", "Kushiyara", "Karimganj town", 24.87, 92.35, "2024-06-19", "2024-06-25"),
]


def daterange(d0, d1):
    days = []
    d = d0
    while d <= d1:
        days.append(d)
        d += timedelta(days=1)
    return days


def pull_window(lat, lon, days, tag):
    """One CDS request covering all days in the window (grouped by month,
    since year/month/day are each single values per GRIB but day accepts
    a list within one month)."""
    area = [lat + BOX_DEG, lon - BOX_DEG, lat - BOX_DEG, lon + BOX_DEG]
    by_month = {}
    for d in days:
        by_month.setdefault((d.year, d.month), []).append(f"{d.day:02d}")

    files = []
    for (y, m), day_list in by_month.items():
        product_type = "intermediate" if y >= 2026 else "consolidated"
        request = {
            "system_version": ["version_4_0"],
            "hydrological_model": ["lisflood"],
            "product_type": [product_type],
            "timespan": ["time_mean"],
            "variable": ["average_river_discharge_in_the_last_24_hours"],
            "year": [str(y)],
            "month": [f"{m:02d}"],
            "day": day_list,
            "data_format": "grib2",
            "download_format": "unarchived",
            "area": area,
        }
        target = f"data/glofas_discharge/{tag}_{y}{m:02d}.grib"
        client.retrieve(DATASET, request, target)
        files.append(target)
    return files


def main():
    os.makedirs("data/glofas_discharge", exist_ok=True)
    all_rows = []

    for district, river, gauge, lat, lon, onset_str, peak_str in REACHES:
        tag = f"{district.lower()}_{gauge.lower().replace(' ', '')}"
        onset = date.fromisoformat(onset_str)
        peak = date.fromisoformat(peak_str)
        window_start = onset - timedelta(days=14)
        window_end = peak + timedelta(days=7)
        days = daterange(window_start, window_end)

        print(f"== {district} / {river} at {gauge}: {window_start} -> {window_end} "
              f"({len(days)} days) ==")

        files = pull_window(lat, lon, days, f"{tag}_hydro")

        ds = xr.open_mfdataset(files, engine="cfgrib", combine="nested", concat_dim="time")
        da = ds.avg_dis.load()

        # channel cell = highest MEAN discharge across the whole window (robust fallback,
        # see module docstring for why the rigorous upstream-area method wasn't available)
        mean_by_cell = da.mean(dim="time")
        flat_idx = int(mean_by_cell.argmax())
        lat_i, lon_i = divmod(flat_idx, mean_by_cell.shape[1])
        chan_lat = float(mean_by_cell.latitude[lat_i])
        chan_lon = float(mean_by_cell.longitude[lon_i])

        series = da.isel(latitude=lat_i, longitude=lon_i).to_series().sort_index()
        for t, val in series.items():
            all_rows.append({
                "district": district, "river": river, "gauge": gauge,
                "channel_cell_lat": chan_lat, "channel_cell_lon": chan_lon,
                "date": pd.Timestamp(t).date().isoformat(),
                "discharge_m3s": round(float(val), 2),
            })

        peak_val = series.max()
        base_val = series.min()
        print(f"   channel cell ({chan_lat:.2f},{chan_lon:.2f}): "
              f"{len(series)} days, min={base_val:.1f} max={peak_val:.1f} m3/s")

    df = pd.DataFrame(all_rows)
    out_csv = "data/glofas_discharge/discharge_hydrographs.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved hydrographs: {out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
