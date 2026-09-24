"""
Production GloFAS-ERA5 discharge pull for all 4 scope districts, at their
verified flood-event dates, as a HEC-RAS inflow-boundary candidate.

Requires a Copernicus CDS/EWDS account (~/.cdsapirc) with the
cems-glofas-historical dataset's Terms of Use accepted (one-time, via the
web UI) -- see manual-0040 in kb_chroma for the verification pull this
builds on.

CHANNEL-CELL SELECTION (a real limitation, not hidden): GloFAS resolves
rivers at 0.05 deg (~5-6km) and discharge varies by an order of magnitude
between adjacent cells depending on whether a cell sits on the modelled
channel (confirmed in manual-0040's 6x6 test box). This script picks the
grid cell with the MAX pre-flood discharge within a small box around each
named gauge/town as a proxy for "the channel cell" (channels carry
elevated baseflow even before a flood; overland cells don't). This is a
pragmatic heuristic, not the rigorous method (which would use GloFAS's
own upstream-area ancillary layer to confirm the channel cell directly) --
flagged as a follow-up, not resolved here.

REACHES/EVENTS (dates from TASK.md's cross-verified peak dates):
  Nagaon    - Kopili at Kampur      (26.20N, 92.63E)  - 2022 Kopili-breach
  Morigaon  - Kopili at Dharamtul   (26.16N, 92.35E)  - 2022 Kopili-breach (shared event)
  Sivasagar - Dikhow at Nazira      (26.92N, 94.73E)  - 2026 Dikhow record flood
  Karimganj - Kushiyara at Karimganj town (24.87N, 92.35E) - 2022 AND 2024 events
"""
import cdsapi
import xarray as xr
import pandas as pd

client = cdsapi.Client()
DATASET = "cems-glofas-historical"

REACHES = [
    # district, river, gauge, lat, lon, pre_date, during_date
    ("Nagaon", "Kopili", "Kampur", 26.20, 92.63, "2022-06-01", "2022-06-30"),
    ("Morigaon", "Kopili", "Dharamtul", 26.16, 92.35, "2022-06-01", "2022-06-30"),
    ("Sivasagar", "Dikhow", "Nazira", 26.92, 94.73, "2026-07-10", "2026-07-20"),
    ("Karimganj", "Kushiyara", "Karimganj town", 24.87, 92.35, "2022-06-01", "2022-06-21"),
    ("Karimganj", "Kushiyara", "Karimganj town", 24.87, 92.35, "2024-06-01", "2024-06-23"),
]

BOX_DEG = 0.15  # gives a 6x6 grid at 0.05 deg resolution, same as the verification pull


def pull_one(lat, lon, date_str, tag):
    y, m, d = date_str.split("-")
    area = [lat + BOX_DEG, lon - BOX_DEG, lat - BOX_DEG, lon + BOX_DEG]
    # consolidated (final ERA5) lags by months; recent dates need "intermediate" (ERA5T)
    product_type = "intermediate" if int(y) >= 2026 else "consolidated"
    request = {
        "system_version": ["version_4_0"],
        "hydrological_model": ["lisflood"],
        "product_type": [product_type],
        "timespan": ["time_mean"],
        "variable": ["average_river_discharge_in_the_last_24_hours"],
        "year": [y],
        "month": [m],
        "day": [d],
        "data_format": "grib2",
        "download_format": "unarchived",
        "area": area,
    }
    target = f"data/glofas_discharge/{tag}_{y}{m}{d}.grib"
    client.retrieve(DATASET, request, target)
    return target


def main():
    import os
    os.makedirs("data/glofas_discharge", exist_ok=True)
    rows = []
    for district, river, gauge, lat, lon, pre_date, during_date in REACHES:
        tag = f"{district.lower()}_{gauge.lower().replace(' ', '')}"
        print(f"== {district} / {river} at {gauge} ({pre_date} -> {during_date}) ==")

        pre_file = pull_one(lat, lon, pre_date, f"{tag}_pre")
        during_file = pull_one(lat, lon, during_date, f"{tag}_during")

        ds_pre = xr.open_dataset(pre_file, engine="cfgrib")
        ds_during = xr.open_dataset(during_file, engine="cfgrib")

        # channel cell = argmax discharge in the PRE reading (proxy for baseflow channel)
        pre_vals = ds_pre.avg_dis
        flat_idx = int(pre_vals.argmax())
        lat_i, lon_i = divmod(flat_idx, pre_vals.shape[1])
        chan_lat = float(pre_vals.latitude[lat_i])
        chan_lon = float(pre_vals.longitude[lon_i])

        pre_val = float(pre_vals.isel(latitude=lat_i, longitude=lon_i))
        during_val = float(
            ds_during.avg_dis.sel(latitude=chan_lat, longitude=chan_lon, method="nearest")
        )

        rows.append({
            "district": district, "river": river, "gauge": gauge,
            "gauge_lat": lat, "gauge_lon": lon,
            "channel_cell_lat": chan_lat, "channel_cell_lon": chan_lon,
            "pre_date": pre_date, "pre_discharge_m3s": round(pre_val, 2),
            "during_date": during_date, "during_discharge_m3s": round(during_val, 2),
            "ratio": round(during_val / pre_val, 2) if pre_val else None,
        })
        print(f"   channel cell ({chan_lat:.2f},{chan_lon:.2f}): "
              f"pre={pre_val:.1f} m3/s -> during={during_val:.1f} m3/s")

    df = pd.DataFrame(rows)
    out_csv = "data/glofas_discharge/discharge_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved summary: {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
