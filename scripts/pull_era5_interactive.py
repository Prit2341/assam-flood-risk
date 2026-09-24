"""
Interactive ERA5 puller - asks for district and year range, then downloads
32 of the 33 A10-A42 forecasting variables from GEE's ECMWF/ERA5/HOURLY
collection, area-averaged over the district's real polygon
(data/boundaries/assam_districts.geojson).

Pulls ONE MONTH at a time, saves each month's CSV immediately, and SKIPS
any month already downloaded - safe to stop (Ctrl+C) and rerun any time,
it resumes instead of re-downloading.

If start year == end year, only that one year is downloaded.

A42 (convection-permitting rainfall at <=3km) is NOT included - no
downloadable product exists at that resolution from any source; it would
require running a regional convection-permitting model (e.g. WRF) yourself,
a modeling task, not a data pull.

Run:
    python pull_era5_interactive.py
"""
import calendar
import json
import os
import time

import ee
import pandas as pd

PROJECT_ID = "assamflood-508209"

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DISTRICTS_GEOJSON = os.path.join(BASE_DIR, "data", "boundaries", "assam_districts.geojson")
OUT_DIR = os.path.join(BASE_DIR, "data", "era5_longrecord")

BANDS = [
    "total_precipitation",
    "convective_available_potential_energy",
    "convective_inhibition",
    "total_column_water_vapour",
    "temperature_2m",
    "dewpoint_temperature_2m",
    "surface_pressure",
    "mean_sea_level_pressure",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "u_component_of_wind_100m",
    "v_component_of_wind_100m",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "boundary_layer_height",
    "instantaneous_10m_wind_gust",
    "skin_temperature",
    "sea_surface_temperature",
    "low_cloud_cover",
    "medium_cloud_cover",
    "high_cloud_cover",
    "total_cloud_cover",
    "cloud_base_height",
    "evaporation",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
    "mean_surface_direct_short_wave_radiation_flux",
]


def load_district_geometry(district_name: str) -> dict:
    with open(DISTRICTS_GEOJSON, encoding="utf-8") as f:
        gj = json.load(f)
    aliases = {district_name.lower(), district_name.replace("Morigaon", "Marigaon").lower()}
    available = []
    for feat in gj["features"]:
        name = feat["properties"].get("dtname", "")
        available.append(name)
        if name.lower() in aliases:
            return feat["geometry"]
    raise ValueError(
        f"District '{district_name}' not found. Available districts:\n"
        + ", ".join(sorted(available))
    )


def pull_month(region, year, month, tries=4):
    start = f"{year}-{month:02d}-01"
    end_excl = f"{year + 1}-01-01" if month == 12 else f"{year}-{month + 1:02d}-01"

    col = (
        ee.ImageCollection("ECMWF/ERA5/HOURLY")
        .filterDate(start, end_excl)
        .filterBounds(region)
        .select(BANDS)
    )

    def reduce_image(img):
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=27830, maxPixels=1e9
        )
        return ee.Feature(None, stats).set("system:time_start", img.get("system:time_start"))

    last_err = None
    for attempt in range(tries):
        try:
            fc = col.map(reduce_image)
            result = fc.getInfo()
            rows = []
            for f in result["features"]:
                props = f["properties"]
                ts = props.pop("system:time_start", None)
                if ts is None:
                    continue
                props["timestamp"] = pd.to_datetime(ts, unit="ms")
                rows.append(props)
            return pd.DataFrame(rows).sort_values("timestamp") if rows else pd.DataFrame()
        except Exception as e:
            last_err = e
            wait = 5 * (attempt + 1)
            print(f"    attempt {attempt + 1}/{tries} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to pull {year}-{month:02d} after {tries} attempts: {last_err}")


def main():
    district = input("District name (e.g. Karimganj, Morigaon, Sivasagar, Nagaon): ").strip()
    start_year = int(input("Start year (e.g. 2024): ").strip())
    end_year = int(input("End year (same as start year to download just one year): ").strip())

    if end_year < start_year:
        print(f"End year ({end_year}) is before start year ({start_year}) - swapping them.")
        start_year, end_year = end_year, start_year

    ee.Initialize(project=PROJECT_ID)
    region = ee.Geometry(load_district_geometry(district))

    dist_dir = os.path.join(OUT_DIR, district)
    os.makedirs(dist_dir, exist_ok=True)

    total_months = (end_year - start_year + 1) * 12
    print(f"\nPulling {district}, {start_year} to {end_year} "
          f"({total_months} months, {len(BANDS)} variables/month)")
    print("Already-downloaded months are skipped automatically - safe to Ctrl+C and rerun.\n")

    done = 0
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            out_path = os.path.join(dist_dir, f"{district}_{year}_{month:02d}.csv")
            done += 1
            if os.path.exists(out_path):
                print(f"[{done}/{total_months}] {year}-{month:02d}: already done, skipping")
                continue
            print(f"[{done}/{total_months}] {year}-{month:02d}: pulling...")
            df = pull_month(region, year, month)
            if df.empty:
                print(f"    no data available yet for {year}-{month:02d} (future month "
                      f"or beyond ERA5's ~5-day lag) - skipping, not writing an empty file")
                continue
            df.to_csv(out_path, index=False)
            print(f"    saved {len(df)} rows -> {out_path}")

    print(f"\nDone. All files in {dist_dir}")


if __name__ == "__main__":
    main()
