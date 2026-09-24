"""
Full long-record pull: 32 of the 33 A10-A42 forecasting variables, for one or
all 4 Assam districts (Cachar-era Barak Valley pick superseded - current
scope is Morigaon, Sivasagar, Nagaon, Karimganj), sourced from GEE's
ECMWF/ERA5/HOURLY collection, area-averaged over each district's real
polygon (data/boundaries/assam_districts.geojson).

WHY CHUNKED: a single GEE interactive request (.getInfo() on a mapped
ImageCollection) cannot handle ~219,000 hourly images (25 years) in one
call - it will time out or hit payload limits. This script pulls ONE MONTH
at a time per district, saves each month to its own CSV immediately, and
SKIPS any month whose file already exists - so if this crashes, loses
network, or you Ctrl+C it, just rerun the same command and it resumes
exactly where it left off instead of re-downloading everything.

A42 (convection-permitting rainfall at <=3km) is NOT included - see
pull_era5_gee.py's docstring for why (a modeling task, not a data pull -
no downloadable product exists at that resolution from any source).

Usage:
    # one district, default 2000-2025 (26 years, matches Arunachal Pradesh's
    # own ERA5 record length)
    python pull_era5_gee_longrecord.py --district Karimganj

    # all 4 current-scope districts
    python pull_era5_gee_longrecord.py --all

    # custom range
    python pull_era5_gee_longrecord.py --district Morigaon --start-year 2010 --end-year 2025

Output: data/era5_longrecord/<district>/<district>_<YYYY>_<MM>.csv (one file
per district per month). Concatenate afterward with pandas if you need one
big file per district - deliberately kept as separate month files so a
crash never corrupts previously-downloaded months.
"""
import argparse
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

CURRENT_SCOPE_DISTRICTS = ["Morigaon", "Sivasagar", "Nagaon", "Karimganj"]

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
    for feat in gj["features"]:
        if feat["properties"].get("dtname", "").lower() in aliases:
            return feat["geometry"]
    raise ValueError(f"District '{district_name}' not found in {DISTRICTS_GEOJSON}")


def pull_month(region, year, month, tries=4):
    start = f"{year}-{month:02d}-01"
    last_day = calendar.monthrange(year, month)[1]
    end = f"{year}-{month:02d}-{last_day:02d}"
    # end date is exclusive in filterDate's usual convention when combined with next-day; use next month start instead
    if month == 12:
        end_excl = f"{year + 1}-01-01"
    else:
        end_excl = f"{year}-{month + 1:02d}-01"

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


def pull_district(district: str, start_year: int, end_year: int):
    print(f"=== {district}: {start_year}-01 through {end_year}-12 ===")
    region = ee.Geometry(load_district_geometry(district))
    dist_dir = os.path.join(OUT_DIR, district)
    os.makedirs(dist_dir, exist_ok=True)

    total_months = (end_year - start_year + 1) * 12
    done = 0
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            out_path = os.path.join(dist_dir, f"{district}_{year}_{month:02d}.csv")
            done += 1
            if os.path.exists(out_path):
                print(f"  [{done}/{total_months}] {year}-{month:02d}: already done, skipping")
                continue
            print(f"  [{done}/{total_months}] {year}-{month:02d}: pulling...")
            df = pull_month(region, year, month)
            df.to_csv(out_path, index=False)
            print(f"    saved {len(df)} rows -> {out_path}")
    print(f"=== {district} complete ===\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--district", help="Single district name, e.g. Karimganj")
    parser.add_argument("--all", action="store_true", help="Pull all 4 current-scope districts")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    if not args.district and not args.all:
        parser.error("specify --district <name> or --all")

    ee.Initialize(project=PROJECT_ID)
    os.makedirs(OUT_DIR, exist_ok=True)

    districts = CURRENT_SCOPE_DISTRICTS if args.all else [args.district]
    for d in districts:
        pull_district(d, args.start_year, args.end_year)


if __name__ == "__main__":
    main()
