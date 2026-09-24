"""
Pull 32 of the 33 A10-A42 forecasting variables (Arunachal Pradesh spec,
reused here) for one Assam district, sourced entirely from GEE's
ECMWF/ERA5/HOURLY collection (single consistent source, no IMDAA/MOSDAC/
CFSR patchwork). Area-averaged over the REAL district polygon (from
data/boundaries/assam_districts.geojson, all 35 districts), not a bbox.

A42 (convection-permitting rainfall at <=3km) is NOT included - it cannot
come from ERA5, or from any downloadable reanalysis/satellite product, at
any resolution. That variable would require running a regional
convection-permitting model (e.g. WRF) yourself - a modeling task, not a
data pull. Confirmed this is a genuine ceiling, not a missed dataset.

Prerequisite: a Google Cloud project registered for Earth Engine access
(see https://code.earthengine.google.com/register). Set PROJECT_ID below.

Usage:
    python pull_era5_gee.py --district Karimganj --start 2024-06-10 --end 2024-06-26
    python pull_era5_gee.py --district Morigaon  --start 2022-06-05 --end 2022-07-10
    python pull_era5_gee.py --district Sivasagar --start 2026-07-01 --end 2026-08-25
    python pull_era5_gee.py --district Nagaon    --start 2022-06-05 --end 2022-07-10
"""
import argparse
import json
import os

import ee
import pandas as pd

PROJECT_ID = "assamflood-508209"

DISTRICTS_GEOJSON = os.path.join(
    os.path.dirname(__file__), "..", "data", "boundaries", "assam_districts.geojson"
)

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
    for feat in gj["features"]:
        # LGD dataset spells Morigaon as "Marigaon" - try both
        name = feat["properties"].get("dtname", "")
        if name.lower() in (district_name.lower(), district_name.replace("Morigaon", "Marigaon").lower()):
            return feat["geometry"]
    raise ValueError(f"District '{district_name}' not found in {DISTRICTS_GEOJSON}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--district", required=True, help="District name, e.g. Karimganj")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD (exclusive)")
    parser.add_argument("--out", default=None, help="Output CSV path (default: data/era5_<district>_<start>.csv)")
    args = parser.parse_args()

    ee.Initialize(project=PROJECT_ID)

    print(f"Pulling {len(BANDS)} ERA5 bands (32 of 33 A10-A42 spec variables; "
          f"A42 convection-permitting <=3km rainfall excluded - not derivable "
          f"from ERA5 at any resolution, a modeling task not a data pull)")

    geom = load_district_geometry(args.district)
    region = ee.Geometry(geom)

    col = (
        ee.ImageCollection("ECMWF/ERA5/HOURLY")
        .filterDate(args.start, args.end)
        .filterBounds(region)
        .select(BANDS)
    )

    def reduce_image(img):
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=27830, maxPixels=1e9
        )
        return ee.Feature(None, stats).set("system:time_start", img.get("system:time_start"))

    fc = col.map(reduce_image)
    result = fc.getInfo()

    rows = []
    for f in result["features"]:
        props = f["properties"]
        ts = props.pop("system:time_start")
        props["timestamp"] = pd.to_datetime(ts, unit="ms")
        rows.append(props)

    if not rows:
        print("No data returned - check the district name and date range.")
        return

    df = pd.DataFrame(rows).sort_values("timestamp")

    out_path = args.out or os.path.join(
        os.path.dirname(__file__), "..", "data",
        f"era5_{args.district.lower()}_{args.start.replace('-', '')}.csv",
    )
    df.to_csv(out_path, index=False)
    print(f"Pulled {len(df)} hourly timesteps, {len(BANDS)} variables for {args.district}")
    print(f"Saved to {out_path}")
    print(df[["timestamp", "total_precipitation", "temperature_2m"]].head())


if __name__ == "__main__":
    main()
