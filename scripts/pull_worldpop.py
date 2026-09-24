"""
Pulls WorldPop population-density data for the 4 current-scope districts,
for exposure/damage-estimate use alongside the hazard-modeling data already
pulled (FABDEM, SoilGrids, Dynamic World, ERA5).

Verified directly via GEE (2026-09-14): WorldPop/GP/100m/pop, filtered to
country=IND, is real and available, but its GEE-hosted archive STOPS AT
2020 (checked: aggregate_array('year') returns 2000-2020, nothing newer).
This does NOT reach any of this project's actual flood events (2022, 2024,
2026) -- same "archive doesn't reach our events" pattern already seen with
IMDAA and MODIS Global Flood Database elsewhere in this project. Using
2020 as the best-available population baseline, stated as a limitation,
not corrected here (WorldPop's own site has more recent "unconstrained
global mosaics" for some years, but those are not in this GEE collection --
not fetched here, flagged as a possible manual follow-up).

Run:
    python scripts/pull_worldpop.py
"""
import json
import os

import ee
import pandas as pd
import requests

PROJECT_ID = "assamflood-508209"
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
SCOPE_GEOJSON = os.path.join(BASE_DIR, "data", "boundaries", "scope4_districts.geojson")
OUT_DIR = os.path.join(BASE_DIR, "data", "worldpop")
SCALE_M = 100  # WorldPop native resolution

DISTRICTS = ["Morigaon", "Sivasagar", "Nagaon", "Karimganj"]
YEAR = 2020  # latest available in this GEE collection, verified 2026-09-14


def load_district_geometry(district_name: str) -> dict:
    with open(SCOPE_GEOJSON, encoding="utf-8") as f:
        gj = json.load(f)
    aliases = {district_name.lower(), district_name.replace("Morigaon", "Marigaon").lower()}
    for feat in gj["features"]:
        if feat["properties"].get("dtname", "").lower() in aliases:
            return feat["geometry"]
    raise ValueError(f"District '{district_name}' not found in {SCOPE_GEOJSON}")


def pull_district(district, region):
    img = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filterMetadata("country", "equals", "IND")
        .filterMetadata("year", "equals", YEAR)
        .first()
        .clip(region)
    )

    stats = img.reduceRegion(
        reducer=ee.Reducer.sum().combine(ee.Reducer.mean(), sharedInputs=True).combine(
            ee.Reducer.max(), sharedInputs=True
        ),
        geometry=region, scale=SCALE_M, maxPixels=1e10, bestEffort=True,
    ).getInfo()

    dist_dir = os.path.join(OUT_DIR, district)
    os.makedirs(dist_dir, exist_ok=True)
    out_path = os.path.join(dist_dir, f"{district}_worldpop_{YEAR}.tif")
    url = img.getDownloadURL({
        "region": region, "scale": SCALE_M, "format": "GEO_TIFF", "crs": "EPSG:4326",
    })
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)

    total_pop = stats.get("population_sum")
    mean_density = stats.get("population_mean")
    max_density = stats.get("population_max")
    print(f"  {district}: total_pop(est)={total_pop:.0f} mean_density/cell={mean_density:.2f} "
          f"max_density/cell={max_density:.2f}  saved {out_path}")

    return {
        "district": district, "year": YEAR, "total_population_est": total_pop,
        "mean_pop_per_100m_cell": mean_density, "max_pop_per_100m_cell": max_density,
        "raster_path": out_path,
    }


def main():
    ee.Initialize(project=PROJECT_ID)
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for district in DISTRICTS:
        print(f"\n[{district}]")
        try:
            region = ee.Geometry(load_district_geometry(district))
            rows.append(pull_district(district, region))
        except Exception as e:
            print(f"  ERROR: {e}")

    summary_path = os.path.join(OUT_DIR, "summary.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\nDone. Summary saved to {summary_path}")
    print("\nCAVEAT: WorldPop's GEE collection archive stops at 2020 -- does NOT reach "
          "any of this project's flood events (2022/2024/2026). Best-available baseline only.")


if __name__ == "__main__":
    main()
