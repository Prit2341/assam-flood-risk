"""
Pulls ISRIC SoilGrids topsoil (0-5cm) properties for the current 4-district
scope (Morigaon, Sivasagar, Nagaon, Karimganj), clipped to each district's
real polygon (data/boundaries/scope4_districts.geojson).

Confirmed available via GEE's community catalog under
projects/soilgrids-isric/{property}_mean, each with 6 depth-band bands
(0-5, 5-15, 15-30, 30-60, 60-100, 100-200 cm). Topsoil (0-5cm) is pulled
here since it's what matters most for surface infiltration/runoff
(curve-number / Green-Ampt style parameters), not deep-profile properties.

Variables pulled (mean predictions):
  - sand, silt, clay (% x10, SoilGrids native units - divide by 10 for %)
  - bdod: bulk density (cg/cm3 x10 - divide by 100 for g/cm3... see ISRIC
    docs for exact scale factors before using in modeling, NOT corrected
    here, this is a raw pull)
  - soc: soil organic carbon (dg/kg)
  - cfvo: coarse fragments volumetric (cm3/100cm3, i.e. %)

Run:
    python scripts/pull_soilgrids.py
"""
import json
import os

import ee
import requests

PROJECT_ID = "assamflood-508209"
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
SCOPE_GEOJSON = os.path.join(BASE_DIR, "data", "boundaries", "scope4_districts.geojson")
OUT_DIR = os.path.join(BASE_DIR, "data", "soilgrids")
SCALE_M = 250  # SoilGrids native resolution

DISTRICTS = ["Morigaon", "Sivasagar", "Nagaon", "Karimganj"]
PROPERTIES = ["sand", "clay", "silt", "bdod", "soc", "cfvo"]
DEPTH_BAND = "0-5cm"


def load_district_geometry(district_name: str) -> dict:
    with open(SCOPE_GEOJSON, encoding="utf-8") as f:
        gj = json.load(f)
    aliases = {district_name.lower(), district_name.replace("Morigaon", "Marigaon").lower()}
    for feat in gj["features"]:
        if feat["properties"].get("dtname", "").lower() in aliases:
            return feat["geometry"]
    raise ValueError(f"District '{district_name}' not found in {SCOPE_GEOJSON}")


def pull_property(prop, region):
    band = f"{prop}_{DEPTH_BAND}_mean"
    img = ee.Image(f"projects/soilgrids-isric/{prop}_mean").select(band).clip(region)
    stats = img.reduceRegion(
        reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), sharedInputs=True),
        geometry=region, scale=SCALE_M, maxPixels=1e10, bestEffort=True,
    ).getInfo()
    return img, stats


def pull_district(district, region):
    dist_dir = os.path.join(OUT_DIR, district)
    os.makedirs(dist_dir, exist_ok=True)
    rows = []
    for prop in PROPERTIES:
        try:
            img, stats = pull_property(prop, region)
            out_path = os.path.join(dist_dir, f"{district}_{prop}_{DEPTH_BAND}.tif")
            url = img.getDownloadURL({
                "region": region, "scale": SCALE_M, "format": "GEO_TIFF", "crs": "EPSG:4326",
            })
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(r.content)
            print(f"  {prop}: {stats}")
            rows.append({"district": district, "property": prop, "depth": DEPTH_BAND, **stats,
                         "raster_path": out_path})
        except Exception as e:
            print(f"  {prop}: ERROR {e}")
    return rows


def main():
    ee.Initialize(project=PROJECT_ID)
    os.makedirs(OUT_DIR, exist_ok=True)
    all_rows = []
    for district in DISTRICTS:
        print(f"\n[{district}]")
        region = ee.Geometry(load_district_geometry(district))
        all_rows.extend(pull_district(district, region))

    import pandas as pd
    summary_path = os.path.join(OUT_DIR, "summary.csv")
    pd.DataFrame(all_rows).to_csv(summary_path, index=False)
    print(f"\nDone. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
