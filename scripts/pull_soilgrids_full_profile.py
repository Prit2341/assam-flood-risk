"""
Pulls the FULL 6-depth SoilGrids profile (extends pull_soilgrids.py, which
only pulled the 0-5cm topsoil layer) for the 4 current-scope districts, and
converts raw SoilGrids values to ISRIC's documented standard units -- see
TASK.md "NOT yet done: raw values not converted to ISRIC's standard units".

Conversion factors are ISRIC's own documented mapped-unit conversion table
(https://www.isric.org/explore/soilgrids/faq-soilgrids -- "Soil property
conversion factors"), NOT derived/assumed here:
  bdod (bulk density):        raw / 100  -> kg/dm3 (= g/cm3)
  cfvo (coarse fragments):    raw / 10   -> cm3/100cm3 (vol %)
  clay, sand, silt:           raw / 10   -> %
  soc (organic carbon):       raw / 10   -> g/kg

Run:
    python scripts/pull_soilgrids_full_profile.py
"""
import json
import os

import ee
import pandas as pd
import requests

PROJECT_ID = "assamflood-508209"
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
SCOPE_GEOJSON = os.path.join(BASE_DIR, "data", "boundaries", "scope4_districts.geojson")
OUT_DIR = os.path.join(BASE_DIR, "data", "soilgrids_full_profile")
SCALE_M = 250

DISTRICTS = ["Morigaon", "Sivasagar", "Nagaon", "Karimganj"]
DEPTHS = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]

# property -> ISRIC divisor to convert raw mapped value to standard unit
PROPERTIES = {
    "sand": {"divisor": 10, "unit": "%"},
    "clay": {"divisor": 10, "unit": "%"},
    "silt": {"divisor": 10, "unit": "%"},
    "bdod": {"divisor": 100, "unit": "kg/dm3 (g/cm3)"},
    "soc": {"divisor": 10, "unit": "g/kg"},
    "cfvo": {"divisor": 10, "unit": "cm3/100cm3 (vol%)"},
}


def load_district_geometry(district_name: str) -> dict:
    with open(SCOPE_GEOJSON, encoding="utf-8") as f:
        gj = json.load(f)
    aliases = {district_name.lower(), district_name.replace("Morigaon", "Marigaon").lower()}
    for feat in gj["features"]:
        if feat["properties"].get("dtname", "").lower() in aliases:
            return feat["geometry"]
    raise ValueError(f"District '{district_name}' not found in {SCOPE_GEOJSON}")


def pull_one(prop, depth, region, divisor, unit):
    band = f"{prop}_{depth}_mean"
    raw_img = ee.Image(f"projects/soilgrids-isric/{prop}_mean").select(band).clip(region)
    converted_img = raw_img.divide(divisor).rename(f"{prop}_{depth}_converted")

    raw_stats = raw_img.reduceRegion(
        reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), sharedInputs=True),
        geometry=region, scale=SCALE_M, maxPixels=1e10, bestEffort=True,
    ).getInfo()
    conv_stats = {k: (v / divisor if v is not None else v) for k, v in raw_stats.items()}
    return converted_img, raw_stats, conv_stats


def pull_district(district, region):
    dist_dir = os.path.join(OUT_DIR, district)
    os.makedirs(dist_dir, exist_ok=True)
    rows = []
    for prop, cfg in PROPERTIES.items():
        for depth in DEPTHS:
            try:
                img, raw_stats, conv_stats = pull_one(prop, depth, region, cfg["divisor"], cfg["unit"])
                out_path = os.path.join(dist_dir, f"{district}_{prop}_{depth}_converted.tif")
                url = img.getDownloadURL({
                    "region": region, "scale": SCALE_M, "format": "GEO_TIFF", "crs": "EPSG:4326",
                })
                r = requests.get(url, timeout=180)
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    f.write(r.content)
                print(f"  {prop} {depth}: raw_mean={raw_stats.get(prop + '_' + depth + '_mean_mean')} "
                      f"-> converted_mean={conv_stats.get(prop + '_' + depth + '_mean_mean')} {cfg['unit']}")
                rows.append({
                    "district": district, "property": prop, "depth": depth, "unit": cfg["unit"],
                    "divisor": cfg["divisor"],
                    "raw_mean": raw_stats.get(f"{prop}_{depth}_mean_mean"),
                    "converted_mean": conv_stats.get(f"{prop}_{depth}_mean_mean"),
                    "raw_min": raw_stats.get(f"{prop}_{depth}_mean_min"),
                    "converted_min": conv_stats.get(f"{prop}_{depth}_mean_min"),
                    "raw_max": raw_stats.get(f"{prop}_{depth}_mean_max"),
                    "converted_max": conv_stats.get(f"{prop}_{depth}_mean_max"),
                    "raster_path": out_path,
                })
            except Exception as e:
                print(f"  {prop} {depth}: ERROR {e}")
    return rows


def main():
    ee.Initialize(project=PROJECT_ID)
    os.makedirs(OUT_DIR, exist_ok=True)
    all_rows = []
    for district in DISTRICTS:
        print(f"\n[{district}]")
        region = ee.Geometry(load_district_geometry(district))
        all_rows.extend(pull_district(district, region))

    summary_path = os.path.join(OUT_DIR, "summary.csv")
    pd.DataFrame(all_rows).to_csv(summary_path, index=False)
    print(f"\nDone. Summary saved to {summary_path}")

    # Sanity check: sand+silt+clay should sum to ~100% per depth per district
    df = pd.DataFrame(all_rows)
    texture = df[df["property"].isin(["sand", "silt", "clay"])]
    check = texture.groupby(["district", "depth"])["converted_mean"].sum()
    print("\nSand+silt+clay sanity check (should be ~100%):")
    print(check)


if __name__ == "__main__":
    main()
