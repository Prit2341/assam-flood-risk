"""
Pulls FABDEM (Forest And Buildings removed Copernicus DEM) for the current
4-district scope (Morigaon, Sivasagar, Nagaon, Karimganj), clipped to each
district's real polygon (data/boundaries/scope4_districts.geojson).

FABDEM was recommended over raw Copernicus GLO-30 on 2026-09-09 (see
TASK.md / KNOWLEDGE_BASE.md DEM vertical-accuracy literature check): raw
GLO-30 carries large positive-bias errors in vegetated/urban areas -
exactly the failure mode that matters on a flat floodplain - while FABDEM
(a bias-corrected GLO-30 derivative) ranked best in a 65-LiDAR-survey
validation study. Free for academic use (CC BY-NC-SA 4.0, Fathom/Univ. of
Bristol), available via GEE's community catalog at
projects/sat-io/open-datasets/FABDEM (19,011 tiles, band 'b1', mosaicked
here per district and clipped to the real polygon, not a bbox).

This is the first real per-district DEM pull for the current 4-district
scope - the only DEM data that existed before this (data/boundaries/
assam_dem_clip.tif) is a whole-state clip of unconfirmed source (no FABDEM
metadata tag), and barak_dem_*.tif is for the superseded 3-district scope.

Run:
    python scripts/pull_fabdem.py
"""
import json
import os

import ee
import requests

PROJECT_ID = "assamflood-508209"
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
SCOPE_GEOJSON = os.path.join(BASE_DIR, "data", "boundaries", "scope4_districts.geojson")
OUT_DIR = os.path.join(BASE_DIR, "data", "dem_fabdem")
SCALE_M = 30

DISTRICTS = ["Morigaon", "Sivasagar", "Nagaon", "Karimganj"]


def load_district_geometry(district_name: str) -> dict:
    with open(SCOPE_GEOJSON, encoding="utf-8") as f:
        gj = json.load(f)
    aliases = {district_name.lower(), district_name.replace("Morigaon", "Marigaon").lower()}
    for feat in gj["features"]:
        if feat["properties"].get("dtname", "").lower() in aliases:
            return feat["geometry"]
    raise ValueError(f"District '{district_name}' not found in {SCOPE_GEOJSON}")


def pull_district(district, region):
    fabdem = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
    dem = fabdem.filterBounds(region).mosaic().clip(region)

    # sanity stats before downloading anything
    stats = dem.reduceRegion(
        reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), sharedInputs=True),
        geometry=region, scale=SCALE_M, maxPixels=1e10, bestEffort=True,
    ).getInfo()
    print(f"  elevation stats (m): {stats}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{district}_fabdem_30m.tif")

    url = dem.getDownloadURL({
        "region": region, "scale": SCALE_M, "format": "GEO_TIFF", "crs": "EPSG:4326",
    })
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    print(f"  saved {out_path} ({len(r.content)/1e6:.1f} MB)")
    return stats


def main():
    ee.Initialize(project=PROJECT_ID)
    for district in DISTRICTS:
        print(f"\n[{district}]")
        region = ee.Geometry(load_district_geometry(district))
        try:
            pull_district(district, region)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone. Files in {OUT_DIR}")


if __name__ == "__main__":
    main()
