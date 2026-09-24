"""
Downloads the 2004-06-20 Morigaon MODIS flood event (GLOBAL_FLOOD_DB/
MODIS_EVENTS/V1, event id=2507) as a real local GeoTIFF, clipped to
Morigaon's actual district polygon - completing the MODIS-GFD finding from
manual-0010 (previously only measured via reduceRegion, never saved as a
raster file).

Run:
    python scripts/pull_modis_2004_morigaon.py
"""
import json
import os

import ee
import google.auth.transport.requests
import requests

PROJECT_ID = "assamflood-508209"
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DISTRICTS_GEOJSON = os.path.join(BASE_DIR, "data", "boundaries", "assam_districts.geojson")
OUT_DIR = os.path.join(BASE_DIR, "data", "modis_gfd")


def main():
    ee.Initialize(project=PROJECT_ID)
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(DISTRICTS_GEOJSON, encoding="utf-8") as f:
        gj = json.load(f)
    morigaon_geom = next(
        feat["geometry"] for feat in gj["features"]
        if feat["properties"].get("dtname", "").lower() in ("morigaon", "marigaon")
    )
    region = ee.Geometry(morigaon_geom)

    img = ee.Image(
        ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1")
        .filter(ee.Filter.eq("id", 2507))
        .first()
    )
    clipped = img.select(
        ["flooded", "duration", "clear_views", "clear_perc", "jrc_perm_water"]
    ).clip(region)

    url = clipped.getDownloadURL({"scale": 250, "region": region, "format": "GEO_TIFF"})

    creds = ee.data.get_persistent_credentials()
    creds.refresh(google.auth.transport.requests.Request())
    resp = requests.get(url, headers={"Authorization": f"Bearer {creds.token}"}, timeout=120)
    resp.raise_for_status()

    out_path = os.path.join(OUT_DIR, "Morigaon_2004-06-20_MODIS_event2507.tif")
    with open(out_path, "wb") as f:
        f.write(resp.content)

    print(f"Saved {len(resp.content)} bytes -> {out_path}")


if __name__ == "__main__":
    main()
