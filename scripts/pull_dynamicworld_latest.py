"""
Pulls the LATEST/current Dynamic World V1 composite for the 4 current-scope
districts -- a present-day baseline LULC layer, separate from the
event-window (pre/during-flood) pulls already done in
pull_dynamicworld_lulc.py. Requested explicitly (2026-09-14) as a distinct
"latest" layer, not a replacement for the event-window pulls.

Takes the per-pixel MODE of the discrete 'label' band across all Sentinel-2
revisits in the most recent LOOKBACK_DAYS window ending today, same
mode-reduction method as the event-window script (reduces noise from a
single date/cloud gap).

Run:
    python scripts/pull_dynamicworld_latest.py
"""
import datetime
import json
import os

import ee
import pandas as pd
import requests

PROJECT_ID = "assamflood-508209"
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DISTRICTS_GEOJSON = os.path.join(BASE_DIR, "data", "boundaries", "scope4_districts.geojson")
OUT_DIR = os.path.join(BASE_DIR, "data", "lulc_dynamicworld_latest")
SCALE_M = 10
DOWNLOAD_SCALE_M = 30
LOOKBACK_DAYS = 60  # composite window ending "today" (2026-09-14)

DISTRICTS = ["Morigaon", "Sivasagar", "Nagaon", "Karimganj"]

CLASS_NAMES = [
    "water", "trees", "grass", "flooded_vegetation", "crops",
    "shrub_and_scrub", "built", "bare", "snow_and_ice",
]


def load_district_geometry(district_name: str) -> dict:
    with open(DISTRICTS_GEOJSON, encoding="utf-8") as f:
        gj = json.load(f)
    aliases = {district_name.lower(), district_name.replace("Morigaon", "Marigaon").lower()}
    for feat in gj["features"]:
        if feat["properties"].get("dtname", "").lower() in aliases:
            return feat["geometry"]
    raise ValueError(f"District '{district_name}' not found in {DISTRICTS_GEOJSON}")


def pull_district(district, region, start, end):
    col = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(start, end)
        .filterBounds(region)
        .select("label")
    )
    n_images = col.size().getInfo()
    if n_images == 0:
        print(f"  {district}: no Dynamic World scenes found for {start} to {end}")
        return None

    mode_img = col.reduce(ee.Reducer.mode()).rename("label").clip(region)

    area_img = ee.Image.pixelArea().addBands(mode_img)
    stats = area_img.reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName="label"),
        geometry=region, scale=SCALE_M, maxPixels=1e10, bestEffort=True,
    ).getInfo()

    groups = stats.get("groups", [])
    total_area_m2 = sum(g["sum"] for g in groups)
    class_rows = []
    for g in groups:
        cls_idx = int(g["label"])
        cls_name = CLASS_NAMES[cls_idx] if 0 <= cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
        area_km2 = g["sum"] / 1e6
        pct = 100 * g["sum"] / total_area_m2 if total_area_m2 else 0.0
        class_rows.append({
            "district": district, "start": start, "end": end, "n_scenes": n_images,
            "class": cls_name, "area_km2": round(area_km2, 3), "pct": round(pct, 3),
        })

    dist_dir = os.path.join(OUT_DIR, district)
    os.makedirs(dist_dir, exist_ok=True)
    out_path = os.path.join(dist_dir, f"{district}_latest_{start}_to_{end}_label.tif")
    try:
        url = mode_img.getDownloadURL({
            "region": region, "scale": DOWNLOAD_SCALE_M, "format": "GEO_TIFF", "crs": "EPSG:4326",
        })
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
        print(f"  {district}: {n_images} scenes, saved {out_path} ({len(class_rows)} classes)")
    except Exception as e:
        print(f"  {district}: {n_images} scenes, stats OK but raster download FAILED ({e})")

    return class_rows


def main():
    ee.Initialize(project=PROJECT_ID)
    os.makedirs(OUT_DIR, exist_ok=True)

    end = datetime.date.today()
    start = end - datetime.timedelta(days=LOOKBACK_DAYS)
    start_s, end_s = start.isoformat(), end.isoformat()
    print(f"Latest Dynamic World composite window: {start_s} to {end_s}")

    all_rows = []
    for district in DISTRICTS:
        print(f"\n[{district}]")
        region = ee.Geometry(load_district_geometry(district))
        try:
            rows = pull_district(district, region, start_s, end_s)
        except Exception as e:
            print(f"  ERROR: {e}")
            rows = None
        if rows:
            all_rows.extend(rows)

    summary_path = os.path.join(OUT_DIR, "summary.csv")
    pd.DataFrame(all_rows).to_csv(summary_path, index=False)
    print(f"\nDone. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
