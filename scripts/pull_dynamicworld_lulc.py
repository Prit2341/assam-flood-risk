"""
Pulls Dynamic World V1 (GOOGLE/DYNAMICWORLD/V1) 10m near-real-time LULC for
the 4 current-scope districts (Morigaon, Sivasagar, Nagaon, Karimganj),
clipped to each district's real polygon from data/boundaries/assam_districts.geojson.

Candidate for replacing/supplementing the originally-planned static ESA
WorldCover (2020/2021 snapshot only) - see KNOWLEDGE_BASE.md entry
2026-09-12 for the literature backing this choice:
  - Brown et al. 2022, Scientific Data 9:251 (dataset paper, 73.8% overall
    accuracy, 9 classes, updated every 2-5 days since 2015-06-27)
  - Land 13(11):1929 (2024) - direct precedent using Dynamic World inside
    GEE for a near-real-time flood hazard assessment tool
  - PNNL/OSTI comparison paper - Dynamic World OVERESTIMATES flooded/mixed
    vegetation classes relative to WorldCover; flagged, not yet corrected
    for here.

For each (district, tag, start, end) window, takes the per-pixel MODE of
the discrete 'label' band across all Sentinel-2 revisits in that window
(reduces noise/misclassification from a single date), clips to the
district polygon, and:
  1. Saves the clipped label raster as a local GeoTIFF.
  2. Computes per-class pixel counts/percentages via a grouped reduceRegion
     and appends a row per class to data/lulc_dynamicworld/summary.csv.

Windows are +/-15 days around each district's already-verified flood event
date (manual-0013 cross-verified dates), one "pre" window ending before
onset and one "during" window spanning onset-to-peak - mirrors the
pre/during structure already used for GFM validation (pull_gfm_validation.py).

Run:
    python scripts/pull_dynamicworld_lulc.py
"""
import json
import os

import ee
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds

PROJECT_ID = "assamflood-508209"
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DISTRICTS_GEOJSON = os.path.join(BASE_DIR, "data", "boundaries", "assam_districts.geojson")
OUT_DIR = os.path.join(BASE_DIR, "data", "lulc_dynamicworld")
SCALE_M = 10          # resolution used for the per-class area/percentage stats
DOWNLOAD_SCALE_M = 30  # coarser resolution for the saved raster - GEE's direct
                        # getDownloadURL is capped at 48MB; native 10m rasters
                        # for these district sizes exceed that, 30m does not

CLASS_NAMES = [
    "water", "trees", "grass", "flooded_vegetation", "crops",
    "shrub_and_scrub", "built", "bare", "snow_and_ice",
]

# (district, tag, start_date, end_date) - windows around the same
# cross-verified event dates used in pull_gfm_validation.py (manual-0013).
TARGETS = [
    ("Morigaon", "2022_pre", "2022-05-01", "2022-06-14"),
    ("Morigaon", "2022_during", "2022-06-15", "2022-07-05"),
    ("Nagaon", "2022_pre", "2022-05-01", "2022-06-14"),
    ("Nagaon", "2022_during", "2022-06-15", "2022-07-05"),
    ("Karimganj", "2022_pre", "2022-05-01", "2022-06-14"),
    ("Karimganj", "2022_during", "2022-06-15", "2022-07-05"),
    ("Karimganj", "2024_pre", "2024-05-01", "2024-06-14"),
    ("Karimganj", "2024_during", "2024-06-15", "2024-07-05"),
    ("Sivasagar", "2026_pre", "2026-06-15", "2026-07-14"),
    ("Sivasagar", "2026_during", "2026-07-15", "2026-08-25"),
]


def load_district_geometry(district_name: str) -> dict:
    with open(DISTRICTS_GEOJSON, encoding="utf-8") as f:
        gj = json.load(f)
    aliases = {district_name.lower(), district_name.replace("Morigaon", "Marigaon").lower()}
    for feat in gj["features"]:
        if feat["properties"].get("dtname", "").lower() in aliases:
            return feat["geometry"]
    raise ValueError(f"District '{district_name}' not found in {DISTRICTS_GEOJSON}")


def pull_target(district, tag, start, end, region):
    col = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(start, end)
        .filterBounds(region)
        .select("label")
    )
    n_images = col.size().getInfo()
    if n_images == 0:
        print(f"  no Dynamic World scenes found for {district}/{tag} ({start} to {end})")
        return None

    mode_img = col.reduce(ee.Reducer.mode()).rename("label").clip(region)

    # per-class pixel-area stats
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
            "district": district, "tag": tag, "start": start, "end": end,
            "n_scenes": n_images, "class": cls_name, "area_km2": round(area_km2, 3),
            "pct": round(pct, 3),
        })

    # download the clipped mode-label raster as a local GeoTIFF (coarser
    # scale than the stats above - see DOWNLOAD_SCALE_M comment)
    import requests
    dist_dir = os.path.join(OUT_DIR, district)
    os.makedirs(dist_dir, exist_ok=True)
    out_path = os.path.join(dist_dir, f"{district}_{tag}_{start}_to_{end}_label.tif")
    try:
        url = mode_img.getDownloadURL({
            "region": region, "scale": DOWNLOAD_SCALE_M, "format": "GEO_TIFF",
            "crs": "EPSG:4326",
        })
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
        print(f"  {district}/{tag}: {n_images} scenes, saved {out_path} "
              f"({len(class_rows)} classes present)")
    except Exception as e:
        print(f"  {district}/{tag}: {n_images} scenes, stats OK but raster "
              f"download FAILED ({e}) - class percentages still recorded")

    return class_rows


def main():
    ee.Initialize(project=PROJECT_ID)
    os.makedirs(OUT_DIR, exist_ok=True)

    all_rows = []
    for district, tag, start, end in TARGETS:
        print(f"\n[{district} / {tag} / {start} to {end}]")
        region = ee.Geometry(load_district_geometry(district))
        try:
            rows = pull_target(district, tag, start, end, region)
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
