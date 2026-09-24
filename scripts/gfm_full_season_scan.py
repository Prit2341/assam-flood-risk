"""
Full-season GFM scan -- addresses a real gap in pull_gfm_validation.py:
that script only checked a handful of HAND-PICKED dates (news-verified
onset/peak +/- a few days), not the complete scene calendar. This script
queries the GFM STAC API for EVERY available scene across the whole flood
season for each district/year and computes flood% for all of them, so we
can see the true full time series and confirm (or correct) which date
actually has the maximum measured flood extent.

Run:
    python scripts/gfm_full_season_scan.py
"""
import json
import os

import numpy as np
import pandas as pd
import rasterio
import requests
from pyproj import Transformer
from rasterio.mask import mask
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform

STAC_URL = "https://stac.eodc.eu/api/v1/search"
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DISTRICTS_GEOJSON = os.path.join(BASE_DIR, "data", "boundaries", "scope4_districts.geojson")
OUT_DIR = os.path.join(BASE_DIR, "data", "gfm_full_season_scan")
PIXEL_AREA_KM2 = (20 * 20) / 1e6

# (district, year, start_date, end_date) - full flood-season windows, wide
# enough to bracket the news-verified onset/peak with real margin on both
# sides, not just +/- a few hand-picked days.
WINDOWS = [
    ("Morigaon", 2022, "2022-05-01", "2022-07-31"),
    ("Nagaon", 2022, "2022-05-01", "2022-07-31"),
    ("Karimganj", 2022, "2022-05-01", "2022-07-31"),
    ("Karimganj", 2024, "2024-05-01", "2024-07-31"),
    ("Sivasagar", 2026, "2026-06-01", "2026-08-31"),
]


def load_district(name: str):
    with open(DISTRICTS_GEOJSON, encoding="utf-8") as f:
        gj = json.load(f)
    aliases = {name.lower(), name.replace("Morigaon", "Marigaon").lower()}
    for feat in gj["features"]:
        if feat["properties"].get("dtname", "").lower() in aliases:
            return shape(feat["geometry"])
    raise ValueError(f"District '{name}' not found")


def search_scenes(bbox, start, end):
    body = {
        "collections": ["GFM"],
        "bbox": bbox,
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": 200,
    }
    r = requests.post(STAC_URL, json=body, timeout=90)
    r.raise_for_status()
    return r.json().get("features", [])


def clip_asset(url, geom_native):
    with rasterio.open(url) as src:
        try:
            arr, transform = mask(src, [geom_native], crop=True, nodata=src.nodata)
        except ValueError:
            return None, None
        return arr[0], src.nodata


def evaluate_scene(item, poly_native):
    assets = item["assets"]
    if "ensemble_flood_extent" not in assets or "exclusion_mask" not in assets:
        return None
    flood, nodata = clip_asset(assets["ensemble_flood_extent"]["href"], poly_native)
    if flood is None:
        return None
    excl, _ = clip_asset(assets["exclusion_mask"]["href"], poly_native)

    valid = flood != nodata if nodata is not None else np.ones_like(flood, dtype=bool)
    if excl is not None and excl.shape == flood.shape:
        valid &= excl == 0

    valid_px = int(valid.sum())
    flood_px = int(((flood == 1) & valid).sum())
    return valid_px, flood_px


def scan_district_year(district, year, start, end, transformer):
    print(f"\n[{district} {year}] scanning {start} to {end} ...")
    poly_wgs84 = load_district(district)
    minx, miny, maxx, maxy = poly_wgs84.bounds
    bbox = [minx, miny, maxx, maxy]
    poly_native = shapely_transform(lambda x, y: transformer.transform(x, y), poly_wgs84)

    features = search_scenes(bbox, start, end)
    print(f"  {len(features)} candidate scenes found in date range")

    rows = []
    for item in features:
        dt = item["properties"].get("datetime", item["id"])
        result = evaluate_scene(item, poly_native)
        if result is None:
            continue
        valid_px, flood_px = result
        if valid_px == 0:
            continue
        pct = 100 * flood_px / valid_px
        rows.append({
            "district": district, "year": year, "scene_id": item["id"],
            "datetime": dt, "valid_km2": round(valid_px * PIXEL_AREA_KM2, 2),
            "flood_km2": round(flood_px * PIXEL_AREA_KM2, 2),
            "flood_pct": round(pct, 4),
        })
        print(f"    {dt}: flood={flood_px * PIXEL_AREA_KM2:.1f}km2 valid={valid_px * PIXEL_AREA_KM2:.1f}km2 ({pct:.2f}%)")
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27703", always_xy=True)

    all_rows = []
    for district, year, start, end in WINDOWS:
        try:
            rows = scan_district_year(district, year, start, end, transformer)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  ERROR: {e}")

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(OUT_DIR, "full_season_timeseries.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved full time series: {out_csv} ({len(df)} usable scenes total)")

    if not df.empty:
        print("\n=== TRUE MAXIMUM flood_pct per district/year (from ALL scenes, not hand-picked dates) ===")
        idx = df.groupby(["district", "year"])["flood_pct"].idxmax()
        print(df.loc[idx, ["district", "year", "datetime", "flood_pct", "valid_km2"]].to_string(index=False))


if __name__ == "__main__":
    main()
