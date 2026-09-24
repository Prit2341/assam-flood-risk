"""
Digs for Copernicus GFM (Sentinel-1) flood-extent coverage in the years the
MODIS GFD scan (manual-0037) could NOT reach: 2019, 2020, 2021, 2023
(MODIS GFD v1 stops at 2018; our own GFM pulls only exist for 2022/2024/2026).

Two-stage, timeout-safe (the earlier full-season scan hung silently for
45+ min with zero output - this version prints progress every STAC call and
only downloads/evaluates rasters for a short, targeted date list per year):

  Stage 1: STAC search (cheap, no raster I/O) over each district's bbox for
           the whole flood season (Apr-Oct) of each gap year, to see what
           scene DATES actually exist at all.
  Stage 2: For a handful of dates inside each year's documented flood window
           (from web-search-derived approximate windows, since exact
           per-district peak dates were not findable for 2019/2020/2021/2023),
           evaluate real flood_pct exactly like pull_gfm_validation.py.

Run:
    python scripts/gfm_gap_years_scan.py
"""
import datetime
import json
import os

import numpy as np
import rasterio
import requests
from pyproj import Transformer
from rasterio.mask import mask
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform

STAC_URL = "https://stac.eodc.eu/api/v1/search"
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DISTRICTS_GEOJSON = os.path.join(BASE_DIR, "data", "boundaries", "assam_districts.geojson")
OUT_DIR = os.path.join(BASE_DIR, "data", "gfm_gap_years")
PIXEL_AREA_KM2 = (20 * 20) / 1e6

DISTRICTS = ["Nagaon"]

# Approximate documented flood windows per year (from web search, manual-0038
# candidates) - not exact per-district peak dates, used only to pick a
# tractable number of candidate dates for Stage 2 evaluation.
YEAR_WINDOWS = {
    2021: [("2021-05-01", "2021-06-15")],
    2023: [("2023-06-14", "2023-06-25")],
}


def load_district(name: str):
    with open(DISTRICTS_GEOJSON, encoding="utf-8") as f:
        gj = json.load(f)
    aliases = {name.lower(), name.replace("Morigaon", "Marigaon").lower()}
    for feat in gj["features"]:
        if feat["properties"].get("dtname", "").lower() in aliases:
            return shape(feat["geometry"])
    raise ValueError(f"District '{name}' not found")


def search_scenes(bbox, dt_from, dt_to):
    body = {
        "collections": ["GFM"],
        "bbox": bbox,
        "datetime": f"{dt_from}T00:00:00Z/{dt_to}T23:59:59Z",
        "limit": 200,
    }
    r = requests.post(STAC_URL, json=body, timeout=30)
    r.raise_for_status()
    return r.json().get("features", [])


def clip_asset(url, geom_native):
    with rasterio.open(url) as src:
        try:
            arr, transform = mask(src, [geom_native], crop=True, nodata=src.nodata)
        except ValueError:
            return None, None, None
        return arr[0], transform, src.nodata


def evaluate_scene(item, poly_native):
    assets = item["assets"]
    if "ensemble_flood_extent" not in assets or "exclusion_mask" not in assets:
        return None
    flood_url = assets["ensemble_flood_extent"]["href"]
    excl_url = assets["exclusion_mask"]["href"]
    flood, transform, nodata = clip_asset(flood_url, poly_native)
    if flood is None:
        return None
    excl, _, _ = clip_asset(excl_url, poly_native)
    valid = flood != nodata if nodata is not None else np.ones_like(flood, dtype=bool)
    if excl is not None and excl.shape == flood.shape:
        valid &= excl == 0
    valid_px = int(valid.sum())
    flood_px = int(((flood == 1) & valid).sum())
    return valid_px, flood_px


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27703", always_xy=True)
    rows = []

    for district in DISTRICTS:
        poly_wgs84 = load_district(district)
        minx, miny, maxx, maxy = poly_wgs84.bounds
        bbox = [minx, miny, maxx, maxy]
        poly_native = shapely_transform(lambda x, y: transformer.transform(x, y), poly_wgs84)

        for year, windows in YEAR_WINDOWS.items():
            print(f"\n=== {district} {year} ===", flush=True)
            all_dates = set()
            for dt_from, dt_to in windows:
                try:
                    feats = search_scenes(bbox, dt_from, dt_to)
                except Exception as ex:
                    print(f"  STAC search FAILED {dt_from}/{dt_to}: {ex}", flush=True)
                    continue
                dates = sorted({f["properties"]["datetime"][:10] for f in feats})
                print(f"  window {dt_from}..{dt_to}: {len(feats)} scenes, dates={dates}", flush=True)
                all_dates.update(dates)

            if not all_dates:
                print("  NO SCENES AT ALL in documented flood window -> no GFM coverage this year", flush=True)
                continue

            # Stage 2: evaluate every distinct date found (usually few)
            for date_str in sorted(all_dates):
                try:
                    feats = search_scenes(bbox, date_str, date_str)
                except Exception as ex:
                    print(f"  {date_str}: STAC re-fetch failed: {ex}", flush=True)
                    continue
                best = None
                for item in feats:
                    try:
                        res = evaluate_scene(item, poly_native)
                    except Exception as ex:
                        print(f"    {item.get('id')}: eval error {ex}", flush=True)
                        continue
                    if res is None:
                        continue
                    valid_px, flood_px = res
                    if valid_px == 0:
                        continue
                    if best is None or valid_px > best[0]:
                        best = (valid_px, flood_px, item["id"])
                if best is None:
                    print(f"  {date_str}: no usable scene (0 valid pixels)", flush=True)
                    continue
                valid_px, flood_px, scene_id = best
                pct = 100 * flood_px / valid_px
                print(f"  {date_str}: scene {scene_id} flood_pct={pct:.2f}% "
                      f"(valid={valid_px*PIXEL_AREA_KM2:.1f}km2)", flush=True)
                rows.append({
                    "district": district, "year": year, "date": date_str,
                    "scene_id": scene_id, "flood_pct": round(pct, 3),
                    "valid_km2": round(valid_px * PIXEL_AREA_KM2, 2),
                    "flood_km2": round(flood_px * PIXEL_AREA_KM2, 2),
                })

    import pandas as pd
    out_csv = os.path.join(OUT_DIR, "gap_years_rerun_nagaon.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nDone. {len(rows)} usable (district,year,date) rows -> {out_csv}")


if __name__ == "__main__":
    main()
