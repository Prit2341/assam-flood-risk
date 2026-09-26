"""
Pull GFM dry-season (non-flood) rasters for RF susceptibility training.

The RF needs two classes:
  1 = flood    -> already in data/gfm_validation/ (done)
  0 = no-flood -> this script

Strategy: 2 dry-season dates per event year, per district.
  - January or February of each event year (peak dry season in Assam)
  - One date early Jan, one date mid-Feb -> gives the model variety
  - Skips any date that returns >1% flood pixels (real flood, unusable as negative)

Uses the same COG-streaming approach as pull_gfm_validation.py:
  - No full tile download, clips over HTTP
  - Saves clipped rasters to data/gfm_nonflood/<district>/
  - Appends a summary row to data/gfm_nonflood/summary.csv

Run:
    python scripts/pull_gfm_nonflood.py
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

STAC_URL      = "https://stac.eodc.eu/api/v1/search"
BASE_DIR      = os.path.join(os.path.dirname(__file__), "..")
DISTRICTS_GJ  = os.path.join(BASE_DIR, "data", "boundaries", "assam_districts.geojson")
OUT_DIR       = os.path.join(BASE_DIR, "data", "gfm_nonflood")
PIXEL_AREA_KM2 = (20 * 20) / 1e6   # GFM native 20m resolution

# Max flood % allowed before a "dry" date is rejected as actually flooded
MAX_FLOOD_PCT = 1.0

# Search whole Jan-Feb window per district per year.
# Sentinel-1 revisits every 6-12 days so there are only ~5-8 scenes per
# 2-month window — we take the best-coverage one that passes the flood check.
# Format: (district, tag, search_start, search_end)
TARGETS = [
    # 2022 event year — all 4 districts
    ("Morigaon",  "dry_2022", "2022-01-01", "2022-02-28"),
    ("Nagaon",    "dry_2022", "2022-01-01", "2022-02-28"),
    ("Karimganj", "dry_2022", "2022-01-01", "2022-02-28"),
    ("Sivasagar", "dry_2022", "2022-01-01", "2022-02-28"),

    # 2024 event year (Karimganj)
    ("Karimganj", "dry_2024", "2024-01-01", "2024-02-29"),

    # 2026 event year (Sivasagar)
    ("Sivasagar", "dry_2026", "2026-01-01", "2026-02-28"),

    # 2020 for Morigaon/Nagaon — extra negative samples for the Kopili basin
    ("Morigaon",  "dry_2020", "2020-01-01", "2020-02-29"),
    ("Nagaon",    "dry_2020", "2020-01-01", "2020-02-29"),

    # 2018 for Karimganj — extra negative sample
    ("Karimganj", "dry_2018", "2018-01-01", "2018-02-28"),
]


def load_district(name: str):
    with open(DISTRICTS_GJ, encoding="utf-8") as f:
        gj = json.load(f)
    aliases = {name.lower(), name.replace("Morigaon", "Marigaon").lower()}
    for feat in gj["features"]:
        if feat["properties"].get("dtname", "").lower() in aliases:
            return shape(feat["geometry"])
    raise ValueError(f"District '{name}' not found in {DISTRICTS_GJ}")


def search_scenes(bbox, date_str):
    body = {
        "collections": ["GFM"],
        "bbox": bbox,
        "datetime": f"{date_str}T00:00:00Z/{date_str}T23:59:59Z",
        "limit": 50,
    }
    r = requests.post(STAC_URL, json=body, timeout=60)
    r.raise_for_status()
    return r.json().get("features", [])


def search_scenes_range(bbox, start_str, end_str):
    """Search across a date range, return all scenes found."""
    body = {
        "collections": ["GFM"],
        "bbox": bbox,
        "datetime": f"{start_str}T00:00:00Z/{end_str}T23:59:59Z",
        "limit": 100,
    }
    r = requests.post(STAC_URL, json=body, timeout=60)
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
    excl_url  = assets["exclusion_mask"]["href"]

    flood, transform, nodata = clip_asset(flood_url, poly_native)
    if flood is None:
        return None
    excl, _, _ = clip_asset(excl_url, poly_native)

    valid = flood != nodata if nodata is not None else np.ones_like(flood, dtype=bool)
    if excl is not None and excl.shape == flood.shape:
        valid &= excl == 0

    valid_px = int(valid.sum())
    flood_px = int(((flood == 1) & valid).sum())
    return {
        "scene_id": item["id"],
        "flood_url": flood_url,
        "flood_arr": flood,
        "transform": transform,
        "valid_px": valid_px,
        "flood_px": flood_px,
    }


def process_target(district, tag, start_str, end_str, transformer):
    print(f"\n[{district} / {tag} / {start_str} → {end_str}]")
    poly_wgs84 = load_district(district)
    minx, miny, maxx, maxy = poly_wgs84.bounds
    bbox = [minx, miny, maxx, maxy]

    features = search_scenes_range(bbox, start_str, end_str)
    if not features:
        print("  no GFM scenes in this window — skip")
        return None

    poly_native = shapely_transform(
        lambda x, y: transformer.transform(x, y), poly_wgs84
    )

    # Evaluate all candidates; pick best valid_px that also passes flood check
    candidates = []
    for item in features:
        result = evaluate_scene(item, poly_native)
        if result is None or result["valid_px"] == 0:
            continue
        flood_pct = 100 * result["flood_px"] / result["valid_px"]
        result["flood_pct"] = flood_pct
        result["date_str"] = item["properties"].get("datetime", "")[:10]
        print(f"  {result['date_str']} {result['scene_id']}: "
              f"valid_px={result['valid_px']}  flood%={flood_pct:.2f}%")
        if flood_pct <= MAX_FLOOD_PCT:
            candidates.append(result)

    if not candidates:
        print("  all scenes rejected (no valid pixels or flood% too high) — skip")
        return None

    # Take the one with the most valid pixels (best coverage over district)
    best = max(candidates, key=lambda x: x["valid_px"])

    flood_km2 = best["flood_px"] * PIXEL_AREA_KM2
    valid_km2 = best["valid_px"] * PIXEL_AREA_KM2
    date_str = best["date_str"]

    dist_dir = os.path.join(OUT_DIR, district)
    os.makedirs(dist_dir, exist_ok=True)
    out_path = os.path.join(dist_dir, f"{district}_{tag}_{date_str}_nonflood.tif")

    if os.path.exists(out_path):
        try:
            with rasterio.open(out_path) as src:
                src.read(1)
            print(f"  already ok — skip")
            return {"district": district, "tag": tag, "date": date_str,
                    "scene_id": best["scene_id"], "flood_km2": round(flood_km2, 2),
                    "valid_km2": round(valid_km2, 2),
                    "flood_pct": round(best["flood_pct"], 4),
                    "label": 0, "raster_path": out_path}
        except Exception:
            os.remove(out_path)

    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=best["flood_arr"].shape[0], width=best["flood_arr"].shape[1],
        count=1, dtype=best["flood_arr"].dtype, crs="EPSG:27703",
        transform=best["transform"], nodata=255,
    ) as dst:
        dst.write(best["flood_arr"], 1)

    print(f"  SAVED {date_str}: flood={flood_km2:.1f} km2 / valid={valid_km2:.1f} km2 "
          f"({best['flood_pct']:.3f}%) -> {out_path}")

    return {"district": district, "tag": tag, "date": date_str,
            "scene_id": best["scene_id"], "flood_km2": round(flood_km2, 2),
            "valid_km2": round(valid_km2, 2), "flood_pct": round(best["flood_pct"], 4),
            "label": 0, "raster_path": out_path}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27703", always_xy=True)

    rows = []
    skipped = []
    for district, tag, start_str, end_str in TARGETS:
        try:
            row = process_target(district, tag, start_str, end_str, transformer)
        except Exception as e:
            print(f"  ERROR: {e}")
            row = None
        if row:
            rows.append(row)
        else:
            skipped.append((district, tag, start_str, end_str))

    summary_path = os.path.join(OUT_DIR, "summary.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print(f"Done. {len(rows)}/{len(TARGETS)} targets saved.")
    if skipped:
        print(f"Skipped ({len(skipped)}): {skipped}")
    print(f"Summary: {summary_path}")
    print(f"\nNext step: combine with data/gfm_validation/ flood rasters")
    print(f"  -> flood label=1 from gfm_validation/")
    print(f"  -> non-flood label=0 from gfm_nonflood/  (this script)")
    print(f"  -> sample terrain features (FABDEM/TWI/slope/flow_acc) at each pixel")
    print(f"  -> train RF classifier (Tehrany et al. 2014)")


if __name__ == "__main__":
    main()
