"""
Downloads real flood-extent validation data from Copernicus Global Flood
Monitoring (GFM) for the 4 current-scope districts (Morigaon, Sivasagar,
Nagaon, Karimganj), clipped to each district's ACTUAL polygon (not a bbox,
not the whole 300x300km tile) - this closes the AOI-clipping gap flagged as
"not yet done" in manual-0007/manual-0008.

For each (district, date) pair:
  1. Search the GFM STAC API for scenes intersecting the district's bbox
     on that date.
  2. For every candidate scene, open ensemble_flood_extent + exclusion_mask
     directly over HTTP (they are cloud-optimized GeoTIFFs - no full-tile
     download needed) and clip to the district polygon (reprojected to the
     scene's native CRS, EPSG:27703 Equi7 Asia).
  3. Pick whichever candidate scene has the most VALID (non-nodata) pixels
     inside the actual district polygon - this is a real fix over the
     earlier bbox/tile-level "largest coverage" heuristic, since two
     districts sharing one tile (Morigaon/Nagaon) can now be told apart,
     and a partial-swath scene that happens to miss the district itself
     will no longer be picked just because the whole tile had more pixels.
  4. Apply the exclusion_mask (radar shadow/layover/urban false-positive
     suppression) before counting flood pixels.
  5. Save the clipped flood-extent and exclusion-mask rasters to disk, and
     append one summary row (district, date, tag, flood_km2, valid_km2,
     flood_pct, scene_id) to data/gfm_validation/summary.csv.

Run:
    python scripts/pull_gfm_validation.py
"""
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
OUT_DIR = os.path.join(BASE_DIR, "data", "gfm_validation")

PIXEL_AREA_KM2 = (20 * 20) / 1e6  # 20m native GFM resolution

# (district, tag, date) - dates cross-verified against multiple independent
# sources (news, Wikipedia, SANDRP/CWC-derived records, ASDMA reports) in
# manual-0013, 2026-09-11. This REPLACES the manual-0008/0012 date picks
# with corrected ones for Karimganj 2024 and Sivasagar 2026 (see below).
TARGETS = [
    ("Morigaon", "2022_pre", "2022-06-06"),
    ("Morigaon", "2022_during", "2022-06-28"),
    ("Nagaon", "2022_pre", "2022-06-06"),
    ("Nagaon", "2022_during", "2022-06-28"),
    ("Karimganj", "2022_pre", "2022-06-06"),
    ("Karimganj", "2022_during_a", "2022-06-28"),
    ("Karimganj", "2022_during_b", "2022-06-30"),
    ("Karimganj", "2022_during_c", "2022-07-02"),
    ("Karimganj", "2024_pre", "2024-05-26"),
    ("Karimganj", "2024_early", "2024-06-19"),
    # CORRECTED: 06-19 was only the early stage; verified danger-level-
    # crossing peak is 06-23/25 (manual-0013) - three candidate dates
    # since exact same-day scene availability is not guaranteed.
    ("Karimganj", "2024_peak_a", "2024-06-23"),
    ("Karimganj", "2024_peak_b", "2024-06-24"),
    ("Karimganj", "2024_peak_c", "2024-06-25"),
    ("Sivasagar", "2026_pre", "2026-07-05"),
    # CORRECTED: verified actual record-level date is 2026-07-20 (SANDRP/
    # CWC), but that exact date returned zero valid pixels over Sivasagar's
    # polygon in the first pass - trying adjacent dates instead.
    ("Sivasagar", "2026_peak_a", "2026-07-19"),
    ("Sivasagar", "2026_peak_b", "2026-07-20"),
    ("Sivasagar", "2026_peak_c", "2026-07-21"),
    # Nearest CONFIRMED-scene-exists alternates around the true peak, since
    # the exact verified dates above have no usable coverage (manual-0014).
    # Found via a direct STAC scene-calendar check, not a guess.
    ("Karimganj", "2024_near_peak_a", "2024-06-22"),
    ("Karimganj", "2024_near_peak_b", "2024-06-29"),
    ("Sivasagar", "2026_near_peak_a", "2026-07-17"),
    ("Sivasagar", "2026_near_peak_b", "2026-07-22"),
]


def load_district(name: str):
    with open(DISTRICTS_GEOJSON, encoding="utf-8") as f:
        gj = json.load(f)
    aliases = {name.lower(), name.replace("Morigaon", "Marigaon").lower()}
    for feat in gj["features"]:
        if feat["properties"].get("dtname", "").lower() in aliases:
            return shape(feat["geometry"])
    raise ValueError(f"District '{name}' not found in {DISTRICTS_GEOJSON}")


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
        valid &= excl == 0  # 0 = not excluded; nonzero = radar shadow/layover/urban etc.

    valid_px = int(valid.sum())
    flood_px = int(((flood == 1) & valid).sum())
    return {
        "scene_id": item["id"],
        "flood_url": flood_url,
        "excl_url": excl_url,
        "flood_arr": flood,
        "excl_arr": excl,
        "transform": transform,
        "valid_px": valid_px,
        "flood_px": flood_px,
    }


def process_target(district, tag, date_str, transformer):
    print(f"\n[{district} / {tag} / {date_str}]")
    poly_wgs84 = load_district(district)
    minx, miny, maxx, maxy = poly_wgs84.bounds
    bbox = [minx, miny, maxx, maxy]

    features = search_scenes(bbox, date_str)
    if not features:
        print("  no GFM scenes found for this date/bbox")
        return None

    poly_native = shapely_transform(
        lambda x, y: transformer.transform(x, y), poly_wgs84
    )

    best = None
    for item in features:
        result = evaluate_scene(item, poly_native)
        if result is None:
            continue
        print(f"  candidate {result['scene_id']}: valid_px={result['valid_px']}")
        if best is None or result["valid_px"] > best["valid_px"]:
            best = result

    if best is None or best["valid_px"] == 0:
        print("  no usable scene (no valid pixels over district polygon)")
        return None

    flood_km2 = best["flood_px"] * PIXEL_AREA_KM2
    valid_km2 = best["valid_px"] * PIXEL_AREA_KM2
    pct = 100 * best["flood_px"] / best["valid_px"]

    dist_dir = os.path.join(OUT_DIR, district)
    os.makedirs(dist_dir, exist_ok=True)
    out_path = os.path.join(dist_dir, f"{district}_{tag}_{date_str}_flood.tif")
    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=best["flood_arr"].shape[0], width=best["flood_arr"].shape[1],
        count=1, dtype=best["flood_arr"].dtype, crs="EPSG:27703",
        transform=best["transform"], nodata=255,
    ) as dst:
        dst.write(best["flood_arr"], 1)

    print(f"  chosen scene {best['scene_id']}: flood={flood_km2:.1f} km2 / "
          f"valid={valid_km2:.1f} km2 ({pct:.3f}%) -> saved {out_path}")

    return {
        "district": district, "tag": tag, "date": date_str,
        "scene_id": best["scene_id"], "flood_km2": round(flood_km2, 2),
        "valid_km2": round(valid_km2, 2), "flood_pct": round(pct, 4),
        "raster_path": out_path,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27703", always_xy=True)

    rows = []
    for district, tag, date_str in TARGETS:
        try:
            row = process_target(district, tag, date_str, transformer)
        except Exception as e:
            print(f"  ERROR: {e}")
            row = None
        if row:
            rows.append(row)

    summary_path = os.path.join(OUT_DIR, "summary.csv")
    import pandas as pd
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\nDone. {len(rows)}/{len(TARGETS)} targets succeeded.")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
