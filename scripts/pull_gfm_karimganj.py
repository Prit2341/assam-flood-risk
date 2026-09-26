"""Pull dry-season non-flood rasters for Karimganj only."""
import json
import numpy as np
import rasterio
import requests
from pathlib import Path
from pyproj import Transformer
from rasterio.mask import mask
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform

STAC_URL      = "https://stac.eodc.eu/api/v1/search"
DISTRICTS_GJ  = "data/boundaries/assam_districts.geojson"
OUT_DIR       = Path("data/gfm_nonflood/Karimganj")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PIXEL_AREA_KM2 = (20 * 20) / 1e6
MAX_FLOOD_PCT  = 1.0

WINDOWS = [
    ("dry_2022", "2022-01-01", "2022-02-28"),
    ("dry_2024", "2024-01-01", "2024-02-29"),
    ("dry_2018", "2018-01-01", "2018-02-28"),
]

with open(DISTRICTS_GJ) as f:
    gj = json.load(f)

poly_wgs84 = None
for feat in gj["features"]:
    if feat["properties"].get("dtname", "").lower() == "karimganj":
        poly_wgs84 = shape(feat["geometry"])
        break

if poly_wgs84 is None:
    raise ValueError("Karimganj not found in districts GeoJSON")

transformer  = Transformer.from_crs("EPSG:4326", "EPSG:27703", always_xy=True)
poly_native  = shapely_transform(lambda x, y: transformer.transform(x, y), poly_wgs84)
minx, miny, maxx, maxy = poly_wgs84.bounds
bbox = [minx, miny, maxx, maxy]

for tag, start_str, end_str in WINDOWS:
    print(f"\n[Karimganj / {tag} / {start_str} -> {end_str}]")

    try:
        r = requests.post(STAC_URL, json={
            "collections": ["GFM"], "bbox": bbox,
            "datetime": f"{start_str}T00:00:00Z/{end_str}T23:59:59Z",
            "limit": 100,
        }, timeout=60)
        r.raise_for_status()
        features = r.json().get("features", [])
        print(f"  {len(features)} scenes found")
    except Exception as e:
        print(f"  ERROR search: {e}")
        continue

    best = None
    for item in features:
        assets   = item["assets"]
        if "ensemble_flood_extent" not in assets:
            continue
        flood_url = assets["ensemble_flood_extent"]["href"]
        excl_url  = assets.get("exclusion_mask", {}).get("href")
        date_str  = item["properties"].get("datetime", "")[:10]
        try:
            with rasterio.open(flood_url) as src:
                arr, tr = mask(src, [poly_native], crop=True, nodata=src.nodata)
                nd = src.nodata
            arr   = arr[0]
            valid = (arr != nd) if nd is not None else np.ones_like(arr, bool)
            if excl_url:
                with rasterio.open(excl_url) as src2:
                    ex = mask(src2, [poly_native], crop=True, nodata=src2.nodata)[0][0]
                if ex.shape == arr.shape:
                    valid &= (ex == 0)
            vpx  = int(valid.sum())
            fpx  = int(((arr == 1) & valid).sum())
            fpct = 100 * fpx / vpx if vpx > 0 else 999
            print(f"  {date_str}  valid={vpx}  flood%={fpct:.2f}%")
            if vpx > 0 and fpct <= MAX_FLOOD_PCT:
                if best is None or vpx > best["vpx"]:
                    best = {"arr": arr, "tr": tr, "vpx": vpx, "fpx": fpx,
                            "fpct": fpct, "date": date_str}
        except Exception as e:
            print(f"  {date_str} read error: {e}")
            continue

    if not best:
        print("  no usable scene — skip")
        continue

    out_path = OUT_DIR / f"Karimganj_{tag}_{best['date']}_nonflood.tif"
    with rasterio.open(
        str(out_path), "w", driver="GTiff",
        height=best["arr"].shape[0], width=best["arr"].shape[1],
        count=1, dtype=best["arr"].dtype, crs="EPSG:27703",
        transform=best["tr"], nodata=255,
    ) as dst:
        dst.write(best["arr"], 1)

    flood_km2 = best["fpx"] * PIXEL_AREA_KM2
    valid_km2 = best["vpx"] * PIXEL_AREA_KM2
    print(f"  SAVED {best['date']}  flood={flood_km2:.1f}km2  "
          f"valid={valid_km2:.1f}km2  ({best['fpct']:.3f}%)  -> {out_path}")

print("\nDone.")
