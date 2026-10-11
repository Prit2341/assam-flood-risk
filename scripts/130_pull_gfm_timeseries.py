"""
Multi-date flood labels: every GFM (Copernicus Global Flood Monitoring) Sentinel-1 scene over each district,
monsoon months (Apr-Oct) 2016-2026, resampled to the ~90 m analysis grid (dyn_common.grid).

WHY: the dynamic model must learn how flooding changes with rain/soil moisture/discharge over time.
The Phase-1 label set had ~13 hand-picked flood scenes + 3 dry scenes (~3 independent events), which
cannot identify a time-varying effect. Multi-date Sentinel-1 inundation maps as training targets for a
model with hydrological + terrain predictors: Dasgupta, Hybbeneth & Waske 2022 (arXiv:2208.09135);
time-matched dynamic factors per event: Remote Sens. 2026, 18(18), 3253 (monthly XGBoost, Hunan).
GFM: three-algorithm ensemble, majority vote (CEMS GFM product description; JRC QA reports).

Per scene label on the 90 m grid (uint8): 1 flooded, 0 observed not flooded, 255 unusable.
A 90 m cell is usable only if >=50% of its 20 m pixels are valid (not GFM nodata, not exclusion_mask,
not reference_water_mask) and it is inside the district and not DEM-flattened water. Flooded if >=50%
of its valid pixels are flood. Tiles of the same date are mosaicked (a district can span 2 Equi7 tiles).

Run: python -u scripts/130_pull_gfm_timeseries.py <District>|all
Out: data/dynamic_susceptibility/labels/<District>/<YYYY-MM-DD>.npz, labels_index.csv
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from shapely.geometry import shape as geom_shape

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dyn_common as dc

STAC = "https://stac.eodc.eu/api/v1/search"
YEARS = range(2016, 2027)
MONTHS = range(4, 11)
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")


def search(bbox, y, mth):
    end = pd.Timestamp(y, mth, 1) + pd.offsets.MonthEnd(0)
    body = {"collections": ["GFM"], "bbox": bbox, "limit": 500,
            "datetime": f"{y}-{mth:02d}-01T00:00:00Z/{end:%Y-%m-%d}T23:59:59Z"}
    for a in range(5):
        try:
            r = requests.post(STAC, json=body, timeout=120)
            r.raise_for_status()
            return r.json().get("features", [])
        except Exception as e:
            time.sleep(5 * (a + 1))
    print(f"  search failed {y}-{mth}", flush=True)
    return None


def warp(href, tr, shape, resampling):
    for a in range(4):
        try:
            with rasterio.open(href) as src:
                with WarpedVRT(src, crs="EPSG:4326", transform=tr, width=shape[1], height=shape[0],
                               resampling=resampling, src_nodata=src.nodata, nodata=255) as v:
                    return v.read(1)
        except Exception as e:
            err = e
            time.sleep(4 * (a + 1))
    raise err


def scene_label(items, tr, shape, usable_grid):
    """Mosaic all tiles of one date. Returns uint8 label grid."""
    flood_sum = np.zeros(shape, np.float32)
    valid_sum = np.zeros(shape, np.float32)
    # fine grid (factor 4 inside each 90 m cell ~ 22 m) to count 20 m pixels, then aggregate
    F = 4
    trf = tr * tr.scale(1 / F, 1 / F)
    shf = (shape[0] * F, shape[1] * F)
    for it in items:
        A = it["assets"]
        if not all(k in A for k in ("ensemble_flood_extent", "exclusion_mask", "reference_water_mask")):
            continue
        fl = warp(A["ensemble_flood_extent"]["href"], trf, shf, Resampling.nearest)
        ex = warp(A["exclusion_mask"]["href"], trf, shf, Resampling.nearest)
        rw = warp(A["reference_water_mask"]["href"], trf, shf, Resampling.nearest)
        valid = (fl != 255) & (ex == 0) & (rw == 0)
        fv = ((fl == 1) & valid).astype(np.float32).reshape(shape[0], F, shape[1], F).sum(axis=(1, 3))
        vv = valid.astype(np.float32).reshape(shape[0], F, shape[1], F).sum(axis=(1, 3))
        take = vv > valid_sum                      # overlapping tiles: keep the better-observed one
        flood_sum[take], valid_sum[take] = fv[take], vv[take]
    lab = np.full(shape, 255, np.uint8)
    ok = (valid_sum >= 0.5 * F * F) & usable_grid
    lab[ok] = (flood_sum[ok] / valid_sum[ok] >= 0.5).astype(np.uint8)
    return lab


def run(d):
    tr, shape, inside, bad = dc.grid(d)
    usable = inside & ~bad
    out_dir = os.path.join(dc.DYN, "labels", d)
    os.makedirs(out_dir, exist_ok=True)
    idx_path = os.path.join(out_dir, "labels_index.csv")
    done = set(pd.read_csv(idx_path)["date"]) if os.path.exists(idx_path) else set()
    poly = dc.district_polygon(d)
    bbox = list(poly.bounds)
    print(f"{d}: grid {shape}, usable cells {usable.sum():,}", flush=True)
    np.savez_compressed(os.path.join(out_dir, "_grid.npz"), usable=usable, inside=inside, dem_water=bad,
                        transform=np.array(tr)[:6])
    rows = []
    for y in YEARS:
        for mth in MONTHS:
            feats = search(bbox, y, mth)
            if not feats:
                continue
            by_date = {}
            # keep only tiles whose footprint overlaps the district (bbox search also returns
            # neighbouring tiles that cover none of it; reading those wasted most of the run time)
            feats = [f for f in feats if f.get("geometry") and
                     geom_shape(f["geometry"]).intersection(poly).area > 0.01 * poly.area]
            for f in feats:
                by_date.setdefault(f["properties"]["datetime"][:10], []).append(f)
            todo = [k for k in sorted(by_date) if k not in done]

            def work(day):
                try:
                    lab = scene_label(by_date[day], tr, shape, usable)
                except Exception as e:
                    print(f"  {day} FAILED {str(e)[:120]}", flush=True)
                    return None
                n_obs = int((lab != 255).sum())
                if n_obs == 0:
                    return {"date": day, "n_tiles": len(by_date[day]), "obs_frac": 0.0, "flood_frac_obs": np.nan}
                np.savez_compressed(os.path.join(out_dir, f"{day}.npz"), label=lab)
                return {"date": day, "n_tiles": len(by_date[day]), "obs_frac": n_obs / usable.sum(),
                        "flood_frac_obs": float((lab == 1).sum() / n_obs)}

            with ThreadPoolExecutor(6) as ex:
                res = [r for r in ex.map(work, todo) if r]
            for r in res:
                print(f"  {r['date']} obs={r['obs_frac']:.2f} flood={r['flood_frac_obs']:.4f}", flush=True)
            rows += res
            if res:
                old = pd.read_csv(idx_path) if os.path.exists(idx_path) else pd.DataFrame()
                pd.concat([old, pd.DataFrame(res)]).drop_duplicates("date").sort_values("date").to_csv(idx_path, index=False)
                done |= {r["date"] for r in res}
    print(f"{d}: done, {len(rows)} new scenes", flush=True)


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "all"
    for d in (dc.DISTRICTS if a == "all" else [a]):
        run(d)
