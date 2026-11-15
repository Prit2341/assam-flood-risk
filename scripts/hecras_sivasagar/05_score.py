"""
Sivasagar HEC-RAS teacher, step 5: score a result HDF against a GFM Sentinel-1 flood scene (rule in docs/sivasagar_teacher_spec.md s3).

  python scripts/hecras_sivasagar/05_score.py results/<name>.p01.hdf 2026-07-17 [--thr 0.1]

Model flooded pixel = (cell water surface at the output hour nearest the scene time) - (unburned 30 m terrain at the pixel) > thr
(metres), the cell being the one whose centre is nearest to the 20 m GFM pixel centre (within 1.2 cell diagonals).
Observed = GFM 'flood' band (1 flood, 0 none, 255 no valid data). Excluded from scoring: no-data pixels, pixels outside the model domain,
and (--pw pre) pixels the GFM 2026-07-05 pre-event scene already flags as flood (pre-existing water).
NOT used any more: the Dynamic World class-0 'permanent water' flag. Found 2026-11-02: both DW label rasters are ~95% zeros and 0 is both the
nodata value and the water class, so that flag removed 98% of the in-domain pixels (valid area 30 km2) and the first scores were invalid.
Scene times (UTC, from the scene ids in data/gfm_validation/summary.csv): 07-05 23:38, 07-17 23:37, 07-22 11:39; 08-20 11:48 (GFM STAC scene ids ENSEMBLE_FLOOD_20260820T1147..; corrected 2026-11-05, was wrongly assumed 23:37).
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "digital_twin" / "sivasagar" / "data"
GFM = ROOT / "data" / "gfm_validation" / "Sivasagar"
DW_PRE = ROOT / "data" / "lulc_dynamicworld" / "Sivasagar" / "Sivasagar_2026_pre_2026-06-15_to_2026-07-14_label.tif"
SCENES = {"2026-07-05": ("pre_2026-07-05", "2026-07-05 23:38"), "2026-07-17": ("near_peak_a_2026-07-17", "2026-07-17 23:37"),
          "2026-07-22": ("near_peak_b_2026-07-22", "2026-07-22 11:39"), "2026-08-20": ("peak_2026-08-20", "2026-08-20 11:48")}
# 08-20 time corrected 2026-11-05: GFM STAC lists only ENSEMBLE_FLOOD_20260820T1147/1148 scenes over the domain that day (no 23:37 pass);
# all 08-20 scores before this fix were taken ~12 h late.
AREA = "Sivasagar_Domain"
TS = "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series"


def hdf_times(f):
    raw = f[f"{TS}/Time Date Stamp"][:]
    return pd.to_datetime([x.decode() if isinstance(x, bytes) else x for x in raw], format="%d%b%Y %H:%M:%S")


def pixel_cell_map(cell_xy, scene_tif, cache):
    """Per GFM pixel: UTM x,y, nearest cell id, in-domain flag, terrain z, permanent-water flag (cached per scene grid + mesh)."""
    if cache.exists():
        d = np.load(cache)
        return d["cid"], d["z"], d["dom"], d["pw"]
    with rasterio.open(scene_tif) as r:
        rows, cols = np.mgrid[0:r.height, 0:r.width]
        xs, ys = r.transform * (cols.ravel() + 0.5, rows.ravel() + 0.5)
        crs = r.crs
    to_utm = Transformer.from_crs(crs, "EPSG:32646", always_xy=True)
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    ux, uy = to_utm.transform(xs, ys)
    lon, lat = to_ll.transform(xs, ys)
    dist, cid = cKDTree(cell_xy).query(np.c_[ux, uy])
    with rasterio.open(DATA / "dem_filled.tif") as r:
        z = np.fromiter((v[0] for v in r.sample(zip(ux, uy), masked=False)), "float32", len(ux))
        inb = (ux > r.bounds.left) & (ux < r.bounds.right) & (uy > r.bounds.bottom) & (uy < r.bounds.top)
    poly = gpd.read_file(DATA / "boundary_utm46n.gpkg").geometry.iloc[0]
    from shapely import contains_xy
    dom = contains_xy(poly, ux, uy) & inb
    with rasterio.open(DW_PRE) as r:
        dw = np.fromiter((v[0] for v in r.sample(zip(lon, lat), masked=False)), "uint8", len(lon))
    pw = dw == 0
    np.savez_compressed(cache, cid=cid.astype("int32"), z=z, dom=dom, pw=pw)
    return cid, z, dom, pw


def raw_z(scene_tif, n):
    """Pixel terrain from data/dem_raw.tif (unfilled FABDEM, 38_raw_burn.py) instead of dem_filled.tif; cached per scene grid (added 2026-11-05, H6)."""
    cache = DATA / f"rawz_{scene_tif.stem}.npy"
    if cache.exists():
        return np.load(cache)
    with rasterio.open(scene_tif) as r:
        rows, cols = np.mgrid[0:r.height, 0:r.width]
        xs, ys = r.transform * (cols.ravel() + 0.5, rows.ravel() + 0.5)
        crs = r.crs
    ux, uy = Transformer.from_crs(crs, "EPSG:32646", always_xy=True).transform(xs, ys)
    with rasterio.open(DATA / "dem_raw.tif") as r:
        a = r.read(1); T = r.transform
    c = np.floor((ux - T.c) / T.a).astype(int); rr = np.floor((uy - T.f) / T.e).astype(int)
    ok = (c >= 0) & (c < a.shape[1]) & (rr >= 0) & (rr < a.shape[0])
    z = np.full(len(ux), np.nan, "float32"); z[ok] = a[rr[ok], c[ok]]
    np.save(cache, z)
    return z


def score(hdf, scene, thr=0.1, pwmode="pre", terrain="filled"):
    tag, when = SCENES[scene]
    tif = next(GFM.glob(f"*{tag}*_flood.tif"))
    with h5py.File(hdf, "r") as f:
        cxy = f[f"Geometry/2D Flow Areas/{AREA}/Cells Center Coordinate"][:]
        ncell = f[f"Geometry/2D Flow Areas/{AREA}/Cells Minimum Elevation"].shape[0]
        times = hdf_times(f)
        i = int(np.argmin(np.abs(times - pd.Timestamp(when))))
        wse = f[f"{TS}/2D Flow Areas/{AREA}/Water Surface"][i, :]
    n = min(len(wse), len(cxy))
    cache = DATA / f"pixmap_{tif.stem}_{n}.npz"
    cid, z, dom, pw = pixel_cell_map(cxy[:n], tif, cache)
    if terrain == "raw":
        z = raw_z(tif, n)
    with rasterio.open(tif) as r:
        obs = r.read(1).ravel()
    if pwmode == "pre":
        with rasterio.open(next(GFM.glob("*pre_2026-07-05*_flood.tif"))) as r:
            pre = r.read(1).ravel()
        assert pre.size == obs.size, "pre-event scene is on a different grid"
        pw = pre == 1
    else:
        pw = np.zeros(obs.size, bool)
    valid = (obs != 255) & dom & ~pw & np.isfinite(z)
    mod = (wse[cid] - z) > thr
    o, m = obs[valid] == 1, mod[valid]
    hits, miss, fa = int((o & m).sum()), int((o & ~m).sum()), int((~o & m).sum())
    out = dict(result=Path(hdf).name, scene=scene, pw=pwmode, terrain=terrain, output_time=str(times[i]), valid_km2=round(valid.sum() * 4e-4, 1),
               obs_flood_km2=round(o.sum() * 4e-4, 1), model_flood_km2=round(m.sum() * 4e-4, 1), hits=hits, misses=miss, false_alarms=fa,
               CSI=round(hits / max(hits + miss + fa, 1), 3), area_ratio=round(m.sum() / max(o.sum(), 1), 2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf")
    ap.add_argument("scene")
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--pw", choices=["pre", "none"], default="pre")
    ap.add_argument("--terrain", choices=["filled", "raw"], default="filled", help="pixel terrain for depth: dem_filled.tif or dem_raw.tif")
    a = ap.parse_args()
    print(score(a.hdf, a.scene, a.thr, a.pw, a.terrain))
