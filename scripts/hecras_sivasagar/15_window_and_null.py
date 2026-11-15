"""(1) DRIMS comparison with the max taken only over 19 Jul 00:00 - 16 Aug 24:00 output hours.
(2) Null baseline: terrain-only map (HAND < h) whose district cropland area equals 283.6 km2; scored on GFM 07-17 (same valid mask/rule as 05_score.py)."""
import sys
from pathlib import Path
import geopandas as gpd, h5py, numpy as np, pandas as pd, rasterio
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from shapely import contains_xy
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib; sc = importlib.import_module("05_score")
D = sc.ROOT; DATA = sc.DATA
f = h5py.File(sys.argv[1]); A = sc.AREA
cxy = f[f"Geometry/2D Flow Areas/{A}/Cells Center Coordinate"][:]
times = sc.hdf_times(f)
ws = f[f"{sc.TS}/2D Flow Areas/{A}/Water Surface"]
sel = np.where((times >= "2026-07-19") & (times <= "2026-08-17 00:00"))[0]
mxw = np.max(ws[sel[0]:sel[-1] + 1, :], axis=0); n = min(len(mxw), len(cxy)); cxy, mxw = cxy[:n], mxw[:n]
print("window hours", times[sel[0]], "->", times[sel[-1]], len(sel))
with rasterio.open(DATA / "worldcover_100m.tif") as r:
    wc = r.read(1); T = r.transform
rr, cc = np.mgrid[0:wc.shape[0], 0:wc.shape[1]]; x, y = T * (cc.ravel() + .5, rr.ravel() + .5)
with rasterio.open(DATA / "dem_filled.tif") as r:
    dz = r.read(1); T3 = r.transform
c3 = ((x - T3.c) / T3.a).astype(int); r3 = ((y - T3.f) / T3.e).astype(int)
ok = (c3 >= 0) & (c3 < dz.shape[1]) & (r3 >= 0) & (r3 < dz.shape[0]); c3 = np.clip(c3, 0, dz.shape[1] - 1); r3 = np.clip(r3, 0, dz.shape[0] - 1)
z = dz[r3, c3]
with rasterio.open(DATA / "stream_mask.tif") as r:
    st = r.read(1) > 0
d, (ri, ci) = distance_transform_edt(~st, return_indices=True)
hand100 = dz[r3, c3] - dz[ri[r3, c3], ci[r3, c3]]
dom = contains_xy(gpd.read_file(DATA / "boundary_utm46n.gpkg").geometry.iloc[0], x, y) & ok & (z > -999)
dist, cid = cKDTree(cxy).query(np.c_[x, y]); dom &= dist < 720
dis = gpd.read_file(D / "data/boundaries/assam_districts.geojson"); dis = dis[dis.dtname.str.contains("Sibsagar|Sivasagar", case=False)].to_crs(32646)
ind = contains_xy(dis.geometry.union_all(), x, y) & dom; crop = wc.ravel() == 40
depth = mxw[cid] - z
print("windowed max depth, district cropland km2:", {t: round((ind & crop & (depth > t)).sum() * .01) for t in (0.1, 0.3, 0.5)}, " (DRIMS 283.6)")
# null baseline
tgt = 283.6
hs = np.linspace(0.05, 20, 400)
areas = np.array([(ind & crop & (hand100 < h)).sum() * .01 for h in hs])
h0 = hs[np.argmin(np.abs(areas - tgt))]; print("HAND cut-off giving district cropland ~284 km2: %.2f m (area %.0f)" % (h0, areas[np.argmin(np.abs(areas - tgt))]))
tag, when = sc.SCENES["2026-07-17"]; tif = next(sc.GFM.glob(f"*{tag}*_flood.tif"))
with rasterio.open(tif) as r:
    obs = r.read(1).ravel(); H, W = r.height, r.width
    rows, cols = np.mgrid[0:H, 0:W]; xs, ys = r.transform * (cols.ravel() + .5, rows.ravel() + .5); crs = r.crs
with rasterio.open(next(sc.GFM.glob("*pre_2026-07-05*_flood.tif"))) as r:
    pre = r.read(1).ravel()
ux, uy = Transformer.from_crs(crs, "EPSG:32646", always_xy=True).transform(xs, ys)
cg = ((ux - T3.c) / T3.a).astype(int); rg = ((uy - T3.f) / T3.e).astype(int)
okg = (cg >= 0) & (cg < dz.shape[1]) & (rg >= 0) & (rg < dz.shape[0]); cg = np.clip(cg, 0, dz.shape[1] - 1); rg = np.clip(rg, 0, dz.shape[0] - 1)
hg = dz[rg, cg] - dz[ri[rg, cg], ci[rg, cg]]
domg = contains_xy(gpd.read_file(DATA / "boundary_utm46n.gpkg").geometry.iloc[0], ux, uy) & okg & (dz[rg, cg] > -999)
valid = (obs != 255) & domg & (pre != 1); o = obs == 1
print("HAND null on 07-17 (same valid mask as 05_score), valid km2 %.0f obs %.1f:" % (valid.sum() * 4e-4, (valid & o).sum() * 4e-4))
for h in (0.25, 0.5, 1.0, 1.5, 2.0, h0, 3.0):
    m = hg < h; hit, mis, fa = (valid & o & m).sum(), (valid & o & ~m).sum(), (valid & ~o & m).sum()
    print("  HAND < %.2f m: model %.0f km2, CSI %.3f, area ratio %.2f" % (h, (valid & m).sum() * 4e-4, hit / max(hit + mis + fa, 1), (valid & m).sum() / max((valid & o).sum(), 1)))

# equal-area null: rank GFM-valid pixels by HAND (ties by distance to stream), flood the lowest-ranked pixels until the area equals the HEC-RAS model's flooded area on this scene
dg = d[rg, cg]; key = hg + 1e-4 * dg
for area_km2, label in [(343.4, "gridded run 343.4"), (331.0, "slope run 331"), (350.8, "c400 350.8"), (52.0, "equal to observed 52")]:
    k = int(area_km2 / 4e-4); idx = np.where(valid)[0]; sel_ = idx[np.argsort(key[idx])[:k]]
    m = np.zeros(obs.size, bool); m[sel_] = True
    hit, mis, fa = (valid & o & m).sum(), (valid & o & ~m).sum(), (valid & ~o & m).sum()
    print("equal-area HAND null (%s km2): CSI %.3f, hits %.1f km2" % (label, hit / max(hit + mis + fa, 1), hit * 4e-4))
