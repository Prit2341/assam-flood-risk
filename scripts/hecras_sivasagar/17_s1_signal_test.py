"""Do model-flooded / GFM-dry ('false alarm') pixels carry a flood signal in the raw Sentinel-1 backscatter?
On the 100 m grid of data/s1_*_100m.tif (16_pull_s1.py): classes hit / miss / FA / TN from the model (depth > 0.1 m at the scene hour) and GFM flood (sampled at the 100 m pixel centre).
Reports mean VV, VH (dB) at the scene date and the change from the 07-05 pre pass, plus AUC of each variable for 'model wet' among GFM-dry cropland (0.5 = none).
  python scripts/hecras_sivasagar/17_s1_signal_test.py results/<hdf> 2026-07-17 0717"""
import sys
from pathlib import Path
import geopandas as gpd, h5py, numpy as np, pandas as pd, rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely import contains_xy
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib; sc = importlib.import_module("05_score")
hdf, scene, tag = sys.argv[1], sys.argv[2], sys.argv[3]
DATA = sc.DATA; tg, when = sc.SCENES[scene]
f = h5py.File(hdf); A = sc.AREA
cxy = f[f"Geometry/2D Flow Areas/{A}/Cells Center Coordinate"][:]; times = sc.hdf_times(f)
i = int(np.argmin(np.abs(times - pd.Timestamp(when)))); wse = f[f"{sc.TS}/2D Flow Areas/{A}/Water Surface"][i, :]
n = min(len(wse), len(cxy)); cxy, wse = cxy[:n], wse[:n]
with rasterio.open(DATA / f"s1_{tag}_100m.tif") as r:
    s1 = r.read().astype("float64"); T = r.transform
with rasterio.open(DATA / "s1_0705_100m.tif") as r:
    s0 = r.read().astype("float64")
H, W = s1.shape[1:]; rr, cc = np.mgrid[0:H, 0:W]; x, y = T * (cc.ravel() + .5, rr.ravel() + .5)
dist, cid = cKDTree(cxy).query(np.c_[x, y])
with rasterio.open(DATA / "dem_filled.tif") as r:
    z = np.fromiter((v[0] for v in r.sample(zip(x, y), masked=False)), "float32", x.size)
dom = contains_xy(gpd.read_file(DATA / "boundary_utm46n.gpkg").geometry.iloc[0], x, y) & (z > -999) & (dist < 720)
tif = next(sc.GFM.glob(f"*{tg}*_flood.tif")); pre_t = next(sc.GFM.glob("*pre_2026-07-05*_flood.tif"))
def sample(fn):
    with rasterio.open(fn) as r:
        gx, gy = Transformer.from_crs("EPSG:32646", r.crs, always_xy=True).transform(x, y)
        return np.fromiter((v[0] for v in r.sample(zip(gx, gy), masked=False)), "uint8", x.size)
obs = sample(tif); pre = sample(pre_t)
with rasterio.open(DATA / "worldcover_100m.tif") as r:
    wcT = r.transform; wc = r.read(1)
c2 = ((x - wcT.c) / wcT.a).astype(int); r2 = ((y - wcT.f) / wcT.e).astype(int)
ok = (c2 >= 0) & (c2 < wc.shape[1]) & (r2 >= 0) & (r2 < wc.shape[0]); lc = np.zeros(x.size, "uint8"); lc[ok] = wc[r2[ok], c2[ok]]
vv, vh = s1[0].ravel(), s1[1].ravel(); vv0, vh0 = s0[0].ravel(), s0[1].ravel()
valid = dom & (obs != 255) & (pre != 1) & np.isfinite(vv) & np.isfinite(vh) & np.isfinite(vv0) & np.isfinite(vh0)
mod = (wse[cid] - z) > 0.1; o = obs == 1
var = {"VV": vv, "VH": vh, "dVV": vv - vv0, "dVH": vh - vh0, "VH-VV": vh - vv}
for name, sel in [("cropland", lc == 40), ("tree cover", lc == 10), ("all land", lc > 0)]:
    s = valid & sel
    print(f"\n== {name} (n={s.sum()} 100 m pixels, {s.sum()*.01:.0f} km2)")
    print("  class  n_px   " + "  ".join(f"{k:>7}" for k in var))
    for cl, m in [("hit", o & mod), ("miss", o & ~mod), ("FA", ~o & mod), ("TN", ~o & ~mod)]:
        q = s & m
        if q.sum() < 20: continue
        print(f"  {cl:5} {q.sum():6d}  " + "  ".join(f"{np.nanmean(v[q]):7.2f}" for v in var.values()))
    q = s & ~o
    if (q & mod).sum() > 20 and (q & ~mod).sum() > 20:
        print("  AUC for 'model wet' among GFM-dry pixels: " + "  ".join(f"{k} {roc_auc_score(mod[q], v[q]):.3f}" for k, v in var.items()))
    q = s
    if (q & o).sum() > 20 and (q & ~o).sum() > 20:
        print("  AUC for 'GFM flood' among all pixels:     " + "  ".join(f"{k} {roc_auc_score(o[q], v[q]):.3f}" for k, v in var.items()))
