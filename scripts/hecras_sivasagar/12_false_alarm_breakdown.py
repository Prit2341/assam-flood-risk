"""False-alarm anatomy (scoring-only, no HEC-RAS): where do model-flooded / GFM-dry pixels sit?
  python scripts/hecras_sivasagar/12_false_alarm_breakdown.py results/<hdf> 2026-07-17
Buckets: distance to nearest 30 m stream pixel, HAND (pixel z minus z of nearest stream pixel), cell crossed by the stream network, model depth.
Reports hits / misses / false alarms per bucket, so a bucket with many false alarms and no hits is a candidate artefact."""
import sys
from pathlib import Path
import h5py, numpy as np, pandas as pd, rasterio
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib; sc = importlib.import_module("05_score")

hdf, scene = sys.argv[1], sys.argv[2]
tag, when = sc.SCENES[scene]
tif = next(sc.GFM.glob(f"*{tag}*_flood.tif"))
with h5py.File(hdf, "r") as f:
    cxy = f[f"Geometry/2D Flow Areas/{sc.AREA}/Cells Center Coordinate"][:]
    times = sc.hdf_times(f); i = int(np.argmin(np.abs(times - pd.Timestamp(when))))
    wse = f[f"{sc.TS}/2D Flow Areas/{sc.AREA}/Water Surface"][i, :]
n = min(len(wse), len(cxy))
cid, z, dom, _ = sc.pixel_cell_map(cxy[:n], tif, sc.DATA / f"pixmap_{tif.stem}_{n}.npz")
with rasterio.open(tif) as r:
    obs = r.read(1).ravel(); H, W = r.height, r.width
    rows, cols = np.mgrid[0:H, 0:W]; xs, ys = r.transform * (cols.ravel() + .5, rows.ravel() + .5)
    crs = r.crs
with rasterio.open(next(sc.GFM.glob("*pre_2026-07-05*_flood.tif"))) as r:
    pre = r.read(1).ravel()
ux, uy = Transformer.from_crs(crs, "EPSG:32646", always_xy=True).transform(xs, ys)
valid = (obs != 255) & dom & (pre != 1) & np.isfinite(z)
depth = wse[cid] - z; mod = depth > 0.1; o = obs == 1
with rasterio.open(sc.DATA / "stream_mask.tif") as r:
    st = r.read(1) > 0; T = r.transform; dem = r.read(1) if False else None
with rasterio.open(sc.DATA / "dem_filled.tif") as r:
    dz = r.read(1)
d, (ri, ci) = distance_transform_edt(~st, return_indices=True)
c = ((ux - T.c) / T.a).astype(int); rr = ((uy - T.f) / T.e).astype(int)
ok = (c >= 0) & (c < st.shape[1]) & (rr >= 0) & (rr < st.shape[0]); valid &= ok
c = np.clip(c, 0, st.shape[1] - 1); rr = np.clip(rr, 0, st.shape[0] - 1)
dist_m = d[rr, c] * 30.0
hand = dz[rr, c] - dz[ri[rr, c], ci[rr, c]]
cell_has_stream = np.zeros(n, bool); sm = valid & st[rr, c]; cell_has_stream[np.unique(cid[sm])] = True
def tab(name, labels, edges):
    b = np.digitize(labels, edges); rows_ = []
    for k in range(len(edges) + 1):
        s = valid & (b == k)
        if s.sum() == 0: continue
        h, m_, f_ = (s & o & mod).sum(), (s & o & ~mod).sum(), (s & ~o & mod).sum()
        lo = edges[k - 1] if k else "-inf"; hi = edges[k] if k < len(edges) else "inf"
        rows_.append((f"{lo}..{hi}", round(s.sum() * 4e-4, 1), round(h * 4e-4, 1), round(m_ * 4e-4, 1), round(f_ * 4e-4, 1)))
    print(f"\n{name}  (km2: valid, hits, misses, false alarms)")
    for r_ in rows_: print("  %-14s valid %8.1f  hits %6.1f  miss %6.1f  FA %7.1f" % r_)
tab("distance to stream (m)", dist_m, [30, 90, 300, 900, 3000])
tab("HAND (m above nearest stream pixel)", hand, [1, 3, 10, 30])
tab("model depth (m)", depth, [0.1, 0.3, 1, 3])
for flag, nm in [(True, "cell crossed by stream network"), (False, "cell NOT crossed")]:
    s = valid & (cell_has_stream[cid] == flag)
    print(f"{nm}: valid {s.sum()*4e-4:.1f}  hits {(s&o&mod).sum()*4e-4:.1f}  miss {(s&o&~mod).sum()*4e-4:.1f}  FA {(s&~o&mod).sum()*4e-4:.1f} km2")

# --- land cover (ESA WorldCover v200) and JRC surface-water occurrence, sampled at 100 m (13_pull_landcover.py)
def samp(fn):
    with rasterio.open(sc.DATA / fn) as r:
        a = r.read(1); T2 = r.transform
    c2 = ((ux - T2.c) / T2.a).astype(int); r2 = ((uy - T2.f) / T2.e).astype(int)
    k = (c2 >= 0) & (c2 < a.shape[1]) & (r2 >= 0) & (r2 < a.shape[0])
    out = np.zeros(ux.size, a.dtype); out[k] = a[r2[k], c2[k]]; return out
wcv = samp("worldcover_100m.tif"); occ = samp("gsw_occurrence_100m.tif")
names = {10: "tree", 20: "shrub", 30: "grass", 40: "cropland", 50: "built", 60: "bare", 80: "water", 90: "wetland", 95: "mangrove"}
print("\nWorldCover class (km2: valid, hits, misses, false alarms)")
for v in np.unique(wcv[valid]):
    s = valid & (wcv == v)
    print("  %-10s valid %8.1f  hits %6.1f  miss %6.1f  FA %7.1f" % (names.get(int(v), v), s.sum() * 4e-4, (s & o & mod).sum() * 4e-4, (s & o & ~mod).sum() * 4e-4, (s & ~o & mod).sum() * 4e-4))
tab("JRC GSW occurrence (% of 1984-2021 months with water)", occ.astype(float), [1, 10, 50])
