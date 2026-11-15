"""
Score a configuration on the pre-registered HELD-OUT GFM set (Chroma manual-0238): all passes in data/gfm_domain/passes.csv
(07-06..07-22 and 08-09..08-22, minus the used 07-17 / 07-22 11:39 / 08-20), clipped to the model domain. Canonical terrain = raw
FABDEM (dem_raw.tif). Pre-existing water = the 07-05 23:3x pass (gfm_20260705T23xx.tif). Model wet pixel = cell WSE (output hour
nearest the pass) - raw z > 0.1 m, cell = nearest cell centre.
  python scripts/hecras_sivasagar/42_score_heldout.py <july_hdf> <aug_hdf> [label]
Out: prints per-pass and pooled hits / misses / FA (km2) and CSI; appends rows to results/heldout_scores.csv
"""
import sys
from pathlib import Path
import h5py, numpy as np, pandas as pd, rasterio, geopandas as gpd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely import contains_xy

ROOT = Path(__file__).resolve().parents[2]; S = ROOT / "digital_twin" / "sivasagar"; D = S / "data"; G = D / "gfm_domain"; RES = S / "results"
AREA = "Sivasagar_Domain"; TS = "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series"
jul, aug = Path(sys.argv[1]), Path(sys.argv[2]); label = sys.argv[3] if len(sys.argv) > 3 else jul.stem
passes = pd.read_csv(G / "passes.csv"); passes["t"] = pd.to_datetime(passes.pass_time).dt.tz_convert(None)
pre_row = passes[passes.key.str.startswith("20260705")]
assert len(pre_row) >= 1, "pre-event 07-05 pass missing"
held = passes[~passes.key.str.startswith("20260705")].copy()

ref = rasterio.open(G / pre_row.file.iloc[0]); H, W, T, crs = ref.height, ref.width, ref.transform, ref.crs
cache = D / f"gfm_domain_pixmap_{H}x{W}.npz"
if cache.exists():
    c = np.load(cache); idx, ux, uy = c["idx"], c["ux"], c["uy"]
else:
    rows, cols = np.mgrid[0:H, 0:W]; xs, ys = T * (cols.ravel() + .5, rows.ravel() + .5)
    ux, uy = Transformer.from_crs(crs, "EPSG:32646", always_xy=True).transform(xs, ys)
    poly = gpd.read_file(D / "boundary_utm46n.gpkg").geometry.iloc[0]
    idx = np.nonzero(contains_xy(poly, ux, uy))[0]; ux, uy = ux[idx], uy[idx]
    np.savez_compressed(cache, idx=idx, ux=ux.astype("float64"), uy=uy.astype("float64"))
with rasterio.open(D / "dem_raw.tif") as r:
    a = r.read(1); TR = r.transform
cc = np.floor((ux - TR.c) / TR.a).astype(int); rr = np.floor((uy - TR.f) / TR.e).astype(int)
ok = (cc >= 0) & (cc < a.shape[1]) & (rr >= 0) & (rr < a.shape[0]); z = np.full(ux.size, np.nan, "float32"); z[ok] = a[rr[ok], cc[ok]]
pre = ref.read(1).reshape(-1)[idx]

def load(h):
    with h5py.File(h, "r") as f:
        cxy = f[f"Geometry/2D Flow Areas/{AREA}/Cells Center Coordinate"][:]
        t = pd.to_datetime([x.decode() for x in f[f"{TS}/Time Date Stamp"][:]], format="%d%b%Y %H:%M:%S")
        wse = f[f"{TS}/2D Flow Areas/{AREA}/Water Surface"][:]
    n = min(wse.shape[1], len(cxy)); _, cid = cKDTree(cxy[:n]).query(np.c_[ux, uy])
    return t, wse, cid
M = {"jul": load(jul), "aug": load(aug)}
out = []; tot = np.zeros(3)
for _, p in held.iterrows():
    t, wse, cid = M["jul" if p.t.month == 7 else "aug"]
    if p.t < t[0] or p.t > t[-1]:
        print("outside run window", p.key); continue
    i = int(np.argmin(np.abs(t - p.t)))
    obs = rasterio.open(G / p.file).read(1).reshape(-1)[idx]
    v = (obs != 255) & (pre != 1) & np.isfinite(z)
    if v.sum() * 4e-4 < 100:
        print("skip (valid < 100 km2)", p.key, round(v.sum() * 4e-4, 1)); continue
    m = (wse[i][cid] - z) > 0.1; o = obs == 1
    h_, mi, fa = int((v & o & m).sum()), int((v & o & ~m).sum()), int((v & ~o & m).sum())
    tot += (h_, mi, fa)
    out.append(dict(label=label, pass_key=p.key, model_time=str(t[i]), valid_km2=round(v.sum() * 4e-4, 1), obs_km2=round((v & o).sum() * 4e-4, 1),
                    model_km2=round((v & m).sum() * 4e-4, 1), hits=round(h_ * 4e-4, 1), misses=round(mi * 4e-4, 1), fa=round(fa * 4e-4, 1),
                    CSI=round(h_ / max(h_ + mi + fa, 1), 3)))
df = pd.DataFrame(out); print(df.to_string(index=False))
h_, mi, fa = tot
print(f"POOLED {label}: hits {h_ * 4e-4:.1f} misses {mi * 4e-4:.1f} FA {fa * 4e-4:.1f} km2  CSI {h_ / max(h_ + mi + fa, 1):.3f}  "
      f"area ratio {(h_ + fa) / max(h_ + mi, 1):.2f}  scenes {len(df)}")
df.assign(pooled_CSI=round(h_ / max(h_ + mi + fa, 1), 3)).to_csv(RES / "heldout_scores.csv", mode="a", header=not (RES / "heldout_scores.csv").exists(), index=False)
