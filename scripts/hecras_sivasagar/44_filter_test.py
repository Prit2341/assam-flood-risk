"""
Puddle test (no HEC-RAS): does removing shallow / disconnected wet cells raise skill? (advisor 2026-11-05; Chroma manual-0243)
Filters applied to model wet pixels (cell WSE - raw z > 0.1 m):
  depth d : keep only cells with WSE - cell minimum elevation >= d
  conn    : keep only wet cells connected (through faces between wet cells) to a channel cell (cell crossed by stream_mask.tif)
Scenes: CALIBRATION = the already-used district-clipped scenes 07-17 23:37, 07-22 11:39, 08-20 11:48 (05_score.py grids);
        HELD-OUT    = data/gfm_domain passes (42_score_heldout.py set). Raw terrain everywhere.
  python scripts/hecras_sivasagar/44_filter_test.py calib <july_hdf> <aug_hdf>
  python scripts/hecras_sivasagar/44_filter_test.py heldout <july_hdf> <aug_hdf> <filter e.g. d0.5 | conn | d0.5+conn | none>
"""
import sys, glob, importlib
from pathlib import Path
import h5py, numpy as np, pandas as pd, rasterio
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
sc = importlib.import_module("05_score")
S = HERE.parents[1] / "digital_twin" / "sivasagar"; D = S / "data"; G = D / "gfm_domain"
mode, jul, aug = sys.argv[1], sys.argv[2], sys.argv[3]
N = 26038
g = np.load(S / "surrogate" / "static_graph.npz"); fc = g["face_cells"]
fc = fc[(fc[:, 0] < N) & (fc[:, 1] < N) & (fc[:, 0] >= 0) & (fc[:, 1] >= 0)]
with rasterio.open(D / "stream_mask.tif") as r:
    st = r.read(1) > 0; T = r.transform
rr, cc = np.nonzero(st); x, y = T * (cc + .5, rr + .5)

def load(h):
    with h5py.File(h, "r") as f:
        A = f"Geometry/2D Flow Areas/{sc.AREA}/"; cxy = f[A + "Cells Center Coordinate"][:N]; zmin = f[A + "Cells Minimum Elevation"][:N]
        t = sc.hdf_times(f); wse = f[f"{sc.TS}/2D Flow Areas/{sc.AREA}/Water Surface"][:, :N]
    return cxy, zmin, t, wse
R = {"jul": load(jul), "aug": load(aug)}
cxy = R["jul"][0]; tree = cKDTree(cxy)
_, chcell = tree.query(np.c_[x, y]); chan = np.zeros(N, bool); chan[np.unique(chcell)] = True

def keep_cells(wse, zmin, flt, wetcell):
    k = wetcell.copy()
    for part in flt.split("+"):
        if part.startswith("d"):
            k &= (wse - zmin) >= float(part[1:])
        elif part == "conn":
            e = fc[k[fc[:, 0]] & k[fc[:, 1]]]
            A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(N, N))
            _, lab = connected_components(A, directed=False)
            good = np.unique(lab[k & chan]); k &= np.isin(lab, good)
    return k

def score_set(scenes, flt):
    tot = np.zeros(3); rows = []
    for name, cid, z, obs, valid, when in scenes:
        cxy_, zmin, t, wse_all = R["jul" if when.month == 7 else "aug"]
        i = int(np.argmin(np.abs(t - when))); wse = wse_all[i]
        pixwet = (wse[cid] - z) > 0.1
        wetcell = np.bincount(cid[pixwet & valid], minlength=N)[:N] > 0
        wetcell |= (wse - zmin) > 0.1
        kc = keep_cells(wse, zmin, flt, wetcell) if flt != "none" else np.ones(N, bool)
        m = pixwet & kc[cid]; o = obs == 1
        h, mi, fa = int((valid & o & m).sum()), int((valid & o & ~m).sum()), int((valid & ~o & m).sum())
        tot += (h, mi, fa); rows.append((name, round(h * 4e-4, 1), round(mi * 4e-4, 1), round(fa * 4e-4, 1), round(h / max(h + mi + fa, 1), 3)))
    h, mi, fa = tot
    return rows, (round(h * 4e-4, 1), round(mi * 4e-4, 1), round(fa * 4e-4, 1), round(h / max(h + mi + fa, 1), 3))

if mode == "calib":
    scenes = []
    pre = rasterio.open(next(sc.GFM.glob("*pre_2026-07-05*_flood.tif"))).read(1).reshape(-1)
    for s in ["2026-07-17", "2026-07-22", "2026-08-20"]:
        tag, when = sc.SCENES[s]; tif = next(sc.GFM.glob(f"*{tag}*_flood.tif"))
        obs = rasterio.open(tif).read(1).reshape(-1)
        cid, _, dom, _ = sc.pixel_cell_map(cxy, tif, sc.DATA / f"pixmap_{tif.stem}_26814.npz")
        z = sc.raw_z(tif, 0); valid = (obs != 255) & dom & (pre != 1) & np.isfinite(z) & (cid < N)
        cid = np.minimum(cid, N - 1)
        scenes.append((s, cid, z, obs, valid, pd.Timestamp(when)))
    for flt in ["none", "d0.3", "d0.5", "d1.0", "conn", "d0.3+conn", "d0.5+conn"]:
        rows, pooled = score_set(scenes, flt)
        print(f"{flt:10s} pooled hits/miss/FA/CSI {pooled}  per-scene CSI {[r[4] for r in rows]}")
else:
    flt = sys.argv[4]
    c = np.load(glob.glob(str(D / "gfm_domain_pixmap_*.npz"))[0]); idx, ux, uy = c["idx"], c["ux"], c["uy"]
    with rasterio.open(D / "dem_raw.tif") as r:
        a = r.read(1); TR = r.transform
    z = a[np.floor((uy - TR.f) / TR.e).astype(int), np.floor((ux - TR.c) / TR.a).astype(int)]
    _, cid = tree.query(np.c_[ux, uy])
    P = pd.read_csv(G / "passes.csv"); P["t"] = pd.to_datetime(P.pass_time).dt.tz_convert(None)
    pre = rasterio.open(G / P[P.key.str.startswith("20260705")].file.iloc[0]).read(1).reshape(-1)[idx]
    scenes = []
    for _, p in P[~P.key.str.startswith("20260705")].iterrows():
        obs = rasterio.open(G / p.file).read(1).reshape(-1)[idx]; valid = (obs != 255) & (pre != 1) & np.isfinite(z)
        if valid.sum() * 4e-4 < 100: continue
        scenes.append((p.key, cid, z, obs, valid, p.t))
    for f_ in (["none", flt] if flt != "none" else ["none"]):
        rows, pooled = score_set(scenes, f_)
        print(f"HELD-OUT {f_:10s} pooled hits/miss/FA/CSI {pooled}"); [print("   ", r) for r in rows]
