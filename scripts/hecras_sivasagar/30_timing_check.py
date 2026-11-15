"""H1b timing check (scoring-only, no HEC-RAS run).
Question: does the observed GFM flood fraction rise AFTER the model's rain-driven flood has receded (lag => river/backwater-driven
flooding not in the model), or do they move together?
Pre-registered (2026-11-05): if, on pixels valid in BOTH scenes, observed flood fraction increases 07-17 -> 07-22 while the model wet
fraction decreases, local rain-on-grid timing does not explain the observations (supports riverine/upstream forcing, H2).
  python scripts/hecras_sivasagar/30_timing_check.py
"""
import sys
from pathlib import Path
import h5py, numpy as np, pandas as pd, rasterio
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
sc = importlib.import_module("05_score")

HDF = sc.ROOT / "digital_twin" / "sivasagar" / "results" / "c600_d4_r30_0705_432h_cal.p01.hdf"
S = ["2026-07-17", "2026-07-22"]
sets = {}
for s in S:
    tag, when = sc.SCENES[s]
    tif = next(sc.GFM.glob(f"*{tag}*_flood.tif"))
    with rasterio.open(tif) as r:
        obs = r.read(1).reshape(-1)
        sets[s] = dict(obs=obs, shape=(r.height, r.width), tr=tuple(r.transform), tif=tif)
    print(s, sets[s]["shape"], sets[s]["tr"][:6])
with rasterio.open(next(sc.GFM.glob("*pre_2026-07-05*_flood.tif"))) as r:
    pre = r.read(1).reshape(-1)

with h5py.File(HDF, "r") as f:
    cxy = f[f"Geometry/2D Flow Areas/{sc.AREA}/Cells Center Coordinate"][:]
    times = sc.hdf_times(f)
    wse = f[f"{sc.TS}/2D Flow Areas/{sc.AREA}/Water Surface"][:]
n = min(wse.shape[1], len(cxy))
print("hdf", wse.shape, times[0], times[-1])

same_grid = sets[S[0]]["shape"] == sets[S[1]]["shape"] and sets[S[0]]["tr"] == sets[S[1]]["tr"]
print("same grid:", same_grid)
pm = {}
for s in S:
    d = np.load(sc.DATA / f"pixmap_{sets[s]['tif'].stem}_{n}.npz")
    pm[s] = (d["cid"], d["z"], d["dom"])
v = {s: (sets[s]["obs"] != 255) & pm[s][2] & (pre != 1) & np.isfinite(pm[s][1]) for s in S} if same_grid and pre.size == sets[S[0]]["obs"].size else None
if v is None:
    print("grids differ or pre mismatch; using each scene's own valid mask")
    v = {s: (sets[s]["obs"] != 255) & pm[s][2] & np.isfinite(pm[s][1]) for s in S}
common = v[S[0]] & v[S[1]] if same_grid else None
print({s: int(v[s].sum()) for s in S}, "common", None if common is None else int(common.sum()))

# model wet fraction vs time on each scene's pixels (common mask if available)
rows = []
cid, z, _ = pm[S[0]]
mask = common if common is not None else v[S[0]]
for k in range(0, len(times), 6):
    w = (wse[k][cid[mask]] - z[mask]) > 0.1
    rows.append((str(times[k]), round(w.mean() * 100, 2)))
df = pd.DataFrame(rows, columns=["time", "model_wet_pct"])
obs_frac = {s: round((sets[s]["obs"][mask] == 1).mean() * 100, 2) for s in S}
print("observed flood % on mask:", obs_frac)
print(df.iloc[::2].to_string(index=False))
rain = pd.read_csv(sc.DATA / "era5land_domain_hourly_2026-07-01_2026-08-26.csv", parse_dates=["timestamp"]).set_index("timestamp").tp_mm
print("daily rain mm (raw ERA5-Land) 07-05..07-23:")
print(rain["2026-07-05":"2026-07-23"].resample("D").sum().round(1).to_string())
