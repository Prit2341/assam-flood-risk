"""H6 filled-DEM test: terrain = RAW FABDEM (no depression filling) with the same 4 m burn on the same >=10 km2 stream pixels
(stream_mask.tif from the filled DEM). Rationale: Costabile et al. 2022 (Water 14:997) - depression filling unnecessary for 2D rain-on-grid,
effect case by case. Found 2026-11-05: filling raised the <100 m plains by 602 Mm3 over 693 km2 (one depression 322 Mm3 / 211 km2 near
Dikhowmukh) and 71% of the 07-17 false-alarm area sits on filled ground.
  python scripts/hecras_sivasagar/38_raw_burn.py      -> data/dem_raw.tif (raw FABDEM on the dem_filled grid), data/dem_rawburn_4m.tif"""
from pathlib import Path
import numpy as np, rasterio
from rasterio.windows import from_bounds
D = Path(__file__).resolve().parents[2] / "digital_twin/sivasagar/data"
RAW = Path(__file__).resolve().parents[2] / "data/basin_delineation/sivasagar_fabdem/dem_utm46.tif"
with rasterio.open(D / "dem_filled.tif") as r:
    zf = r.read(1); prof = r.profile; b = r.bounds; T = r.transform
with rasterio.open(RAW) as r:
    w = from_bounds(*b, transform=r.transform).round_offsets().round_lengths(); w = rasterio.windows.Window(w.col_off, w.row_off, zf.shape[1], zf.shape[0])
    assert abs(r.window_transform(w).c - T.c) < 1 and abs(r.window_transform(w).f - T.f) < 1, "grid mismatch"
    z = r.read(1, window=w).astype("float32"); nd = r.nodata
bad = ~np.isfinite(z) | ((z == nd) if nd is not None else False) | ~np.isfinite(zf)
z[bad] = zf[bad]                                  # keep filled value where raw is missing (edge)
with rasterio.open(D / "stream_mask.tif") as r:
    st = r.read(1) > 0
zb = z.copy(); zb[st] -= 4.0
print("raw px replaced by filled (nodata):", int(bad.sum()), " mean(filled-raw)", float(np.nanmean(zf - z)), " stream px", int(st.sum()))
for fn, a in [("dem_raw.tif", z), ("dem_rawburn_4m.tif", zb)]:
    with rasterio.open(D / fn, "w", **dict(prof, dtype="float32")) as o:
        o.write(a.astype("float32"), 1)
