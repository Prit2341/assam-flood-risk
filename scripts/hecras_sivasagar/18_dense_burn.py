"""Denser drainage network test: burn 4 m along every 30 m cell with >= THR km2 upstream (baseline THR = 10 km2). Writes data/dem_burn_9m.tif (label 9 = dense 4 m) and stream_mask_dense.tif.
  python scripts/hecras_sivasagar/18_dense_burn.py 1.0"""
import sys
from pathlib import Path
import numpy as np, rasterio
from rasterio.windows import from_bounds
from rasterio.features import geometry_mask
import geopandas as gpd
D = Path(__file__).resolve().parents[2] / "digital_twin/sivasagar/data"; thr = float(sys.argv[1])
with rasterio.open(D / "dem_filled.tif") as r:
    z = r.read(1); prof = r.profile; b = r.bounds; T = r.transform
def clip(fn, dt):
    with rasterio.open(D / fn) as r:
        w = from_bounds(*b, transform=r.transform).round_offsets().round_lengths(); w = rasterio.windows.Window(w.col_off, w.row_off, z.shape[1], z.shape[0])
        assert abs(r.window_transform(w).c - T.c) < 1 and abs(r.window_transform(w).f - T.f) < 1, "grid mismatch"
        return r.read(1, window=w).astype(dt)
acc = clip("d8_accum_full.tif", "float64") * 900e-6
poly = gpd.read_file(D / "boundary_utm46n.gpkg").geometry.iloc[0]
inside = geometry_mask([poly], out_shape=z.shape, transform=T, invert=True)
old = (acc >= 10) & inside; st = (acc >= thr) & inside
with rasterio.open(D / "stream_mask.tif") as r: base = r.read(1) > 0
print("baseline stream cells", int(base.sum()), "recomputed at 10 km2", int(old.sum()), "dense", int(st.sum()), "-> network length km %.0f vs %.0f" % (st.sum() * .03, base.sum() * .03), "density km/km2 %.2f vs %.2f" % (st.sum() * .03 / (inside.sum() * 9e-4), base.sum() * .03 / (inside.sum() * 9e-4)))
zz = z.copy(); zz[st] -= 4.0
with rasterio.open(D / "dem_burn_9m.tif", "w", **prof) as o: o.write(zz.astype("float32"), 1)
with rasterio.open(D / "stream_mask_dense.tif", "w", **dict(prof, dtype="uint8", nodata=0)) as o: o.write(st.astype("uint8"), 1)
