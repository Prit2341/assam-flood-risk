"""H5 under-conveyance test: burn the 4 m channel with a WIDTH that grows with upstream area, instead of one 30 m pixel.
Width rule: w = 101 m * (A / 3542 km2)^0.5, floor 30 m. Exponent 0.5 = downstream hydraulic geometry w ~ Q^0.5 (Leopold & Maddock 1953,
USGS PP 252) with Q ~ A (simplification); anchor 101 m at 3,542 km2 = GRWL median width of the lower-Dikhow segment
(Allen & Pavelsky 2018, Science 361:585; GRWL widths are at mean discharge, so bankfull is wider -> this rule is conservative).
Same stream network as the baseline (>= 10 km2 upstream). Bed of each widened pixel = (z of its nearest stream pixel) - 4 m.
  python scripts/hecras_sivasagar/35_wide_burn.py
Out: data/dem_burn_w4m.tif (label 'w4' = wide 4 m), data/channel_width_m.tif
"""
from pathlib import Path
import numpy as np, rasterio, geopandas as gpd
from rasterio.windows import from_bounds
from rasterio.features import geometry_mask
from scipy.ndimage import distance_transform_edt

D = Path(__file__).resolve().parents[2] / "digital_twin/sivasagar/data"
with rasterio.open(D / "dem_filled.tif") as r:
    z = r.read(1).astype("float64"); prof = r.profile; b = r.bounds; T = r.transform
with rasterio.open(D / "d8_accum_full.tif") as r:
    w = from_bounds(*b, transform=r.transform).round_offsets().round_lengths()
    w = rasterio.windows.Window(w.col_off, w.row_off, z.shape[1], z.shape[0])
    assert abs(r.window_transform(w).c - T.c) < 1 and abs(r.window_transform(w).f - T.f) < 1, "grid mismatch"
    acc = r.read(1, window=w).astype("float64") * 900e-6
inside = geometry_mask([gpd.read_file(D / "boundary_utm46n.gpkg").geometry.iloc[0]], out_shape=z.shape, transform=T, invert=True)
st = (acc >= 10) & inside
width = np.where(st, np.maximum(30.0, 101.0 * (acc / 3542.0) ** 0.5), 0.0)
dist, (ri, ci) = distance_transform_edt(~st, return_indices=True)
near_w = width[ri, ci]; near_z = z[ri, ci]
burn = inside & (dist * 30.0 <= np.maximum(0.0, (near_w - 30.0) / 2.0) + 1e-6)     # half-width beyond the centre pixel
zz = z.copy()
zz[burn] = np.minimum(z[burn], near_z[burn] - 4.0)
print(f"stream px {st.sum()}, burned px {burn.sum()} ({burn.sum() / st.sum():.2f}x); width at stream px p50/p90/max "
      f"{np.percentile(width[st], [50, 90]).round(0)} {width[st].max():.0f} m; outlet-area width {101 * (acc[st].max() / 3542) ** .5:.0f} m")
with rasterio.open(D / "dem_burn_w4m.tif", "w", **dict(prof, dtype="float32")) as o:
    o.write(zz.astype("float32"), 1)
with rasterio.open(D / "channel_width_m.tif", "w", **dict(prof, dtype="float32", nodata=0)) as o:
    o.write(np.where(burn, near_w, 0).astype("float32"), 1)
# compare with the baseline 4 m burn
with rasterio.open(D / "dem_burn_4m.tif") as r:
    z4 = r.read(1)
print("baseline burned px", int(((z - z4) > 1).sum()), " max |diff| at stream px (should be 0):", float(np.abs(zz[st] - z4[st]).max()))
