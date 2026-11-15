"""
H2b test: clone the gridded-rain project (hecras_c600_d4_grid; u01 already has Met BC Gridded + ET Mode=None + CRLF, decision log
Chroma 'HEC-RAS 7.0.1 gridded rain fix') and swap in a different hourly rain grid, then import it into u01.hdf the same way as before
(RasUnsteady._update_precipitation_hdf). Teacher otherwise frozen (600 m, 4 m burn, n 0.04, ratio 0.3, cold start, west exit).

  python -u scripts/hecras_sivasagar/33_grid_rain_project.py imerg  digital_twin/sivasagar/data/imerg/imerg_grid_070500_072300.json
  python -u scripts/hecras_sivasagar/33_grid_rain_project.py era5full digital_twin/sivasagar/data/era5land_grid_070500_072300.json
Out: digital_twin/sivasagar/hecras_c600_d4_grid_<name>/ with rain_grid.nc (APCP_surface, mm x 0.3, EPSG:32646 2 km, same grid
recipe as 08_make_rain_netcdf.py).
"""
import json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd, rasterio, xarray as xr, geopandas as gpd, pyproj
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
from ras_commander.RasUnsteady import RasUnsteady

ROOT = Path(__file__).resolve().parents[2]; D = ROOT / "digital_twin" / "sivasagar"
name, js = sys.argv[1], ROOT / sys.argv[2]; RATIO = 0.3
src_prj, dst = D / "hecras_c600_d4_grid", D / f"hecras_c600_d4_grid_{name}"
if dst.exists(): shutil.rmtree(dst)
shutil.copytree(src_prj, dst, ignore=shutil.ignore_patterns("*.p01.hdf", "rain_grid.nc", "_compute_p01.log", "*.bco01", "*.ic.o01"))

meta = json.load(open(js)); bands = []
for p in meta["parts"]:
    with rasterio.open(js.parent / p) as r:
        bands.append(r.read()); s = (r.transform, r.crs)
A = np.nan_to_num(np.concatenate(bands).astype("float32"))
assert A.shape[0] == len(meta["stamps"]), (A.shape, len(meta["stamps"]))
poly = gpd.read_file(D / "data" / "boundary_utm46n.gpkg").geometry.iloc[0]
RES = 2000; bx0, by0, bx1, by1 = poly.bounds
x0, y0 = np.floor((bx0 - 15000) / RES) * RES, np.ceil((by1 + 15000) / RES) * RES
W = int((bx1 + 15000 - x0) / RES) + 1; H = int((y0 - (by0 - 15000)) / RES) + 1
tr = from_origin(x0, y0, RES, RES); out = np.zeros((A.shape[0], H, W), "float32")
for k in range(A.shape[0]):
    reproject(A[k], out[k], src_transform=s[0], src_crs=s[1], dst_transform=tr, dst_crs="EPSG:32646", resampling=Resampling.nearest)
out = np.nan_to_num(out) * RATIO
xs = x0 + RES * (np.arange(W) + .5); ys = y0 - RES * (np.arange(H) + .5)
ds = xr.Dataset({"APCP_surface": (("time", "y", "x"), out, {"units": "mm", "grid_mapping": "spatial_ref"})},
                coords={"time": pd.to_datetime(meta["stamps"]), "y": ("y", ys, {"axis": "Y", "units": "m", "standard_name": "projection_y_coordinate"}),
                        "x": ("x", xs, {"axis": "X", "units": "m", "standard_name": "projection_x_coordinate"})})
wkt = pyproj.CRS(32646).to_wkt()
ds["spatial_ref"] = xr.DataArray(0, attrs={"crs_wkt": wkt, "spatial_ref": wkt, "grid_mapping_name": "transverse_mercator"})
nc = dst / "rain_grid.nc"; ds.to_netcdf(nc)
m = geometry_mask([poly], out_shape=(H, W), transform=tr, invert=True)
print(f"{name}: {len(meta['stamps'])} h {meta['stamps'][0]}..{meta['stamps'][-1]}, domain-mean total x{RATIO} = {out[:, m].mean(1).sum():.1f} mm; "
      f"cell totals min/med/max {np.percentile(out[:, m].sum(0), [0, 50, 100]).round(0)}")
RasUnsteady._update_precipitation_hdf(dst / "Sivasagar_2D.u01.hdf", nc, r".\rain_grid.nc", "Bilinear", "APCP_surface")
import h5py
with h5py.File(dst / "Sivasagar_2D.u01.hdf", "r") as f:
    g = f["Event Conditions/Meteorology/Precipitation"]
    for k, v in g.items():
        print(" u01.hdf", k, getattr(v, "shape", v))
print("project ready:", dst)
