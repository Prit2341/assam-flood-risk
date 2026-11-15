"""
Step 8: ERA5-Land hourly grid (07_pull_rain_grid.py) -> HEC-RAS GDAL-raster NetCDF in EPSG:32646, hourly depth in mm times RATIO, variable APCP_surface.
Run: python -u scripts/hecras_sivasagar/08_make_rain_netcdf.py 0705_0718 0.3
Out: digital_twin/sivasagar/data/rain_grid_<tag>_r<ratio%>.nc ; prints the domain-mean check against the domain-mean CSV used by the uniform runs.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, rasterio, xarray as xr, geopandas as gpd
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
ROOT = Path(__file__).resolve().parents[2]; D = ROOT / "digital_twin" / "sivasagar" / "data"
tag, ratio = sys.argv[1], float(sys.argv[2])
meta = json.load(open(next(D.glob(f"era5land_grid_*{tag.split('_')[0]}*_*.json"))))
bands, src = [], None
for p in meta["parts"]:
    with rasterio.open(D / p) as r:
        bands.append(r.read()); src = (r.transform, r.crs, r.height, r.width)
A = np.concatenate(bands).astype("float32"); print("grid", A.shape, "stamps", len(meta["stamps"]))
b = gpd.read_file(D / "boundary_utm46n.gpkg"); poly = b.geometry.iloc[0]
x0, y0, x1, y1 = poly.bounds; RES = 2000
x0, y0 = np.floor((x0 - 15000) / RES) * RES, np.ceil((y1 + 15000) / RES) * RES; W = int((x1 + 15000 - x0) / RES) + 1; H = int((y0 - (y0 - (y1 - b.total_bounds[1] + 30000))) / RES) + 1
H = int((y0 - (b.total_bounds[1] - 15000)) / RES) + 1
tr = from_origin(x0, y0, RES, RES)
out = np.zeros((A.shape[0], H, W), "float32")
for k in range(A.shape[0]):
    reproject(A[k], out[k], src_transform=src[0], src_crs=src[1], dst_transform=tr, dst_crs="EPSG:32646", resampling=Resampling.nearest)
out *= ratio
out = np.nan_to_num(out)
xs = x0 + RES * (np.arange(W) + .5); ys = y0 - RES * (np.arange(H) + .5)
ds = xr.Dataset({"APCP_surface": (("time", "y", "x"), out, {"units": "mm", "grid_mapping": "spatial_ref"})},
                coords={"time": pd.to_datetime(meta["stamps"]), "y": ("y", ys, {"axis": "Y", "units": "m", "standard_name": "projection_y_coordinate"}),
                        "x": ("x", xs, {"axis": "X", "units": "m", "standard_name": "projection_x_coordinate"})})
import pyproj
ds["spatial_ref"] = xr.DataArray(0, attrs={"crs_wkt": pyproj.CRS(32646).to_wkt(), "spatial_ref": pyproj.CRS(32646).to_wkt(), "grid_mapping_name": "transverse_mercator"})
fn = D / f"rain_grid_{tag}_r{int(ratio * 100)}.nc"; ds.to_netcdf(fn); print("wrote", fn, fn.stat().st_size)
m = geometry_mask([poly], out_shape=(H, W), transform=tr, invert=True)
gm = out[:, m].mean(axis=1); csv = pd.read_csv(D / "era5land_domain_hourly_2026-07-01_2026-08-26.csv", parse_dates=["timestamp"]).set_index("timestamp").tp_mm * ratio
c = csv.reindex(pd.to_datetime(meta["stamps"])).values
print("domain-mean total mm grid", gm.sum().round(1), "csv", np.nansum(c).round(1), "corr", np.corrcoef(gm, c)[0, 1].round(3))
print("spatial spread of 13-day total (mm, within domain): min/median/max", np.percentile(out[:, m].sum(0), [0, 50, 100]).round(0))
