"""
Sivasagar HEC-RAS teacher v1 test, step 7: hourly ERA5-Land precipitation as a SPATIAL grid over the model domain (cause (a) test, spec s9).
Same source/band as 02_pull_rain.py, but not averaged: one GeoTIFF band per hour (mm), WGS84, 0.1 deg native cells.
Run: python -u scripts/hecras_sivasagar/07_pull_rain_grid.py "2026-07-05 00:00" "2026-07-18 02:00"
Out: digital_twin/sivasagar/data/era5land_grid_<start>_<end>.tif  (band k = k-th hourly stamp), + .json list of stamps
"""
import json, os, sys, urllib.request
import ee, geopandas as gpd, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "digital_twin", "sivasagar", "data")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from pull_era5_gee_longrecord import PROJECT_ID
start, end = pd.Timestamp(sys.argv[1]), pd.Timestamp(sys.argv[2])
ee.Initialize(project=PROJECT_ID)
b = gpd.read_file(os.path.join(OUT, "boundary_utm46n.gpkg")).to_crs(4326).total_bounds
region = ee.Geometry.Rectangle([b[0] - 0.15, b[1] - 0.15, b[2] + 0.15, b[3] + 0.15])
col = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterDate(start.strftime("%Y-%m-%dT%H:%M:%S"), (end + pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S")).select("total_precipitation_hourly").sort("system:time_start")
stamps = [pd.to_datetime(t, unit="ms").strftime("%Y-%m-%d %H:%M") for t in col.aggregate_array("system:time_start").getInfo()]
print(len(stamps), stamps[0], stamps[-1])
img = col.map(lambda i: i.multiply(1000)).toBands()
tag = f"{start:%m%d%H}_{end:%m%d%H}"
names = img.bandNames().getInfo()
parts = []
for k in range(0, len(stamps), 120):
    sub = img.select(names[k:k + 120])
    url = sub.getDownloadURL(dict(region=region, scale=11132, crs="EPSG:4326", format="GEO_TIFF"))
    fn = os.path.join(OUT, f"_grid_part{k}.tif"); urllib.request.urlretrieve(url, fn); parts.append(fn); print("part", k, os.path.getsize(fn), flush=True)
json.dump(dict(stamps=stamps, parts=[os.path.basename(p) for p in parts]), open(os.path.join(OUT, f"era5land_grid_{tag}.json"), "w"))
