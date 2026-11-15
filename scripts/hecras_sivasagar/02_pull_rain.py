"""
Sivasagar HEC-RAS teacher, step 2: hourly ERA5-Land precipitation averaged over the model DOMAIN (decision D3).
The existing data/era5_longrecord/Sivasagar series is a district-polygon mean of ERA5 (31 km) and does not represent the
basins, which lie mostly in the Nagaland/Patkai hills.

ERA5-Land hourly (ECMWF/ERA5_LAND/HOURLY, band total_precipitation_hourly, metres per hour), 0.1 deg (~11 km) native,
mean over digital_twin/sivasagar/data/boundary_utm46n.gpkg. Time stamp = end of the accumulation hour as stored by GEE
(value at T covers T-1h..T) -- checked against the HEC-RAS interval-ending convention in the plan builder, not here.

Run:  python -u scripts/hecras_sivasagar/02_pull_rain.py 2026-07-01 2026-08-26
Out:  digital_twin/sivasagar/data/era5land_domain_hourly_<start>_<end>.csv  (timestamp, tp_mm)
"""
import json
import os
import sys

import ee
import geopandas as gpd
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "digital_twin", "sivasagar", "data")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from pull_era5_gee_longrecord import PROJECT_ID  # noqa: E402

start, end = sys.argv[1], sys.argv[2]
ee.Initialize(project=PROJECT_ID)
poly = gpd.read_file(os.path.join(OUT, "boundary_utm46n.gpkg")).to_crs(4326).geometry.iloc[0].simplify(0.002)
region = ee.Geometry(json.loads(gpd.GeoSeries([poly]).to_json())["features"][0]["geometry"])
col = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterDate(start, end).select("total_precipitation_hourly")


def red(img):
    s = img.reduceRegion(ee.Reducer.mean(), region, 11132, maxPixels=1e9)
    return ee.Feature(None, s).set("t", img.get("system:time_start"))


rows = []
days = pd.date_range(start, end, freq="5D")
edges = list(days) + [pd.Timestamp(end)]
for a, b in zip(edges[:-1], edges[1:]):
    if a >= b:
        continue
    fc = col.filterDate(a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")).map(red).getInfo()["features"]
    rows += [dict(timestamp=pd.to_datetime(f["properties"]["t"], unit="ms"), tp_mm=f["properties"].get("total_precipitation_hourly", float("nan")) * 1000) for f in fc]
    print(a.date(), len(fc), flush=True)
df = pd.DataFrame(rows).sort_values("timestamp")
fn = os.path.join(OUT, f"era5land_domain_hourly_{start}_{end}.csv")
df.to_csv(fn, index=False)
print(len(df), "rows", df.timestamp.min(), df.timestamp.max(), "total mm", round(df.tp_mm.sum(), 1), "max mm/h", round(df.tp_mm.max(), 2), "NaN", int(df.tp_mm.isna().sum()))
