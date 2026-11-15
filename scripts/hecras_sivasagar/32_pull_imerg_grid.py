"""
H2b: GPM IMERG V07 (GEE NASA/GPM_L3/IMERG_V07, half-hourly 'precipitation' mm/hr, 0.1 deg) over the model domain, as an
independent rain product to compare with ERA5-Land (does ERA5-Land miss the 19 July 2026 hill rain reported by the
Assam University situation report, Chroma manual-02xx 'NWDP...').
Hourly depth (mm) = mean of the two half-hour rates in each hour, stamped at the END of the hour (ERA5-Land convention).
Run:  python -u scripts/hecras_sivasagar/32_pull_imerg_grid.py "2026-07-05 00:00" "2026-07-23 00:00"
Out:  digital_twin/sivasagar/data/imerg/_imerg_part<k>.tif + imerg_grid_<tag>.json  (separate folder, no clash with ERA5 parts)
"""
import json, os, sys, urllib.request
import ee, geopandas as gpd, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "digital_twin", "sivasagar", "data")
OUT = os.path.join(DATA, "imerg"); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from pull_era5_gee_longrecord import PROJECT_ID
start, end = pd.Timestamp(sys.argv[1]), pd.Timestamp(sys.argv[2])
ee.Initialize(project=PROJECT_ID)
b = gpd.read_file(os.path.join(DATA, "boundary_utm46n.gpkg")).to_crs(4326).total_bounds
region = ee.Geometry.Rectangle([b[0] - 0.15, b[1] - 0.15, b[2] + 0.15, b[3] + 0.15])
col = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").select("precipitation")
first = col.filterDate(start.strftime("%Y-%m-%d"), (start + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
print("IMERG V07 images on first day:", first.size().getInfo(), "latest image:",
      ee.Date(col.sort("system:time_start", False).first().get("system:time_start")).format().getInfo(), flush=True)
hours = pd.date_range(start + pd.Timedelta(hours=1), end, freq="h")       # end-of-hour stamps
def hourly(ms):
    t1 = ee.Date(ms); t0 = t1.advance(-1, "hour")
    return col.filterDate(t0, t1).mean().rename("p").set("system:time_start", t1.millis())
imgs = ee.ImageCollection(ee.List([int(h.value // 10**6) for h in hours]).map(lambda ms: hourly(ms)))
img = imgs.toBands()
names = img.bandNames().getInfo()
parts = []
for k in range(0, len(hours), 120):
    url = img.select(names[k:k + 120]).getDownloadURL(dict(region=region, scale=11132, crs="EPSG:4326", format="GEO_TIFF"))
    fn = os.path.join(OUT, f"_imerg_part{k}.tif"); urllib.request.urlretrieve(url, fn); parts.append(os.path.basename(fn))
    print("part", k, os.path.getsize(fn), flush=True)
tag = f"{start:%m%d%H}_{end:%m%d%H}"
json.dump(dict(stamps=[h.strftime("%Y-%m-%d %H:%M") for h in hours], parts=parts), open(os.path.join(OUT, f"imerg_grid_{tag}.json"), "w"))
print("done", len(hours), "hours")
