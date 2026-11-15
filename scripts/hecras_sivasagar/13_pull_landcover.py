"""ESA WorldCover v200 (class codes) and JRC GSW occurrence/seasonality at 100 m, EPSG:32646, over the model domain bbox. Used only for the false-alarm breakdown."""
import os, sys, urllib.request
import ee, geopandas as gpd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "digital_twin", "sivasagar", "data")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from pull_era5_gee_longrecord import PROJECT_ID
ee.Initialize(project=PROJECT_ID)
bb = gpd.read_file(os.path.join(OUT, "boundary_utm46n.gpkg")).total_bounds
region = ee.Geometry.Rectangle([float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])], "EPSG:32646", False)
wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
for name, img in [("worldcover", wc), ("gsw_occurrence", gsw.select("occurrence").unmask(0)), ("gsw_seasonality", gsw.select("seasonality").unmask(0))]:
    url = img.getDownloadURL(dict(region=region, scale=100, crs="EPSG:32646", format="GEO_TIFF"))
    fn = os.path.join(OUT, f"{name}_100m.tif"); urllib.request.urlretrieve(url, fn); print(name, os.path.getsize(fn), flush=True)
