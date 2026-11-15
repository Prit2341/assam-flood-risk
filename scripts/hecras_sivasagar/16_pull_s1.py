"""Sentinel-1 GRD (IW) VV/VH backscatter, mean-aggregated to 100 m EPSG:32646 over the model domain bbox, for the GFM scene passes and the 07-05 pre-event pass.
Used only to test whether model-flooded / GFM-dry pixels carry a flood signal in the raw SAR. Out: data/s1_<tag>_100m.tif (bands VV_dB, VH_dB, n)."""
import os, sys, urllib.request
import ee, geopandas as gpd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "digital_twin", "sivasagar", "data")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from pull_era5_gee_longrecord import PROJECT_ID
ee.Initialize(project=PROJECT_ID)
bb = gpd.read_file(os.path.join(OUT, "boundary_utm46n.gpkg")).total_bounds
region = ee.Geometry.Rectangle([float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])], "EPSG:32646", False)
WIN = {"0705": ("2026-07-05T18:00:00", "2026-07-06T06:00:00"), "0717": ("2026-07-17T18:00:00", "2026-07-18T06:00:00"),
       "0722": ("2026-07-22T06:00:00", "2026-07-22T18:00:00"), "0820": ("2026-08-20T18:00:00", "2026-08-21T06:00:00")}
for tag, (a, b) in [kv for kv in WIN.items() if kv[0] != "0820"]:
    col = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(region).filterDate(a, b).filter(ee.Filter.eq("instrumentMode", "IW")).filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    print(tag, "images:", col.size().getInfo(), col.aggregate_array("system:index").getInfo()[:6], flush=True)
    lin = col.select(["VV", "VH"]).map(lambda i: ee.Image(10).pow(i.divide(10))).mean()
    n = col.select("VV").count().rename("n")
    proj = col.first().select("VV").projection()
    img = lin.addBands(n).setDefaultProjection(proj).reduceResolution(ee.Reducer.mean(), maxPixels=4096).reproject("EPSG:32646", None, 100)
    db = img.select(["VV", "VH"]).log10().multiply(10).addBands(img.select("n")).float()
    import requests, numpy as np, rasterio
    from rasterio.merge import merge
    nx = ny = 4; xs = np.linspace(bb[0], bb[2], nx + 1); ys = np.linspace(bb[1], bb[3], ny + 1); parts = []
    for ix in range(nx):
        for iy in range(ny):
            reg = ee.Geometry.Rectangle([float(xs[ix]), float(ys[iy]), float(xs[ix + 1]), float(ys[iy + 1])], "EPSG:32646", False)
            rr = requests.get(db.getDownloadURL(dict(region=reg, scale=100, crs="EPSG:32646", format="GEO_TIFF")))
            if rr.status_code != 200: print("tile fail", ix, iy, rr.text[:200]); continue
            fn = os.path.join(OUT, f"_s1tile_{tag}_{ix}{iy}.tif"); open(fn, "wb").write(rr.content); parts.append(fn)
    srcs = [rasterio.open(f) for f in parts]; arr, tr = merge(srcs, nodata=np.nan)
    meta = srcs[0].meta.copy(); meta.update(height=arr.shape[1], width=arr.shape[2], transform=tr, count=arr.shape[0], nodata=np.nan, dtype="float32")
    with rasterio.open(os.path.join(OUT, f"s1_{tag}_100m.tif"), "w", **meta) as dst: dst.write(arr.astype("float32"))
    [s_.close() for s_ in srcs]; [os.remove(f) for f in parts]; print(tag, "merged", arr.shape, flush=True)
