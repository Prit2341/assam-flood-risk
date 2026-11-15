"""
FABDEM (V1-2, GEE community catalog projects/sat-io/open-datasets/FABDEM) for the area upstream of Sivasagar
(Dikhow + Desang catchments traced in scripts/191), so the basins can be re-traced on FABDEM like Sri Lanka did.
Box = bounds of data/basin_delineation/sivasagar/basins_traced.gpkg (layer basins_smoothed) + 0.02 deg, snapped to the
1 arc-second grid, pulled in NX x NY pieces (GEE limits one direct download to ~50 MB) that share one global grid.

Run:  python scripts/192_pull_fabdem_upstream.py
Out:  data/dem_fabdem_upstream_sivasagar/fabdem_piece_<i>_<j>.tif  (EPSG:4326, ~30 m, float32)
"""
import math
import os

import ee
import geopandas as gpd
import rasterio
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ID = "assamflood-508209"
AREA = os.environ.get("AREA", "sivasagar")          # sivasagar | nagaon_morigaon | karimganj
BOXES = {"nagaon_morigaon": (91.65, 25.05, 93.75, 26.65, 3, 2),     # Kopili upstream of Jagibhakatgaon (HydroBASINS L9 upstream bounds + margin)
         "karimganj": (91.95, 23.75, 92.75, 25.00, 1, 2)}          # Longai / Singla upstream of the 4 Karimganj gauges
OUT = os.path.join(ROOT, "data", "dem_fabdem_upstream_" + AREA)
NX, NY = 2, 2
RES = 1 / 3600
MARGIN = 0.02


def main():
    global NX, NY
    if AREA in BOXES:
        x0, y0, x1, y1, NX, NY = BOXES[AREA]
    else:
        T = gpd.read_file(os.path.join(ROOT, "data", "basin_delineation", "sivasagar", "basins_traced.gpkg"), layer="basins_smoothed")
        x0, y0, x1, y1 = T.total_bounds
    x0 = math.floor((x0 - MARGIN) * 3600) / 3600; y0 = math.floor((y0 - MARGIN) * 3600) / 3600
    x1 = math.ceil((x1 + MARGIN) * 3600) / 3600; y1 = math.ceil((y1 + MARGIN) * 3600) / 3600
    nx, ny = round((x1 - x0) / RES), round((y1 - y0) / RES)
    print(f"box lon {x0:.4f}-{x1:.4f} lat {y0:.4f}-{y1:.4f}: {nx} x {ny} cells", flush=True)
    os.makedirs(OUT, exist_ok=True)
    ee.Initialize(project=PROJECT_ID)
    dem = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM").filterBounds(ee.Geometry.Rectangle([x0, y0, x1, y1])).mosaic().toFloat()
    cols = [round(nx * k / NX) for k in range(NX + 1)]
    rows = [round(ny * k / NY) for k in range(NY + 1)]
    total = 0
    for j in range(NY):
        for i in range(NX):
            px0, px1, py0, py1 = cols[i], cols[i + 1], rows[j], rows[j + 1]
            west, east = x0 + px0 * RES, x0 + px1 * RES
            north, south = y1 - py0 * RES, y1 - py1 * RES
            region = ee.Geometry.Rectangle([west, south, east, north], proj="EPSG:4326", geodesic=False)
            url = dem.getDownloadURL({"region": region, "crs": "EPSG:4326", "crsTransform": [RES, 0, west, 0, -RES, north],
                                      "dimensions": f"{px1 - px0}x{py1 - py0}", "format": "GEO_TIFF"})
            path = os.path.join(OUT, f"fabdem_piece_{i}_{j}.tif")
            r = requests.get(url, timeout=600)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
            with rasterio.open(path) as s:                      # catch truncated or error-page downloads
                a = s.read(1)
                print(f"  piece {i}_{j}: {s.shape} bounds {[round(b, 4) for b in s.bounds]} min {a.min():.1f} max {a.max():.1f} "
                      f"zeros {(a == 0).mean():.3%}  {len(r.content) / 1e6:.1f} MB", flush=True)
            total += len(r.content)
    print(f"done, {total / 1e6:.1f} MB in {OUT}")


if __name__ == "__main__":
    main()
