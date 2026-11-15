"""Compare model maximum flooded cropland inside Sivasagar district with the DRIMS cumulative figure (28,356 ha = 283.6 km2, 19 Jul-16 Aug 2026,
Sentinel Assam 2026-08-18). Model: max water surface over the whole run minus unburned 30 m terrain, on the 100 m WorldCover grid, class 40.
  python scripts/hecras_sivasagar/14_drims_check.py results/<1056h hdf>"""
import sys
from pathlib import Path
import geopandas as gpd, h5py, numpy as np, rasterio
from scipy.spatial import cKDTree
from shapely import contains_xy
D = Path(__file__).resolve().parents[2]; DATA = D / "digital_twin/sivasagar/data"
f = h5py.File(sys.argv[1]); A = "Sivasagar_Domain"
cxy = f[f"Geometry/2D Flow Areas/{A}/Cells Center Coordinate"][:]
mx = f["Results/Unsteady/Output/Output Blocks/Base Output/Summary Output/2D Flow Areas/%s/Maximum Water Surface" % A][0]
n = min(len(mx), len(cxy)); cxy, mx = cxy[:n], mx[:n]
with rasterio.open(DATA / "worldcover_100m.tif") as r:
    wc = r.read(1); T = r.transform
rr, cc = np.mgrid[0:wc.shape[0], 0:wc.shape[1]]; x, y = T * (cc.ravel() + .5, rr.ravel() + .5)
with rasterio.open(DATA / "dem_filled.tif") as r:
    z = np.fromiter((v[0] for v in r.sample(zip(x, y), masked=False)), "float32", x.size)
dom = contains_xy(gpd.read_file(DATA / "boundary_utm46n.gpkg").geometry.iloc[0], x, y) & (z > -999)
dist, cid = cKDTree(cxy).query(np.c_[x, y]); dom &= dist < 600 * 1.2
dis = gpd.read_file(D / "data/boundaries/assam_districts.geojson"); dis = dis[dis.dtname.str.contains("Sibsagar|Sivasagar", case=False)].to_crs(32646)
print("district rows:", dis.dtname.tolist(), "district km2", round(dis.area.sum() / 1e6))
ind = contains_xy(dis.geometry.union_all(), x, y)
crop = wc.ravel() == 40; depth = mx[cid] - z
for nm, s in [("district & domain", ind & dom), ("whole domain", dom)]:
    print(nm, "area km2 %.0f, cropland km2 %.0f" % (s.sum() * .01, (s & crop).sum() * .01))
    for t in (0.1, 0.3, 0.5, 1.0):
        print("   max depth > %.1f m: cropland %.0f km2, all land %.0f km2" % (t, (s & crop & (depth > t)).sum() * .01, (s & (depth > t)).sum() * .01))
print("DRIMS: 283.6 km2 cropland submerged in the district, cumulative 19 Jul-16 Aug")
