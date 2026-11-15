"""
Sri Lanka step D boundary smoothing for the traced basins (buffer +100 m, -100 m, simplify 80 m, keep the largest polygon,
fill holes) on the raw D8 traces from scripts/191, for the areas that were not smoothed when 191 ran (Sivasagar was smoothed earlier).
Run:  python scripts/194_smooth_basins.py nagaon_morigaon karimganj
Out:  layer basins_smoothed in data/basin_delineation/<area>_fabdem/basins_traced.gpkg, summary_smoothed.csv
"""
import os
import sys

import geopandas as gpd
from shapely.geometry import Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRS = "EPSG:32646"
for area in sys.argv[1:]:
    f = os.path.join(ROOT, "data", "basin_delineation", area + "_fabdem", "basins_traced.gpkg")
    g = gpd.read_file(f, layer="basins").to_crs(CRS)
    out = []
    for _, r in g.iterrows():
        geom = r.geometry.buffer(100).buffer(-100).simplify(80)
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda p: p.area)
        out.append(dict(gauge=r.gauge, river=r.river, raw_km2=r.basin_km2, smoothed_km2=round(Polygon(geom.exterior).area / 1e6, 1), geometry=Polygon(geom.exterior)))
    s = gpd.GeoDataFrame(out, crs=CRS).to_crs(4326)
    s.to_file(f, layer="basins_smoothed", driver="GPKG")
    s.drop(columns="geometry").to_csv(os.path.join(os.path.dirname(f), "summary_smoothed.csv"), index=False)
    print(area); print(s.drop(columns="geometry").to_string(index=False))
