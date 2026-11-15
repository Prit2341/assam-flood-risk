"""
Build docs/site/basins.html: a self-contained map (inline SVG, no external libraries) of the HydroBASINS levels 6/7/8
basins that touch the 4 scope districts, the district outlines and the 10 CWC gauges and (if present) the basins traced from terrain.
Inputs: data/basin_delineation/scope4_basins_taken_hydrobasins.gpkg, data/boundaries/scope4_districts.geojson,
        data/basin_delineation/cwc_stations_vs_hydrobasins.csv
Run:    python scripts/190_basin_map.py     then serve docs/site (launch.json 'assam-progress', port 8090)
"""
import json
import math
import os

import geopandas as gpd
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPKG = os.path.join(ROOT, "data", "basin_delineation", "scope4_basins_taken_hydrobasins.gpkg")
DIST = os.path.join(ROOT, "data", "boundaries", "scope4_districts.geojson")
CWC = os.path.join(ROOT, "data", "basin_delineation", "cwc_stations_vs_hydrobasins.csv")
OUT = os.path.join(ROOT, "docs", "site", "basins.html")
LON0, LON1, LAT0, LAT1 = 90.9, 96.7, 23.6, 28.1
K = 170.0                                           # px per degree of latitude
COS = math.cos(math.radians((LAT0 + LAT1) / 2))
W, H = (LON1 - LON0) * COS * K, (LAT1 - LAT0) * K
TOL = 0.003                                         # ~300 m simplification, for display only


def xy(lon, lat):
    return round((lon - LON0) * COS * K, 1), round((LAT1 - lat) * K, 1)


def path(geom):
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    d = []
    for p in polys:
        for ring in [p.exterior, *p.interiors]:
            pts = [xy(*c[:2]) for c in ring.coords]
            d.append("M" + "L".join(f"{x},{y}" for x, y in pts) + "Z")
    return "".join(d)


levels = {}
for lev in (6, 7, 8):
    g = gpd.read_file(GPKG, layer=f"lev{lev:02d}")
    g["geometry"] = g.geometry.simplify(TOL, preserve_topology=True)
    rows = []
    for _, r in g.iterrows():
        rows.append(dict(id=str(int(r.HYBAS_ID)), sub=int(r.SUB_AREA), up=int(r.UP_AREA), next=str(int(r.NEXT_DOWN)),
                         dist={k.replace("km2_in_", ""): int(r[k]) for k in g.columns if k.startswith("km2_in_") and r[k] >= 1},
                         d=path(r.geometry)))
    levels[lev] = rows

D = gpd.read_file(DIST).replace({"Marigaon": "Morigaon"})
D["geometry"] = D.geometry.simplify(0.001)
districts = [dict(name=r.dtname, d=path(r.geometry)) for _, r in D.iterrows()]
S = pd.read_csv(CWC)
stations = [dict(name=r["name"], river=r["river_name"], basin=r["basin_name"], district=r["district_n"].title(),
                 x=xy(r["lon"], r["lat"])[0], y=xy(r["lon"], r["lat"])[1]) for _, r in S.iterrows()]

TRACED = os.path.join(ROOT, "data", "basin_delineation", "sivasagar_fabdem", "basins_traced.gpkg")      # primary: FABDEM (Sri Lanka choice)
TRACED_RAW = os.path.join(ROOT, "data", "basin_delineation", "sivasagar", "basins_traced.gpkg")         # sensitivity: raw Copernicus GLO-30
traced = []
for area in ("sivasagar", "nagaon_morigaon", "karimganj"):          # one FABDEM trace per gauge, all 10 CWC gauges of the 4 districts
    f = TRACED if area == "sivasagar" else os.path.join(ROOT, "data", "basin_delineation", area + "_fabdem", "basins_traced.gpkg")
    if os.path.exists(f):
        T = gpd.read_file(f, layer="basins_smoothed")   # Sri Lanka step D: buffer +100/-100, simplify 80 m, single polygon
        T["geometry"] = T.geometry.simplify(TOL, preserve_topology=True)
        traced += [dict(gauge=r.gauge, river=r.river, km2=float(r.smoothed_km2), d=path(r.geometry)) for _, r in T.iterrows()]

traced_raw = []
if os.path.exists(TRACED_RAW):
    T = gpd.read_file(TRACED_RAW, layer="basins_smoothed")
    T["geometry"] = T.geometry.simplify(TOL, preserve_topology=True)
    traced_raw = [dict(gauge=r.gauge, river=r.river, km2=float(r.smoothed_km2), d=path(r.geometry)) for _, r in T.iterrows()]

data = json.dumps(dict(w=round(W), h=round(H), levels=levels, districts=districts, stations=stations, traced=traced, traced_raw=traced_raw), separators=(",", ":"))
html = open(os.path.join(ROOT, "scripts", "190_basin_map_template.html"), encoding="utf-8").read().replace("/*DATA*/null", data)
open(OUT, "w", encoding="utf-8").write(html)
print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB), basins per level:", {k: len(v) for k, v in levels.items()})
