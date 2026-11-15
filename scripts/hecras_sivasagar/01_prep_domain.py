"""
Sivasagar HEC-RAS teacher, step 1: hydrological domain, hydro-conditioned terrain, channel variants.
Decisions D1, D2, D4 in docs/sivasagar_teacher_spec.md (D1 revised 2026-10-31, see that doc).

Domain = D8 watershed of ONE exit cell on the Dikhow-Desang main stem, just inside the west edge of the FABDEM mosaic
(traced downstream from the gauges: Bihubar, Sivasagar and Nanglamoraghat all reach the same exit cell). A watershed has a single
outlet, so no stray exits along an administrative perimeter.
Terrain: FABDEM 30 m (data/basin_delineation/sivasagar_fabdem/dem_utm46.tif), depressions filled with flat resolution (Whitebox).
Channel variants: DEPTHS metres lowered along cells with at least STREAM_KM2 upstream (one 30 m pixel wide).
Outlet normal-depth slope: linear fit of filled elevation vs path distance over the 10 km upstream of the exit.

Out: digital_twin/sivasagar/data/{boundary_utm46n.gpkg, dem_filled.tif, dem_burn_<d>m.tif, stream_mask.tif, outlet.json}
"""
import json
import os

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask, shapes
from rasterio.windows import Window
from scipy import ndimage
from shapely.geometry import Point, Polygon, shape
from whitebox import WhiteboxTools

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "digital_twin", "sivasagar", "data")
BD = os.path.join(ROOT, "data", "basin_delineation", "sivasagar_fabdem")
CRS = 32646
DEPTHS = [2, 4, 8]
STREAM_KM2 = 10.0
INSET_M = 300            # inset from the terrain nodata edge
EXIT_COL = 40            # exit cell: first path cell at least this many columns (1.2 km) inside the mosaic edge
D8 = {1: (-1, 1), 2: (0, 1), 4: (1, 1), 8: (1, 0), 16: (1, -1), 32: (0, -1), 64: (-1, -1), 128: (-1, 0)}  # Whitebox pointer codes
os.makedirs(OUT, exist_ok=True)
P = lambda n: os.path.join(OUT, n)


def trace(ptr, z, r0, c0):
    path, seen = [(r0, c0)], {(r0, c0)}
    while True:
        r, c = path[-1]
        p = int(ptr[r, c])
        if p not in D8:
            break
        n = (r + D8[p][0], c + D8[p][1])
        if not (0 <= n[0] < ptr.shape[0] and 0 <= n[1] < ptr.shape[1]) or n in seen or z[n] < -1000:
            break
        path.append(n)
        seen.add(n)
    return path


def main():
    wbt = WhiteboxTools()
    wbt.set_working_dir(OUT)
    wbt.verbose = False
    with rasterio.open(os.path.join(BD, "dem_utm46.tif")) as r:
        prof = r.profile
        a = r.read(1)
    with rasterio.open(P("dem_mosaic.tif"), "w", **prof) as o:
        o.write(a, 1)
    wbt.fill_depressions("dem_mosaic.tif", "dem_filled_full.tif", fix_flats=True)
    wbt.d8_pointer("dem_filled_full.tif", "d8_pointer_full.tif")
    wbt.d8_flow_accumulation("dem_filled_full.tif", "d8_accum_full.tif", out_type="cells")
    with rasterio.open(P("dem_filled_full.tif")) as r:
        z = r.read(1)
        tr = r.transform
        fprof = r.profile
    ptr = rasterio.open(P("d8_pointer_full.tif")).read(1)
    acc = rasterio.open(P("d8_accum_full.tif")).read(1) * 900e-6   # km2

    g = gpd.read_file(os.path.join(BD, "gauges_snapped.shp")).to_crs(CRS)
    q = g[g["name"] == "Sivasagar"].iloc[0]
    row, col = rasterio.open(P("dem_filled_full.tif")).index(q.geometry.x, q.geometry.y)
    w = acc[row - 3:row + 4, col - 3:col + 4]
    k = np.unravel_index(np.argmax(w), w.shape)
    path = trace(ptr, z, row - 3 + k[0], col - 3 + k[1])
    ex = next(p for p in path if p[1] <= EXIT_COL)   # the path runs west, so columns fall downstream
    ox, oy = tr * (ex[1] + 0.5, ex[0] + 0.5)
    print("path cells", len(path), "exit cell", ex, "UTM", round(ox), round(oy), "upstream km2", round(float(acc[ex])),
          "z", round(float(z[ex]), 2))

    pts = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(ox, oy)], crs=CRS)
    pts.to_file(P("exit.shp"))
    wbt.watershed("d8_pointer_full.tif", "exit.shp", "ws.tif")
    with rasterio.open(P("ws.tif")) as r:
        ws = r.read(1) > 0
    geoms = [shape(s) for s, v in shapes(ws.astype("uint8"), mask=ws, transform=tr)]
    poly = max(geoms, key=lambda p: p.area).buffer(45).buffer(-45)
    print("watershed km2", round(poly.area / 1e6, 1), "cells", int(ws.sum()), "area from cells km2", round(ws.sum() * 900e-6, 1),
          "holes", len(poly.interiors))
    # HEC-RAS needs terrain under every perimeter face. The mosaic has a rotated nodata edge, so inset the polygon from it.
    # Where the watershed touches the data edge its catchment is TRUNCATED (inflow from outside the terrain is lost): reported.
    valid = z > -1000
    inset = ndimage.binary_erosion(valid, iterations=int(INSET_M / 30))
    vgeoms = [shape(s) for s, v in shapes(inset.astype("uint8"), mask=inset, transform=tr)]
    vpoly = max(vgeoms, key=lambda p: p.area)
    full_km2 = poly.area / 1e6
    poly = poly.intersection(vpoly)
    if poly.geom_type != "Polygon":
        poly = max(poly.geoms, key=lambda p: p.area)
    poly = Polygon(poly.exterior)
    print("after inset", INSET_M, "m from the terrain edge:", round(poly.area / 1e6, 1), "km2; cut off", round(full_km2 - poly.area / 1e6, 1),
          "km2 (", round(100 * (1 - poly.area / 1e6 / full_km2), 2), "% of the watershed, catchment truncated there)")
    gpd.GeoDataFrame({"name": ["domain"]}, geometry=[poly], crs=CRS).to_file(P("boundary_utm46n.gpkg"), driver="GPKG")

    # clip rasters to the watershed bounds + 1 km
    minx, miny, maxx, maxy = poly.bounds
    c0, r1 = ~tr * (minx - 1000, miny - 1000)
    c1, r0 = ~tr * (maxx + 1000, maxy + 1000)
    c0, r0 = max(0, int(c0)), max(0, int(r0))
    c1, r1 = min(z.shape[1], int(c1) + 1), min(z.shape[0], int(r1) + 1)
    win = (slice(r0, r1), slice(c0, c1))
    ctr = tr * rasterio.Affine.translation(c0, r0)
    fprof.update(width=c1 - c0, height=r1 - r0, transform=ctr, dtype="float32", nodata=-9999)
    zc, ac, wsc = z[win].astype("float32"), acc[win], ws[win]
    with rasterio.open(P("dem_filled.tif"), "w", **fprof) as o:
        o.write(zc, 1)
    wsc = wsc & geometry_mask([poly], out_shape=wsc.shape, transform=ctr, invert=True)
    stream = (ac >= STREAM_KM2) & wsc
    print("clip", zc.shape, "stream cells", int(stream.sum()))
    for d in DEPTHS:
        zz = zc.copy()
        zz[stream] -= d
        with rasterio.open(P(f"dem_burn_{d}m.tif"), "w", **fprof) as o:
            o.write(zz, 1)
    with rasterio.open(P("stream_mask.tif"), "w", **dict(fprof, dtype="uint8", nodata=0)) as o:
        o.write(stream.astype("uint8"), 1)

    # normal-depth slope: elevation vs distance along the path over the 10 km upstream of the exit
    i_ex = path.index(ex)
    xs = np.array([tr * (path[i][1] + .5, path[i][0] + .5) for i in range(i_ex + 1)])
    dist = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xs, axis=0).T))])
    d_to_exit = dist[i_ex] - dist[:i_ex + 1]
    sel = d_to_exit <= 10000
    zz = np.array([z[p] for p in path[:i_ex + 1]])
    slope = float(abs(np.polyfit(d_to_exit[sel], zz[sel], 1)[0]))
    print("slope over 10 km upstream of exit", slope, "z drop", float(zz[sel].max() - zz[sel].min()), "cells", int(sel.sum()))
    json.dump(dict(outlet_x=float(ox), outlet_y=float(oy), upstream_km2=float(acc[ex]), friction_slope=slope,
                   domain_km2=poly.area / 1e6, stream_km2_threshold=STREAM_KM2, exit_rc=[int(ex[0]), int(ex[1])]),
              open(P("outlet.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
