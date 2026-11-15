"""
Pour-point basin delineation for Sivasagar with WhiteboxTools (the Arunachal Pradesh method: fill_depressions ->
d8 pointer / accumulation -> basin above a gauge-derived outlet), on Copernicus GLO-30 tiles that cover the
upstream area (Nagaland / Patkai hills), which data/boundaries/assam_dem_clip.tif does not (nodata outside Assam).

Outlets: the 3 CWC SMART-AXOM gauges in Sivasagar (Sivasagar and Bihubar on the Dikhow, Nanglamoraghat on the Desang).
Each gauge is snapped to the highest-accumulation cell within SNAP_M; the basin is everything draining to that cell.
Checks per basin: area, whether it touches the DEM edge (truncated catchment), area vs HydroRIVERS / HydroBASINS.

Run:  python -u scripts/191_sivasagar_basins.py            (raw Copernicus)
      DEM=fabdem python -u scripts/191_sivasagar_basins.py     (FABDEM from scripts/192)
Out:  data/basin_delineation/sivasagar/{basins_traced.gpkg, summary.csv} + WhiteboxTools rasters in the same folder
"""
import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import Point, shape
from whitebox import WhiteboxTools

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEM_SRC = os.environ.get("DEM", "copernicus")      # copernicus (raw GLO-30 tiles) | fabdem (scripts/192 pieces, forest/building bias removed)
AREA = os.environ.get("AREA", "sivasagar")          # sivasagar | nagaon_morigaon | karimganj (FABDEM only for the last two)
DISTRICTS = {"sivasagar": ["SIVASAGAR"], "nagaon_morigaon": ["NAGAON", "MORIGAON"], "karimganj": ["KARIMGANJ"]}[AREA]
if DEM_SRC == "fabdem":
    TILES = os.path.join(ROOT, "data", "dem_fabdem_upstream_" + AREA)
    OUT = os.path.join(ROOT, "data", "basin_delineation", AREA + "_fabdem")
else:
    TILES = os.path.join(ROOT, "data", "dem_upstream_sivasagar")
    OUT = os.path.join(ROOT, "data", "basin_delineation", "sivasagar")
CWC = os.path.join(ROOT, "data", "basin_delineation", "cwc_stations_vs_hydrobasins.csv")
CRS = "EPSG:32646"
RES = 30.0
SNAP_M = float(os.environ.get("SNAP_M", 500.0))
os.makedirs(OUT, exist_ok=True)
P = lambda n: os.path.join(OUT, n)


def build_dem():
    if os.path.exists(P("dem_utm46.tif")):
        return
    src = [rasterio.open(f) for f in sorted(glob.glob(os.path.join(TILES, "*.tif")))]
    mos, tr = merge(src)
    mos[~np.isfinite(mos)] = -9999.0                  # FABDEM pieces can carry -inf cells (seen in karimganj piece 0_0)
    prof = dict(driver="GTiff", height=mos.shape[1], width=mos.shape[2], count=1, dtype="float32", crs="EPSG:4326", transform=tr, nodata=-9999.0)
    with rasterio.open(P("dem_mosaic_4326.tif"), "w", **prof) as d:
        d.write(mos[0].astype("float32"), 1)
    with rasterio.open(P("dem_mosaic_4326.tif")) as s:
        t, w, h = calculate_default_transform(s.crs, CRS, s.width, s.height, *s.bounds, resolution=RES)
        prof.update(crs=CRS, transform=t, width=w, height=h)
        with rasterio.open(P("dem_utm46.tif"), "w", **prof) as d:
            reproject(rasterio.band(s, 1), rasterio.band(d, 1), src_transform=s.transform, src_crs=s.crs, dst_transform=t, dst_crs=CRS,
                      resampling=Resampling.bilinear, dst_nodata=-9999.0)
    print("DEM mosaic + UTM 46N done", flush=True)


def run_wbt():
    wbt = WhiteboxTools()
    wbt.set_verbose_mode(False)
    wbt.set_working_dir(OUT)
    if not os.path.exists(P("dem_filled.tif")):
        wbt.fill_depressions("dem_utm46.tif", "dem_filled.tif"); print("filled", flush=True)
    if not os.path.exists(P("d8_pointer.tif")):
        wbt.d8_pointer("dem_filled.tif", "d8_pointer.tif"); print("d8 pointer", flush=True)
    if not os.path.exists(P("d8_accum.tif")):
        wbt.d8_flow_accumulation("dem_filled.tif", "d8_accum.tif", out_type="cells"); print("d8 accumulation", flush=True)
    return wbt


def main():
    build_dem()
    wbt = run_wbt()
    S = pd.read_csv(CWC)
    S = S[S.district_n.isin(DISTRICTS)].reset_index(drop=True)
    G = gpd.GeoDataFrame(S[["name", "river_name"]], geometry=gpd.points_from_xy(S.lon, S.lat), crs=4326).to_crs(CRS)
    G["FID"] = range(1, len(G) + 1)
    G.to_file(P("gauges.shp"))
    wbt.snap_pour_points("gauges.shp", "d8_accum.tif", "gauges_snapped.shp", snap_dist=SNAP_M)
    snapped = gpd.read_file(P("gauges_snapped.shp"))
    # WhiteboxTools snapped Patharkandi (Karimganj) to a ~0 km2 cell although an 838 km2 channel lies 360 m away:
    # for any snapped point with < 5 km2 of upstream area, redo the snap in numpy (max accumulation within SNAP_M)
    with rasterio.open(P("d8_accum.tif")) as a:
        A = a.read(1); at = a.transform; cell = abs(at.a * at.e) / 1e6
        for i, r in G.iterrows():
            sp = snapped.geometry.iloc[i]
            if float(list(a.sample([(sp.x, sp.y)]))[0][0]) * cell >= 5:
                continue
            row, col = a.index(r.geometry.x, r.geometry.y); k = int(SNAP_M / abs(at.a))
            r0, c0 = max(row - k, 0), max(col - k, 0)
            win = A[r0:row + k + 1, c0:col + k + 1].astype(float)
            yy, xx = np.mgrid[r0:r0 + win.shape[0], c0:c0 + win.shape[1]]
            win[np.hypot(yy - row, xx - col) * abs(at.a) > SNAP_M] = -1
            j = np.unravel_index(np.argmax(win), win.shape)
            x, y = at * (c0 + j[1] + 0.5, r0 + j[0] + 0.5)
            snapped.loc[i, "geometry"] = Point(x, y)
            print(f"re-snapped {r['name']} manually: {win[j] * cell:.0f} km2", flush=True)
    rows, feats = [], []
    for i, r in G.iterrows():
        # one trace per gauge: nested gauges (Sivasagar is downstream of Bihubar on the Dikhow) each get their FULL basin
        snapped.iloc[[i]].to_file(P(f"snap_{i}.shp"))
        wbt.watershed("d8_pointer.tif", f"snap_{i}.shp", f"basin_{i}.tif")
        with rasterio.open(P(f"basin_{i}.tif")) as b, rasterio.open(P("d8_accum.tif")) as a:
            B = b.read(1); tr = b.transform
            cell_km2 = abs(tr.a * tr.e) / 1e6
            lab = B > 0
            n = int(lab.sum())
            rr, cc = np.where(lab)
            edge = bool(rr.min() == 0 or cc.min() == 0 or rr.max() == B.shape[0] - 1 or cc.max() == B.shape[1] - 1)
            sp = snapped.geometry.iloc[i]
            acc = float(list(a.sample([(sp.x, sp.y)]))[0][0])
            rows.append(dict(gauge=r["name"], river=r.river_name, snapped_m=round(r.geometry.distance(sp)), accum_area_km2=round(acc * cell_km2),
                             basin_km2=round(n * cell_km2, 1), touches_dem_edge=edge))
            for g, v in shapes(lab.astype("uint8"), mask=lab, transform=tr):
                feats.append(dict(gauge=r["name"], river=r.river_name, basin_km2=round(n * cell_km2, 1), geometry=shape(g)))
    gdf = gpd.GeoDataFrame(feats, crs=CRS).dissolve(by=["gauge", "river", "basin_km2"], as_index=False).to_crs(4326)
    gdf.to_file(P("basins_traced.gpkg"), layer="basins", driver="GPKG")
    D = pd.DataFrame(rows)
    D.to_csv(P("summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print(D.to_string(index=False))


if __name__ == "__main__":
    main()
