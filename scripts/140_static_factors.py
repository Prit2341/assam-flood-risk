"""
Static flood-conditioning factors on the ~90 m analysis grid (dyn_common.grid), all 4 districts.

Factor set = the AHP factor set of Choudhury & Dutta 2026 (Discover Environment 4:138, Dimoria block,
Assam; open access, pairwise matrix in their Table 4) minus lithology (no open digital geology map at
usable scale for our districts), plus HAND (used by the ML arms only; Nobre et al. 2011, J. Hydrol.
404:13; Nobre et al. 2016, J. Hydrol. 541:1001 for HAND as a flood-proneness descriptor).

Definitions (computed on FABDEM 30 m, then 3x3 mean to 90 m):
  slope      degrees, metric spacing per row (dx = res*111320*cos(lat), dy = res*110574)
  TWI        ln(a / tan b), a = specific catchment area (Beven & Kirkby 1979), tan b floored at 0.001
  STI        (a/22.13)^0.6 * (sin b/0.0896)^1.3 (Moore & Burch 1986, SSSAJ 50:1294)
  TRI        sqrt(sum of squared elevation differences to 8 neighbours) (Riley et al. 1999)
  HAND       height above nearest drainage, channels = upstream area >= 10 km2 (HydroRIVERS'
             inclusion threshold, Lehner & Grill 2013, so the channel network matches the river layer)
  dist_river km to nearest HydroRIVERS v1.0 reach (Lehner & Grill 2013, Hydrol. Process. 27:2171)
  dist_major km to nearest HydroRIVERS reach with upstream area >= 1000 km2 (AHP distance factor)
  drain_dens river length (HydroRIVERS, rasterised 30 m) per km2 within a 2.5 km radius
  soil_hsg   SoilGrids 0-5 cm sand/clay -> USDA texture class -> NRCS hydrologic soil group
             (USDA NRCS NEH Part 630 ch. 7, 2009, group descriptions), A=1 B=2 C=3 D=4
  ndvi       Sentinel-2 SR median NDVI, dry season (Nov-Mar) 2022-23..2024-25, SCL cloud mask (GEE)
  lulc       Dynamic World V1 mode label, same dry-season windows (Brown et al. 2022, Sci. Data 9:251)
Masks (outside district, DEM-flattened water, DEM fill) are applied in dyn_common.

Run: python -u scripts/140_static_factors.py <District>|all
Out: data/dynamic_susceptibility/static/<District>_static.npz (+ GeoTIFFs for inspection)
"""
import os
import sys

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from scipy.ndimage import convolve, distance_transform_edt

if not hasattr(np, "in1d"):
    np.in1d = np.isin
from pysheds.grid import Grid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dyn_common as dc

OUT = os.path.join(dc.DYN, "static")
RIVERS = os.path.join(dc.DATA, "hydrorivers", "HydroRIVERS_v10_as_shp", "HydroRIVERS_v10_as.shp")
CHANNEL_KM2 = 10.0
DD_RADIUS_KM = 2.5
MAJOR_KM2 = 1000.0


def terrain30(d):
    path = os.path.join(dc.DATA, "dem_fabdem", f"{d}_fabdem_30m.tif")
    g = Grid.from_raster(path)
    dem = g.read_raster(path)
    inflated = g.resolve_flats(g.fill_depressions(g.fill_pits(dem)))
    fdir = g.flowdir(inflated)
    acc_r = g.accumulation(fdir)
    acc = np.asarray(acc_r, dtype=np.float64) + 1.0
    elev, tr, _ = dc.fabdem(d)
    res = abs(tr.a)
    lat = tr.f + (np.arange(elev.shape[0]) + 0.5) * tr.e
    dx = (res * 111320.0 * np.cos(np.radians(lat)))[:, None] * np.ones((1, elev.shape[1]))
    dy = res * 110574.0
    gy, gx = np.gradient(elev.astype(np.float64))
    tanb = np.sqrt((gx / dx) ** 2 + (gy / dy) ** 2)
    slope = np.degrees(np.arctan(tanb))
    cell_area = dx * dy
    sca = acc * cell_area / dx
    tanb_f = np.maximum(tanb, 0.001)
    twi = np.log(sca / tanb_f)
    sinb = np.sin(np.arctan(tanb_f))
    sti = (sca / 22.13) ** 0.6 * (sinb / 0.0896) ** 1.3
    k = np.ones((3, 3)); k[1, 1] = 0
    e = elev.astype(np.float64)
    tri = np.sqrt(sum((np.roll(np.roll(e, i, 0), j, 1) - e) ** 2
                      for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)))
    chan = acc_r > (CHANNEL_KM2 * 1e6 / cell_area.mean())       # Raster-typed mask for pysheds
    hand = np.asarray(g.compute_hand(fdir, inflated, chan), dtype=np.float64)
    return {"elevation": e, "slope": slope, "twi": twi, "sti": np.log1p(sti), "tri": tri,
            "log_acc": np.log10(acc), "hand": hand}, tr, cell_area


def rivers_layers(d, tr30, shape30, tr90, shape90):
    poly = dc.district_polygon(d)
    b = poly.buffer(0.2).bounds
    rv = gpd.read_file(RIVERS, bbox=b)
    r30 = rasterize(((g, 1) for g in rv.geometry), out_shape=shape30, transform=tr30, all_touched=True,
                    dtype="uint8")
    res30 = abs(tr30.a)
    lat30 = tr30.f + (np.arange(shape30[0]) + 0.5) * tr30.e
    dxm = res30 * 111320.0 * np.cos(np.radians(lat30.mean()))
    dym = res30 * 110574.0
    dist = distance_transform_edt(r30 == 0, sampling=(dym, dxm)) / 1000.0
    # AHP distance factor: the paper's breaks (0-2.6 ... >10.5 km) assume major rivers; with every
    # HydroRIVERS reach (>=10 km2) almost all cells fall in the first class. Major = upstream area
    # >= MAJOR_KM2 (HydroRIVERS UPLAND_SKM). Threshold is a stated choice, not from the paper.
    maj = rv[rv["UPLAND_SKM"] >= MAJOR_KM2]
    m30 = rasterize(((g, 1) for g in maj.geometry), out_shape=shape30, transform=tr30, all_touched=True,
                    dtype="uint8") if len(maj) else np.zeros(shape30, np.uint8)
    dist_major = distance_transform_edt(m30 == 0, sampling=(dym, dxm)) / 1000.0
    # drainage density: rasterised length ~ cell size per river cell (Cartesian approx.)
    rad_px = int(round(DD_RADIUS_KM * 1000 / dym))
    yy, xx = np.mgrid[-rad_px:rad_px + 1, -rad_px:rad_px + 1]
    disk = ((yy * dym) ** 2 + (xx * dxm) ** 2 <= (DD_RADIUS_KM * 1000) ** 2).astype(np.float32)
    length_km = r30.astype(np.float32) * ((dxm + dym) / 2 / 1000.0)
    dd = convolve(length_km, disk, mode="constant") / (disk.sum() * dxm * dym / 1e6)
    return {"dist_river": dist, "dist_major": dist_major, "drain_dens": dd}, len(rv)


def usda_hsg(sand, clay):
    """NRCS hydrologic soil group from texture, NEH Part 630 ch. 7 (2009) group descriptions:
    A <10% clay & >90% sand; B 10-20% clay & 50-90% sand; C 20-40% clay & <50% sand; D >40% clay.
    Combinations outside those boxes are assigned by clay content (the controlling criterion).
    Returns 1 (A, lowest runoff) .. 4 (D, highest runoff)."""
    g = np.where(clay > 40, 4, np.where(clay > 20, 3, np.where(clay >= 10, 2, np.where(sand > 90, 1, 2))))
    g = g.astype(np.float32)
    g[np.isnan(sand) | np.isnan(clay)] = np.nan
    return g


def soil_layer(d, tr90, shape90):
    out = {}
    for p in ("sand", "clay"):
        src_p = os.path.join(dc.DATA, "soilgrids_full_profile", d, f"{d}_{p}_0-5cm_converted.tif")
        with rasterio.open(src_p) as s:
            a = s.read(1).astype(np.float32)
            a[a <= 0] = np.nan
            dst = np.full(shape90, np.nan, np.float32)
            reproject(a, dst, src_transform=s.transform, src_crs=s.crs, dst_transform=tr90,
                      dst_crs="EPSG:4326", resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
            out[p] = dst
    return usda_hsg(out["sand"], out["clay"]), out["sand"], out["clay"]


def gee_layers(d, tr90, shape90):
    import ee
    ee.Initialize(project="assamflood-508209")
    h, w = shape90
    west, north = tr90.c, tr90.f
    east, south = west + w * tr90.a, north + h * tr90.e
    region = ee.Geometry.Rectangle([west, south, east, north], proj="EPSG:4326", geodesic=False)
    wins = [("2022-11-01", "2023-03-31"), ("2023-11-01", "2024-03-31"), ("2024-11-01", "2025-03-31")]

    def s2():
        col = None
        for a, b in wins:
            c = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region).filterDate(a, b)
            col = c if col is None else col.merge(c)

        def nd(img):
            scl = img.select("SCL")
            ok = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
            return img.normalizedDifference(["B8", "B4"]).rename("ndvi").updateMask(ok)
        return col.map(nd).median()

    def dw():
        col = None
        for a, b in wins:
            c = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(region).filterDate(a, b).select("label")
            col = c if col is None else col.merge(c)
        return col.mode().rename("lulc")

    out = {}
    NT = 3                                   # 3x3 tiles: one whole-district request exceeded GEE user memory
    for name, img in (("ndvi", s2()), ("lulc", dw())):
        # GEE resamples inputs to the 90 m request grid via its pyramid (mean for S2 bands,
        # mode for the Dynamic World label band) before compositing.
        dst = np.full(shape90, np.nan, np.float32)
        rb = np.linspace(0, h, NT + 1).astype(int); cb = np.linspace(0, w, NT + 1).astype(int)
        for i in range(NT):
            for j in range(NT):
                r0, r1, c0, c1 = rb[i], rb[i + 1], cb[j], cb[j + 1]
                x0, y0 = west + c0 * tr90.a, north + r0 * tr90.e
                sub = ee.Geometry.Rectangle([x0, north + r1 * tr90.e, west + c1 * tr90.a, y0],
                                            proj="EPSG:4326", geodesic=False)
                for attempt in range(4):
                    try:
                        url = img.getDownloadURL({"region": sub, "crs": "EPSG:4326",
                                                  "crs_transform": [tr90.a, 0, x0, 0, tr90.e, y0], "format": "GEO_TIFF"})
                        r = requests.get(url, timeout=900)
                        if r.status_code != 200:
                            raise RuntimeError(f"GEE {name}: {r.status_code} {r.text[:300]}")
                        break
                    except Exception as e:
                        print(f"   {name} tile {i},{j} attempt {attempt+1} failed: {str(e)[:120]}", flush=True)
                        if attempt == 3:
                            raise
                tmp = os.path.join(OUT, f"_{d}_{name}_{i}{j}.tif")
                open(tmp, "wb").write(r.content)
                with rasterio.open(tmp) as src:
                    reproject(src.read(1).astype(np.float32), dst, src_transform=src.transform, src_crs=src.crs,
                              dst_transform=tr90, dst_crs="EPSG:4326", resampling=Resampling.nearest,
                              src_nodata=(None if name == "lulc" else src.nodata), dst_nodata=np.nan,
                              init_dest_nodata=False)
        out[name] = dst
    return out


def run(d):
    os.makedirs(OUT, exist_ok=True)
    tr90, shape90, inside, bad = dc.grid(d)
    h, w = shape90
    cache = os.path.join(OUT, f"_{d}_nongee.npz")
    if os.path.exists(cache):
        z = np.load(cache); layers = {k: z[k] for k in z.files if k != "nrv"}; nrv = int(z["nrv"])
    else:
        print(f"{d}: terrain ...", flush=True)
        t30, tr30, _ = terrain30(d)
        layers = {k: dc.agg_mean(v, h, w) for k, v in t30.items()}
        print(f"{d}: rivers ...", flush=True)
        rl, nrv = rivers_layers(d, tr30, t30["elevation"].shape, tr90, shape90)
        layers.update({k: dc.agg_mean(v, h, w) for k, v in rl.items()})
        del t30
        print(f"{d}: soil ...", flush=True)
        layers["soil_hsg"], layers["sand"], layers["clay"] = soil_layer(d, tr90, shape90)
        np.savez_compressed(cache, nrv=nrv, **layers)
    print(f"{d}: GEE ndvi/lulc ...", flush=True)
    layers.update(gee_layers(d, tr90, shape90))
    usable = inside & ~bad
    for k in layers:
        layers[k] = np.where(usable, layers[k], np.nan).astype(np.float32)
        dc.save_tif(os.path.join(OUT, f"{d}_{k}.tif"), layers[k], tr90)
    np.savez_compressed(os.path.join(OUT, f"{d}_static.npz"), usable=usable, **layers)
    print(f"{d}: {nrv} HydroRIVERS reaches; usable cells {usable.sum():,}", flush=True)
    for k, v in layers.items():
        print(f"   {k:11s} nan={np.isnan(v[usable]).mean():.3f} p5={np.nanpercentile(v, 5):9.3f} "
              f"p50={np.nanpercentile(v, 50):9.3f} p95={np.nanpercentile(v, 95):9.3f}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "all"
    for d in (dc.DISTRICTS if a == "all" else [a]):
        run(d)
