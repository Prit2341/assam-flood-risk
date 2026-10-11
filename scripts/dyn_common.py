"""
Shared grid + masks for the dynamic flood susceptibility work (scripts 130-160).

Analysis grid: each district's FABDEM 30 m grid (EPSG:4326, 1 arc-second) aggregated 3x3 -> ~90 m.
90 m keeps ~10^5-10^6 cells per district so hundreds of dated GFM scenes fit in memory; it is the
same order as SRTM/MERIT-based susceptibility work. Stated as a compute choice, not a finding.

Masks applied to every layer (diagnosed 2026-10-10, Chroma manual-0131):
  * outside the district polygon (data/boundaries/scope4_districts.geojson);
  * FABDEM fill value 0 m (Nagaon SE corner, rows 2599+, cols 2244+: no source tile, filled with 0);
  * hydro-flattened water: Copernicus GLO-30 (FABDEM's source) flattens rivers/lakes to constant,
    stepped heights; detected as 30 m windows with exactly zero 7x7 relief. These produced the
    "straight-edged bands" in the Phase-1 RF maps. River channels are permanent water, not
    floodable land, so they are excluded (GFM also excludes reference water from flood extent).
"""
import json
import os

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from scipy.ndimage import maximum_filter, minimum_filter, binary_opening
from shapely.geometry import shape

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
DYN = os.path.join(DATA, "dynamic_susceptibility")
DISTRICTS = ["Morigaon", "Nagaon", "Karimganj", "Sivasagar"]
POLY_NAME = {"Morigaon": "Marigaon"}
AGG = 3


def district_polygon(d):
    gj = json.load(open(os.path.join(DATA, "boundaries", "scope4_districts.geojson"), encoding="utf-8"))
    name = POLY_NAME.get(d, d)
    for f in gj["features"]:
        if f["properties"]["dtname"] == name:
            return shape(f["geometry"])
    raise KeyError(d)


def fabdem(d):
    with rasterio.open(os.path.join(DATA, "dem_fabdem", f"{d}_fabdem_30m.tif")) as s:
        return s.read(1).astype(np.float32), s.transform, s.crs


def water_flat_mask30(elev):
    """True where the DEM is hydro-flattened (exact zero relief in a 7x7 window) or fill value 0."""
    relief = maximum_filter(elev, 7) - minimum_filter(elev, 7)
    flat = relief == 0
    flat = binary_opening(flat, iterations=2)   # drop isolated tiny flat patches (real flat fields)
    return flat | (elev <= 0)


def grid(d):
    """Return (transform90, shape90, in_district_mask90, invalid_dem_mask90)."""
    elev, tr, crs = fabdem(d)
    h, w = elev.shape[0] // AGG, elev.shape[1] // AGG
    tr90 = tr * tr.scale(AGG, AGG)
    poly = district_polygon(d)
    inside = ~geometry_mask([poly.__geo_interface__], (h, w), tr90, invert=False)
    bad30 = water_flat_mask30(elev)[: h * AGG, : w * AGG]
    bad90 = bad30.reshape(h, AGG, w, AGG).mean(axis=(1, 3)) >= 0.5
    return tr90, (h, w), inside, bad90


def agg_mean(a30, h, w):
    return a30[: h * AGG, : w * AGG].reshape(h, AGG, w, AGG).mean(axis=(1, 3))


def save_tif(path, arr, tr, crs="EPSG:4326", nodata=-9999.0):
    a = np.where(np.isnan(arr), nodata, arr).astype(np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1,
                       dtype="float32", crs=crs, transform=tr, nodata=nodata, compress="deflate") as dst:
        dst.write(a, 1)
