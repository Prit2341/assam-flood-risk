"""
Flood susceptibility for Nagaon district — Assam Phase 3
Pipeline:
  1. Mosaic + clip GLO-30 DEM to district boundary
  2. Derive: slope, TWI (Topographic Wetness Index), flow accumulation
  3. Distance-to-river from HydroSHEDS (or DEM-derived drainage)
  4. Weighted linear combination -> susceptibility [0-1]
  5. Resample to 0.001deg (~100m) GeoJSON grid tiles for dashboard

Methodology: Tehrany et al. (2014) Nat Hazards Earth Syst Sci 14:1881-1897
  TWI = ln(a / tan(beta))  where a=flow_accum_area, beta=slope_radians
"""

from pathlib import Path
import numpy as np
# pysheds 0.5 uses np.in1d removed in NumPy 2.x
if not hasattr(np, 'in1d'):
    np.in1d = lambda ar1, ar2, **kw: np.isin(ar1, ar2, **kw).ravel()
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import shape, mapping
import json

DEM_DIR  = Path(__file__).parent.parent / 'data' / 'dem'
OUT_DIR  = Path(__file__).parent
GEO_JSON = Path(__file__).parent.parent / 'dashboard-app' / 'frontend' / 'public' / 'districts_geo.json'

DISTRICT = 'nagaon'
OUT_DIR.mkdir(exist_ok=True)


# ── 1. Load district boundary ─────────────────────────────────────────────────
with open(GEO_JSON) as f:
    geo_data = json.load(f)

district_geom = shape(geo_data[DISTRICT]['geometry'])
district_gdf  = gpd.GeoDataFrame(geometry=[district_geom], crs='EPSG:4326')

print(f'District bounds: {district_geom.bounds}')


# ── 2. Mosaic DEM tiles ───────────────────────────────────────────────────────
tif_files = sorted(DEM_DIR.glob('*.tif'))
print(f'Found {len(tif_files)} DEM tiles: {[f.name for f in tif_files]}')

src_files = [rasterio.open(f) for f in tif_files]
mosaic, mosaic_transform = merge(src_files)
mosaic_meta = src_files[0].meta.copy()
mosaic_meta.update({'driver': 'GTiff', 'height': mosaic.shape[1],
                    'width': mosaic.shape[2], 'transform': mosaic_transform,
                    'crs': 'EPSG:4326'})

mosaic_path = OUT_DIR / 'dem_mosaic.tif'
with rasterio.open(mosaic_path, 'w', **mosaic_meta) as dst:
    dst.write(mosaic)
for s in src_files:
    s.close()
print(f'Mosaic saved: {mosaic_path}')


# ── 3. Clip to district boundary ─────────────────────────────────────────────
with rasterio.open(mosaic_path) as src:
    out_image, out_transform = mask(src, [mapping(district_geom)], crop=True, nodata=-9999)
    out_meta = src.meta.copy()

out_meta.update({'height': out_image.shape[1], 'width': out_image.shape[2],
                 'transform': out_transform, 'nodata': -9999})

dem_path = OUT_DIR / f'dem_{DISTRICT}.tif'
with rasterio.open(dem_path, 'w', **out_meta) as dst:
    dst.write(out_image)
print(f'Clipped DEM saved: {dem_path}  shape={out_image.shape}')


# ── 4. Terrain derivatives with pysheds ──────────────────────────────────────
from pysheds.grid import Grid

grid  = Grid.from_raster(str(dem_path))
dem   = grid.read_raster(str(dem_path))

# Fill pits/depressions
print('Filling pits...')
pit_filled  = grid.fill_pits(dem)
flooded     = grid.fill_depressions(pit_filled)
inflated    = grid.resolve_flats(flooded)

# Flow direction (D8)
print('Computing flow direction...')
dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
fdir   = grid.flowdir(inflated, dirmap=dirmap)

# Flow accumulation
print('Computing flow accumulation...')
acc    = grid.accumulation(fdir, dirmap=dirmap)

# Slope (radians) — central difference on DEM
print('Computing slope...')
dem_arr = np.array(inflated)
cell_size_m = 30.0  # GLO-30 ~30m
dx = np.gradient(dem_arr, axis=1) / cell_size_m
dy = np.gradient(dem_arr, axis=0) / cell_size_m
slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
slope_deg = np.degrees(slope_rad)

# TWI = ln(a / tan(beta))  [Beven & Kirkby 1979]
print('Computing TWI...')
acc_arr   = np.array(acc, dtype=float)
acc_area  = (acc_arr + 1) * (cell_size_m ** 2)  # contributing area m²
tan_beta  = np.tan(slope_rad + 1e-6)             # avoid /0
twi       = np.log(acc_area / tan_beta)
twi       = np.clip(twi, 0, 30)


# ── 5. Normalise layers [0,1] ────────────────────────────────────────────────
def norm(arr, nodata_val=-9999):
    valid = arr[arr != nodata_val]
    mn, mx = np.percentile(valid, 2), np.percentile(valid, 98)
    return np.clip((arr - mn) / (mx - mn + 1e-9), 0, 1)

dem_arr_raw = np.array(inflated)
elev_norm  = 1.0 - norm(dem_arr_raw)   # low elevation → high susceptibility
slope_norm = 1.0 - norm(slope_deg)     # low slope → high susceptibility (floodplain)
twi_norm   =       norm(twi)           # high TWI → high susceptibility
acc_norm   =       norm(np.log1p(np.array(acc, dtype=float)))  # high acc → high

# Weighted combination — weights from Tehrany et al. 2014 adapted
# TWI 35%, elevation 30%, flow_acc 20%, slope 15%
susceptibility = (
    0.35 * twi_norm +
    0.30 * elev_norm +
    0.20 * acc_norm +
    0.15 * slope_norm
)

print(f'Susceptibility range: {susceptibility.min():.3f} – {susceptibility.max():.3f}')


# ── 6. Save susceptibility raster ─────────────────────────────────────────────
susc_path = OUT_DIR / f'susceptibility_{DISTRICT}.tif'
with rasterio.open(dem_path) as src:
    susc_meta = src.meta.copy()
    susc_meta.update({'count': 1, 'dtype': 'float32', 'nodata': -9999})
    with rasterio.open(susc_path, 'w', **susc_meta) as dst:
        out = susceptibility.astype(np.float32)
        # mask nodata
        with rasterio.open(dem_path) as d:
            nd = d.read(1)
        out[nd == -9999] = -9999
        dst.write(out[np.newaxis, :, :])

print(f'Susceptibility raster saved: {susc_path}')


# ── 7. Export grid tiles as GeoJSON for dashboard ────────────────────────────
print('Exporting grid tiles for dashboard...')

GRID_DEG = 0.01  # ~1km grid cells
from rasterio.windows import from_bounds as window_from_bounds
from shapely.geometry import box as shapely_box

with rasterio.open(susc_path) as src:
    susc_arr = src.read(1)
    transform = src.transform
    bounds    = src.bounds
    nodata    = src.nodata

    print(f'Raster bounds: {bounds}')
    print(f'Susc array shape: {susc_arr.shape}, range {susc_arr[susc_arr != nodata].min():.3f}-{susc_arr[susc_arr != nodata].max():.3f}')

    features = []
    minlon, minlat, maxlon, maxlat = bounds.left, bounds.bottom, bounds.right, bounds.top

    lons = np.arange(minlon, maxlon, GRID_DEG)
    lats = np.arange(minlat, maxlat, GRID_DEG)

    for lat0 in lats:
        lat1 = lat0 + GRID_DEG
        for lon0 in lons:
            lon1 = lon0 + GRID_DEG

            # Quick spatial filter
            cell_box = shapely_box(lon0, lat0, lon1, lat1)
            if not district_geom.intersects(cell_box):
                continue

            win = window_from_bounds(lon0, lat0, lon1, lat1, transform)
            try:
                cell = src.read(1, window=win)
                valid = cell[(cell != nodata) & np.isfinite(cell)]
                if len(valid) == 0:
                    continue
                val = float(np.median(valid))
            except Exception:
                continue

            if not np.isfinite(val) or val <= 0:
                continue

            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[[lon0, lat0],[lon0, lat1],[lon1, lat1],[lon1, lat0],[lon0, lat0]]]
                },
                'properties': {'susc': round(val, 3)}
            })

out_geojson = OUT_DIR / f'susceptibility_{DISTRICT}_grid.geojson'
with open(out_geojson, 'w') as f:
    json.dump({'type': 'FeatureCollection', 'features': features}, f)

print(f'Grid GeoJSON: {out_geojson}  ({len(features)} cells)')
print('Done.')
