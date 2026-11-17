"""Build isolated B1 project after registered withheld-rain checks pass."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import MultiPoint
from shapely import intersects_xy
from ras_commander.RasUnsteady import RasUnsteady

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'digital_twin/sivasagar'
OUT = BASE/'results/rain_b1'
assert json.loads((OUT/'validation.json').read_text())['hydraulic_pilot_allowed']
source = BASE/'hecras_c600_d4_grid_imerg'
dest = BASE/'hecras_c600_d4_grid_rainb1'
if dest.exists():
    raise FileExistsError(f'Refusing to overwrite existing experiment {dest}')
daily = pd.read_csv(OUT/'daily_gauge_residuals.csv')
with xr.open_dataset(source/'rain_grid.nc') as src:
    ds = src.load()
times = pd.DatetimeIndex(ds.time.values).as_unit('ns')
original = ds.APCP_surface.values.copy()
rain = original.astype(float)/.3
xx, yy = np.meshgrid(ds.x.values,ds.y.values)
target_xy = np.c_[xx.ravel(),yy.ravel()]
domain = gpd.read_file(BASE/'data/boundary_utm46n.gpkg').geometry.iloc[0]
in_domain = intersects_xy(domain,xx,yy)
details = []
for end,g in daily.groupby('day_end_utc'):
    end = pd.Timestamp(end)
    sel = (times > end-pd.Timedelta(days=1)) & (times <= end)
    assert sel.sum() == 24
    xy = g[['x','y']].to_numpy()
    hull = MultiPoint(xy).convex_hull
    supported = intersects_xy(hull,xx,yy)
    distances = np.linalg.norm(target_xy[:,None,:]-xy[None,:,:],axis=2)
    weights = 1/np.maximum(distances,1)**2
    weights /= weights.sum(axis=1,keepdims=True)
    residual = (weights@(g.obs_mm-g.sat_mm).to_numpy()).reshape(xx.shape)
    totals = rain[sel].sum(axis=0)
    wanted = np.maximum(0,totals+residual)
    zero_positive = supported & (totals == 0) & (wanted > 0)
    # Preserve zero-total timing rather than inventing a subdaily pulse.
    apply = supported & (totals > 0)
    factor = np.ones_like(totals)
    factor[apply] = wanted[apply]/totals[apply]
    rain[sel] *= factor
    assert np.allclose(rain[sel].sum(axis=0)[apply],wanted[apply],atol=1e-5)
    assert np.array_equal(rain[sel][:,~supported],original[sel][:,~supported].astype(float)/.3)
    details.append(dict(day_end_utc=str(end),gauges=len(g),
        supported_domain_grid_fraction=float((supported&in_domain).sum()/in_domain.sum()),
        positive_correction_on_zero_total_pixels_skipped=int(zero_positive.sum()),
        mean_raw_input_mm=float(totals[in_domain].mean()),
        mean_corrected_input_mm=float(rain[sel][:,in_domain].sum(axis=0).mean())))
effective = (rain*.3).astype(original.dtype)
assert np.isfinite(effective).all() and (effective >= 0).all()
assert np.any(effective != original)
shutil.copytree(source,dest,ignore=shutil.ignore_patterns('*.p01.hdf','*.b01','*.bco01','*.ic.o01','*.x01','_compute_p01.log','rain_grid.nc'))
ds['APCP_surface'].values = effective
ds.to_netcdf(dest/'rain_grid.nc')
RasUnsteady._update_precipitation_hdf(dest/'Sivasagar_2D.u01.hdf',dest/'rain_grid.nc',r'.\rain_grid.nc','Bilinear','APCP_surface')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
geometry = 'Sivasagar_2D.g01'
assert sha(source/geometry) == sha(dest/geometry)
terrain = []
for p in (source/'Terrain').rglob('*'):
    if p.is_file():
        rel = p.relative_to(source)
        assert sha(p) == sha(dest/rel)
        terrain.append(str(rel))
report = dict(source=str(source),project=str(dest),geometry_sha256=sha(source/geometry),
    terrain_files_identical=terrain,days=details,runoff_multiplier=.3,
    unsupported_grid_cells_unchanged=True,
    partial_days_unchanged=True,method='Daily IDW2 additive residual, nonnegative totals; retain original within-day IMERG fractions. Convex-hull support only. Zero-original-total positive corrections flagged/skipped, no invented timing.')
(OUT/'build.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
