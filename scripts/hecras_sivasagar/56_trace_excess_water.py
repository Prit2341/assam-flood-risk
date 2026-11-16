"""Diagnostic only: trace July 17 false alarms through saved raw RAS output."""
from pathlib import Path
import importlib.util
import json
import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from pyproj import Transformer
from shapely.geometry import Polygon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R = ROOT / 'digital_twin/sivasagar/results'
spec = importlib.util.spec_from_file_location('score', HERE / '05_score.py')
score = importlib.util.module_from_spec(spec); spec.loader.exec_module(score)
tag, when = score.SCENES['2026-07-17']
tif = next(score.GFM.glob(f'*{tag}*_flood.tif'))
pre_tif = next(score.GFM.glob('*pre_2026-07-05*_flood.tif'))
with rasterio.open(tif) as src:
    obs = src.read(1).ravel(); shape = src.shape; tr = src.transform; crs = src.crs
with rasterio.open(pre_tif) as src:
    assert src.shape == shape and src.transform == tr and src.crs == crs
    pre = src.read(1).ravel()
z = score.raw_z(tif, 0)
with h5py.File(R / 'c600_d4_r30_0705_432h_raw.p01.hdf') as f:
    g = f['Geometry/2D Flow Areas/Sivasagar_Domain']
    vertices = g['Cells FacePoint Indexes'][:]
    real = np.flatnonzero((vertices >= 0).sum(axis=1) >= 3)
    points = g['FacePoints Coordinate'][:]
    x, y = Transformer.from_crs(32646, crs, always_xy=True).transform(points[:, 0], points[:, 1])
    projected = np.c_[x, y]
    poly = [(Polygon(projected[row[row >= 0]]), int(cid)) for cid, row in zip(real, vertices[real])]
    owner = rasterize(poly, out_shape=shape, transform=tr, fill=-1, dtype='int32').ravel()
    times = score.hdf_times(f)
    base = 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Sivasagar_Domain'
    wse = f[f'{base}/Water Surface']
    at = int(np.argmin(abs(times-pd.Timestamp(when))))
    valid = (obs != 255) & (pre != 1) & np.isfinite(z) & (owner >= 0)
    model_at = wse[at,:]
    fa = valid & (obs == 0) & (model_at[np.maximum(owner, 0)]-z > .1)
    hit = valid & (obs == 1) & (model_at[np.maximum(owner, 0)]-z > .1)
    # Daily checkpoints reveal timing without storing a 433-by-pixel cube.
    fa_idx = np.flatnonzero(fa); fa_cell = owner[fa_idx]; fa_z = z[fa_idx]
    onset = np.full(fa_idx.size, -1, np.int16)
    for hour in range(0, at+1, 24):
        wet = wse[hour,:][fa_cell]-fa_z > .1
        onset[(onset < 0) & wet] = hour
    wet = model_at[fa_cell]-fa_z > .1
    onset[(onset < 0) & wet] = at
    assert np.all(onset >= 0)
    # Projected pixel x gives west-to-east spatial distribution; not flow paths.
    rr, cc = np.divmod(fa_idx, shape[1])
    px, py = tr*(cc+.5, rr+.5)
    ux, uy = Transformer.from_crs(crs, 32646, always_xy=True).transform(px, py)
    xbounds = np.linspace(float(ux.min()), float(ux.max()), 4)
    west = [dict(zone=k, fa_km2=float(np.count_nonzero((ux >= xbounds[k]) & (ux <= xbounds[k+1]))*.0004)) for k in range(3)]
    rain = f[f'{base}/Cell Cumulative Precipitation Depth'][at,real]
    bc = f[f'{base}/Boundary Conditions/Outlet_West'][:at+1]
    flow = bc[:,1]; out_m3 = float(np.trapezoid(flow, dx=3600))
    volume_m3 = float(f[f'{base}/Computations/Volume'][at,0]*1000)
    n = g["Cells Center Manning's n"][:]
    report = dict(scene='2026-07-17', output_time=str(times[at]), valid_km2=float(valid.sum()*.0004),
        false_alarm_km2=float(fa.sum()*.0004), hits_km2=float(hit.sum()*.0004),
        fa_first_wet_by_day=[dict(days=f'{a//24}..{b//24-1}', km2=float(np.count_nonzero((onset>=a)&(onset<b))*.0004)) for a,b in [(0,96),(96,192),(192,264),(264,at+1)]],
        fa_wet_at_day7_km2=float(np.count_nonzero(wse[168,:][fa_cell]-fa_z>.1)*.0004),
        fa_wet_at_day10_km2=float(np.count_nonzero(wse[240,:][fa_cell]-fa_z>.1)*.0004),
        fa_wet_at_start_km2=float(np.count_nonzero(wse[0,:][fa_cell]-fa_z>.1)*.0004),
        fa_wet_at_day1_km2=float(np.count_nonzero(wse[24,:][fa_cell]-fa_z>.1)*.0004),
        fa_x_utm_equal_width_bounds_m=xbounds.tolist(),fa_by_x_third=west,
        cumulative_rain_mm_min=float(np.min(rain)), cumulative_rain_mm_max=float(np.max(rain)),
        outlet_stage_m=float(bc[-1,0]),outlet_flow_m3s=float(flow[-1]),
        outlet_cumulative_m3=out_m3,stored_water_m3=volume_m3,
        all_real_n_004=bool(np.allclose(n[real],.04)))
R.joinpath('excess_water_trace_0717.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
