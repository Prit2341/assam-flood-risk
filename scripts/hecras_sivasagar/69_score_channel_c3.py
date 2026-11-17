"""Registered paired extent and gauge-rise checks for the changed C3 mesh."""
from pathlib import Path
import importlib.util
import json
import sys
import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from pyproj import Transformer
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[2]
R = ROOT/'digital_twin/sivasagar/results'
label = sys.argv[1] if len(sys.argv)>1 else 'channelc3'
assert label in ['channelc3','channelc3fast']
OUT = R/('channel_c3' if label == 'channelc3' else 'channel_c3fast')
TS = 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series'
GEO = 'Geometry/2D Flow Areas/Sivasagar_Domain'
def module(filename,name):
    s = importlib.util.spec_from_file_location(name,Path(__file__).with_name(filename))
    m = importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
score = module('05_score.py','score')
ownership = module('52_audit_map_ownership.py','ownership')
files = dict(control=R/'c600_d4_gridimerg_r30_0705_432h.p01.hdf',
             channel_c3=R/f'c600_d4_grid{label}_r30_0705_432h.p01.hdf')
metadata = {}
for arm,path in files.items():
    with h5py.File(path) as f:
        g = f[GEO]
        real = np.flatnonzero((g['Cells FacePoint Indexes'][:] >= 0).sum(axis=1) >= 3)
        assert np.allclose(g["Cells Center Manning's n"][:][real],.04)
        surfaces = f[f'{TS}/2D Flow Areas/Sivasagar_Domain/Water Surface']
        for i in range(0,surfaces.shape[0],64):
            assert np.isfinite(surfaces[i:i+64,real]).all(), f'Nonfinite real-cell output in {arm}'
        area = g['Cells Surface Area'][:][real].astype(float)
        rain = f[f'{TS}/2D Flow Areas/Sivasagar_Domain/Cell Cumulative Precipitation Depth'][-1,real].astype(float)
        metadata[arm] = dict(real_cells=len(real),domain_real_cell_area_km2=float(area.sum()/1e6),
            end_effective_rain_Mm3=float(np.sum(rain*area)/1e9),
            mass_error_percent=float(f['Results/Unsteady/Summary/Volume Accounting'].attrs['Error Percent']))
rows = []
for scene in ['2026-07-17','2026-07-22']:
    tag,when = score.SCENES[scene]
    tif = next(score.GFM.glob(f'*{tag}*_flood.tif'))
    with rasterio.open(tif) as src:
        obs,shape,tr,crs = src.read(1).ravel(),src.shape,src.transform,src.crs
    with rasterio.open(next(score.GFM.glob('*pre_2026-07-05*_flood.tif'))) as src:
        assert (src.shape,src.transform,src.crs) == (shape,tr,crs)
        pre = src.read(1).ravel()
    z = score.raw_z(tif,0)
    valid = (obs != 255) & (pre != 1) & np.isfinite(z)
    unpaired_valid_count = int(valid.sum())
    outputs = {}
    for arm,path in files.items():
        with h5py.File(path) as f:
            g = f[GEO]
            ids = g['Cells FacePoint Indexes'][:]
            real = np.flatnonzero((ids >= 0).sum(axis=1) >= 3)
            xy = g['FacePoints Coordinate'][:]
            x,y = Transformer.from_crs(32646,crs,always_xy=True).transform(xy[:,0],xy[:,1])
            xy = np.c_[x,y]
            polygons = [(Polygon(xy[ids[c][ids[c]>=0]]),int(c)) for c in real]
            owner = rasterize(polygons,out_shape=shape,transform=tr,fill=-1,dtype='int32').ravel()
            times = score.hdf_times(f)
            i = int(np.argmin(abs(times-pd.Timestamp(when))))
            assert abs(times[i]-pd.Timestamp(when)) <= pd.Timedelta(hours=1)
            wse = f[f'{TS}/2D Flow Areas/Sivasagar_Domain/Water Surface'][i]
            assert np.isfinite(wse[real]).all()
            valid &= owner >= 0
            outputs[arm] = wse[np.maximum(owner,0)]-z > .1
    for arm,wet in outputs.items():
        h,miss,fa = [int(np.count_nonzero(v)) for v in [valid&(obs==1)&wet,valid&(obs==1)&~wet,valid&(obs==0)&wet]]
        rows.append(dict(scene=scene,arm=arm,valid_km2=float(valid.sum()*.0004),hits_km2=h*.0004,
            fraction_valid_pixels_covered_by_both_meshes=float(valid.sum()/max(unpaired_valid_count,1)),
            misses_km2=miss*.0004,false_alarms_km2=fa*.0004,CSI=h/max(h+miss+fa,1),area_ratio=(h+fa)/max(h+miss,1)))
extent = pd.DataFrame(rows)
extent.to_csv(OUT/'extent_scores.csv',index=False)
obs = pd.read_csv(R/'cwc_four_gauges_july2026.csv')
obs['utc'] = pd.to_datetime(obs.utc)
stations = [('Sivasagar','018-UBDDIB',26+58/60+54/3600,94+34/60+44/3600),
            ('Bihubar','017-UBDDIB',26.853333,94.800556),
            ('Nanglamoraghat','016-UBDDIB',26.985,94.776667),
            ('Desangpani','015-UBDDIB',27.046667,94.912222)]
start,end = pd.Timestamp('2026-07-18 00:00'),pd.Timestamp('2026-07-20 03:30')
transform = Transformer.from_crs(4326,32646,always_xy=True)
xy = np.array([transform.transform(lon,lat) for _,_,lat,lon in stations])
gauge_rows,neighborhoods = [],[]
for arm,path in files.items():
    with h5py.File(path) as f:
        owners,_ = ownership.mesh_owners(f[GEO],xy[:,0],xy[:,1])
        assert (owners >= 0).all()
        times = score.hdf_times(f).as_unit('ns')
        tx = times.astype('int64').to_numpy()
        assert tx[0] <= start.value < end.value <= tx[-1]
        centers = f[f'{GEO}/Cells Center Coordinate'][:]
        real = np.flatnonzero((f[f'{GEO}/Cells FacePoint Indexes'][:] >= 0).sum(axis=1) >= 3)
        for station_i,(cid,(name,code,_,_)) in enumerate(zip(owners,stations)):
            o = obs[(obs.station_code == code)&(obs.datatype_code == 'HHS')].sort_values('utc')
            ox = pd.DatetimeIndex(o.utc).as_unit('ns').astype('int64').to_numpy()
            observed = float(np.interp(end.value,ox,o.value)-np.interp(start.value,ox,o.value))
            wse = f[f'{TS}/2D Flow Areas/Sivasagar_Domain/Water Surface'][:,int(cid)]
            rise = float(np.interp(end.value,tx,wse)-np.interp(start.value,tx,wse))
            gauge_rows.append(dict(station=name,arm=arm,containing_cell=int(cid),observed_rise_m=observed,
                model_rise_m=rise,absolute_rise_error_m=abs(rise-observed)))
            distance = np.linalg.norm(centers-xy[station_i],axis=1)
            nearby = real[distance[real] <= 1000]
            values = f[f'{TS}/2D Flow Areas/Sivasagar_Domain/Water Surface'][:,nearby]
            rises = np.array([np.interp(end.value,tx,values[:,j])-np.interp(start.value,tx,values[:,j]) for j in range(len(nearby))])
            neighborhoods.append(dict(station=name,arm=arm,cells_within1km=len(nearby),
                rise_min_m=float(rises.min()),rise_max_m=float(rises.max()),observed_rise_m=observed))
gauges = pd.DataFrame(gauge_rows)
gauges.to_csv(OUT/'stage_rise_scores.csv',index=False)
pd.DataFrame(neighborhoods).to_csv(OUT/'stage_rise_neighborhoods.csv',index=False)
p = extent.pivot(index='scene',columns='arm',values='CSI')
gain = p.channel_c3-p.control
errors = gauges.groupby('arm').absolute_rise_error_m.mean()
checks = dict(mean_CSI_gain_001=bool(gain.mean() >= .01),neither_scene_degrades_over001=bool((gain >= -.01).all()),
    mean_stage_rise_error_reduced10pct=bool(errors.channel_c3 <= .9*errors.control),
    numerical_mass_error_below01pct=bool(abs(metadata['channel_c3']['mass_error_percent']) < .1))
report = dict(metadata=metadata,CSI_gain_by_scene=gain.to_dict(),mean_CSI_gain=float(gain.mean()),
    mean_absolute_rise_error_m=errors.to_dict(),checks=checks,pilot_pass=all(checks.values()),G1_pass=False,
    note='Exploratory mesh test with identical forcing/terrain; independent event required. Gauge coordinates kept identical to previous diagnostics; exact bank alignment/vertical datum remain uncertain. Nearby1km rise envelopes are secondary mapping sensitivity only and do not change registered point-based gates.')
(OUT/'hydraulic_verdict.json').write_text(json.dumps(report,indent=2))
print(extent.to_string(index=False));print(gauges.to_string(index=False));print(json.dumps(report,indent=2))
