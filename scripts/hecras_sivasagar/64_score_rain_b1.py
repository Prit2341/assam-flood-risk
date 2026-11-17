"""Paired containing-cell flood extent and four-gauge rise checks for B1."""
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

ROOT = Path(__file__).resolve().parents[2]
R = ROOT/'digital_twin/sivasagar/results'
OUT = R/'rain_b1'
TS = 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series'
GEO = 'Geometry/2D Flow Areas/Sivasagar_Domain'
spec = importlib.util.spec_from_file_location('scorer',Path(__file__).with_name('05_score.py'))
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)
files = dict(control=R/'c600_d4_gridimerg_r30_0705_432h.p01.hdf',
             rain_b1=R/'c600_d4_gridrainb1_r30_0705_432h.p01.hdf')
metadata = {}
with h5py.File(files['control']) as f:
    vertices = f[f'{GEO}/Cells FacePoint Indexes'][:]
    points = f[f'{GEO}/FacePoints Coordinate'][:]
real = np.flatnonzero((vertices >= 0).sum(axis=1) >= 3)
for label,path in files.items():
    with h5py.File(path) as f:
        assert np.array_equal(vertices,f[f'{GEO}/Cells FacePoint Indexes'][:])
        assert np.array_equal(points,f[f'{GEO}/FacePoints Coordinate'][:])
        assert np.allclose(f[f"{GEO}/Cells Center Manning's n"][:][real],.04)
        metadata[label] = dict(mass_error_percent=float(f['Results/Unsteady/Summary/Volume Accounting'].attrs['Error Percent']))
rows = []
for scene in ['2026-07-17','2026-07-22']:
    tag,when = score.SCENES[scene]
    tif = next(score.GFM.glob(f'*{tag}*_flood.tif'))
    pre_tif = next(score.GFM.glob('*pre_2026-07-05*_flood.tif'))
    with rasterio.open(tif) as src:
        obs,shape,tr,crs = src.read(1).ravel(),src.shape,src.transform,src.crs
    with rasterio.open(pre_tif) as src:
        assert (src.shape,src.transform,src.crs) == (shape,tr,crs)
        pre = src.read(1).ravel()
    x,y = Transformer.from_crs(32646,crs,always_xy=True).transform(points[:,0],points[:,1])
    projected = np.c_[x,y]
    polygons = [(Polygon(projected[vertices[c][vertices[c]>=0]]),int(c)) for c in real]
    owner = rasterize(polygons,out_shape=shape,transform=tr,fill=-1,dtype='int32').ravel()
    z = score.raw_z(tif,0)
    valid = (owner >= 0) & (obs != 255) & (pre != 1) & np.isfinite(z)
    for label,path in files.items():
        with h5py.File(path) as f:
            times = score.hdf_times(f)
            i = int(np.argmin(abs(times-pd.Timestamp(when))))
            assert abs(times[i]-pd.Timestamp(when)) <= pd.Timedelta(hours=1)
            wse = f[f'{TS}/2D Flow Areas/Sivasagar_Domain/Water Surface'][i]
            assert np.isfinite(wse[real]).all()
        wet = wse[np.maximum(owner,0)]-z > .1
        h,miss,fa = [int(np.count_nonzero(v)) for v in [valid&(obs==1)&wet,valid&(obs==1)&~wet,valid&(obs==0)&wet]]
        rows.append(dict(scene=scene,arm=label,output_time=str(times[i]),valid_km2=valid.sum()*.0004,
            hits_km2=h*.0004,misses_km2=miss*.0004,false_alarms_km2=fa*.0004,
            CSI=h/max(h+miss+fa,1),area_ratio=(h+fa)/max(h+miss,1)))
extent = pd.DataFrame(rows)
extent.to_csv(OUT/'extent_scores.csv',index=False)
obs = pd.read_csv(R/'cwc_four_gauges_july2026.csv')
obs['utc'] = pd.to_datetime(obs.utc)
mapping = pd.read_csv(R/'gauge_mapping_audit.csv')
gauge_rows = []
start,end = pd.Timestamp('2026-07-18 00:00'),pd.Timestamp('2026-07-20 03:30')
names = {'018-UBDDIB':'Sivasagar','017-UBDDIB':'Bihubar','016-UBDDIB':'Nanglamoraghat','015-UBDDIB':'Desangpani'}
for code,name in names.items():
    o = obs[(obs.station_code == code) & (obs.datatype_code == 'HHS')].sort_values('utc')
    ox = pd.DatetimeIndex(o.utc).as_unit('ns').astype('int64').to_numpy()
    observed = np.interp(end.value,ox,o.value)-np.interp(start.value,ox,o.value)
    cid = int(mapping[(mapping.station == name) & (mapping.run == 'IMERG_grid')].containing_cell.iloc[0])
    for label,path in files.items():
        with h5py.File(path) as f:
            times = score.hdf_times(f).as_unit('ns')
            tx = times.astype('int64').to_numpy()
            wse = f[f'{TS}/2D Flow Areas/Sivasagar_Domain/Water Surface'][:,cid]
        assert tx[0] <= start.value < end.value <= tx[-1]
        rise = float(np.interp(end.value,tx,wse)-np.interp(start.value,tx,wse))
        gauge_rows.append(dict(station=name,arm=label,observed_rise_m=float(observed),
                               model_rise_m=rise,absolute_rise_error_m=abs(rise-observed)))
gauges = pd.DataFrame(gauge_rows)
gauges.to_csv(OUT/'stage_rise_scores.csv',index=False)
e = extent.pivot(index='scene',columns='arm',values='CSI')
delta = e.rain_b1-e.control
errors = gauges.groupby('arm').absolute_rise_error_m.mean()
checks = dict(mean_CSI_gain_001=bool(delta.mean() >= .01),
    neither_scene_degrades_over001=bool((delta >= -.01).all()),
    mean_stage_rise_error_decreases=bool(errors.rain_b1 < errors.control),
    numerical_mass_error_below01pct=bool(abs(metadata['rain_b1']['mass_error_percent']) < .1))
report = dict(metadata=metadata,CSI_gain_by_scene=delta.to_dict(),
    mean_CSI_gain=float(delta.mean()),mean_absolute_rise_error_m=errors.to_dict(),
    checks=checks,pilot_pass=all(checks.values()),G1_pass=False,
    note='Exploratory calibration-period pilot; independent event/year still required. Stage-rise test avoids a constant datum offset but station-to-channel placement remains approximate.')
(OUT/'hydraulic_verdict.json').write_text(json.dumps(report,indent=2))
print(extent.to_string(index=False))
print(gauges.to_string(index=False))
print(json.dumps(report,indent=2))
