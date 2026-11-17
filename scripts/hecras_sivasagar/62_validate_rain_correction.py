"""B1: fixed daily IDW2 residual correction, leave-one-station-out validation.

Daily windows end 03 UTC (08:30 IST). This uses contemporaneous rain
observations for hindcasting; it is not a rainfall forecasting experiment.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import rasterio
import pyarrow.parquet as pq
from pyproj import Transformer
from shapely.geometry import MultiPoint, Point

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT/'docs/research/2026-11-16'
OUT = ROOT/'digital_twin/sivasagar/results/rain_b1'
OUT.mkdir(exist_ok=True)
RAIN = ROOT/'digital_twin/sivasagar/data/imerg'
NAMES = ['Dibrugarh','Margherita','Naharkatia','Chenimari','Dillighat','Bihubar','Sivasagar']
codes = ['010-UBDDIB','011-UBDDIB','012-UBDDIB','013-UBDDIB','014-UBDDIB','017-UBDDIB','018-UBDDIB']
inventory = pd.read_csv(SRC/'cwc_archive_stations.csv').set_index('station_code')
meta = json.loads((RAIN/'imerg_grid_070500_072300.json').read_text())
times = pd.DatetimeIndex(pd.to_datetime(meta['stamps'])).as_unit('ns')
with rasterio.open(RAIN/meta['parts'][0]) as r:
    rc = [r.index(inventory.loc[c,'lon'],inventory.loc[c,'lat']) for c in codes]
arrays = []
for part in meta['parts']:
    with rasterio.open(RAIN/part) as r:
        a = r.read().astype(float)
        assert np.isfinite(a).all() and (a >= 0).all()
        arrays.append(a[:,[p[0] for p in rc],[p[1] for p in rc]])
sat = np.concatenate(arrays)
assert len(sat) == len(times) == 432
xy = np.array([Transformer.from_crs(4326,32646,always_xy=True).transform(
    inventory.loc[c,'lon'],inventory.loc[c,'lat']) for c in codes])
observations = []
verification = []
for name, code in zip(NAMES,codes):
    p = SRC/f'cwc_primary_{name.lower()}_hydrology.json'
    d = pd.DataFrame(json.loads(p.read_text()))
    d = d[d.datatype_code == 'MPS'].copy()
    d['datetime'] = pd.to_datetime(d.datetime)
    d['value'] = d.validated_value.combine_first(d.raw_value).astype(float)
    d = d[(d.datetime >= '2026-07-01') & (d.datetime < '2026-08-01')]
    arc = pq.read_table(SRC/'cwc_hourly_latest.parquet', filters=[
        ('station_code','=',code),('datatype_code','=','MPS'),
        ('datetime','>=',pd.Timestamp('2026-07-01')),
        ('datetime','<',pd.Timestamp('2026-08-01'))]).to_pandas()
    paired = d.merge(arc,on='datetime',suffixes=('_primary','_archive'))
    assert not d.datetime.duplicated().any()
    assert len(paired) == len(d) and np.allclose(paired.value_primary,paired.value_archive)
    verification.append(dict(station=name,primary_rows=len(d),exact_archive_match=True,
                             sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    d['utc'] = d.datetime-pd.Timedelta(hours=5,minutes=30)
    observations.append(d.set_index('utc').value)

days = pd.date_range('2026-07-05 03:00','2026-07-21 03:00',freq='D')
daily = []
for start in days:
    end = start+pd.Timedelta(days=1)
    sel = (times > start) & (times <= end)
    assert sel.sum() == 24
    for j,(name,code) in enumerate(zip(NAMES,codes)):
        values = observations[j].loc[(observations[j].index > start) & (observations[j].index <= end)]
        if len(values) != 8 or not values.between(0,1000).all():
            continue
        daily.append(dict(station=name,code=code,day_end_utc=str(end),
                          obs_mm=float(values.sum()),sat_mm=float(sat[sel,j].sum()),
                          x=xy[j,0],y=xy[j,1],inside_domain=bool(code in ['014-UBDDIB','017-UBDDIB','018-UBDDIB'])))
daily = pd.DataFrame(daily)
daily.to_csv(OUT/'daily_gauge_residuals.csv',index=False)
predictions = []
for day,g in daily.groupby('day_end_utc'):
    for _,target in g.iterrows():
        train = g[g.station != target.station]
        assert len(train) >= 3
        txy = train[['x','y']].to_numpy()
        distance = np.linalg.norm(txy-target[['x','y']].to_numpy(dtype=float),axis=1)
        weights = 1/np.maximum(distance,1)**2
        corrected = max(0,float(target.sat_mm+np.average(train.obs_mm-train.sat_mm,weights=weights)))
        predictions.append(dict(target,corrected_mm=corrected,n_train=len(train),
            inside_training_hull=MultiPoint(txy).convex_hull.covers(Point(target.x,target.y))))
predictions = pd.DataFrame(predictions)
predictions.to_csv(OUT/'withheld_station_predictions.csv',index=False)
def metrics(d):
    if not len(d):return dict(n=0)
    def m(col):
        e = (d[col]-d.obs_mm).to_numpy()
        return dict(rmse_mm=float(np.sqrt(np.mean(e**2))),mae_mm=float(np.mean(abs(e))),bias_mm=float(e.mean()))
    return dict(n=len(d),raw=m('sat_mm'),corrected=m('corrected_mm'))
pooled = metrics(predictions)
wet = metrics(predictions[predictions.obs_mm >= 20])
inside = metrics(predictions[predictions.inside_domain])
checks = dict(pooled_rmse_gain_10pct=pooled['corrected']['rmse_mm'] <= .9*pooled['raw']['rmse_mm'],
              pooled_mae_no_worse=pooled['corrected']['mae_mm'] <= pooled['raw']['mae_mm'],
              wet_rmse_no_worse=wet['corrected']['rmse_mm'] <= wet['raw']['rmse_mm'],
              domain_gauges_rmse_no_worse=inside['corrected']['rmse_mm'] <= inside['raw']['rmse_mm'])
report = dict(method='Daily additive residual IDW power2; complete 03UTC-ending days; whole-station withholding',
    literature='Biswas and Hossain2017, Journal of Hydroinformatics, https://saswe.net/publications/SASWMS.pdf',
    pooled=pooled,wet_days=wet,domain_gauges=inside,
    within_training_hull=metrics(predictions[predictions.inside_training_hull]),
    by_station={name:metrics(g) for name,g in predictions.groupby('station')},
    primary_verification=verification,checks=checks,hydraulic_pilot_allowed=bool(all(checks.values())),
    limitations='Inspected-event exploratory hindcast, not independent event validation. MPS interval-ending interpretation follows archive; official accumulation convention still unresolved. Plain gauges do not validate hills. Negative corrected amounts clipped to0. No stage or flood-map fitting.')
(OUT/'validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
