"""Read-only gauge/cell sensitivity audit of existing July teacher runs.

Nearby cells are a sensitivity envelope, not alternative calibration targets.
No datum correction, parameter fitting or model modification is performed.
"""
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer
from shapely.geometry import Point, Polygon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'digital_twin/sivasagar/results'
TS = 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series'
AREA = 'Sivasagar_Domain'
STATIONS = [
    ('Sivasagar', '018-UBDDIB', 26+58/60+54/3600, 94+34/60+44/3600),
    ('Bihubar', '017-UBDDIB', 26.853333, 94.800556),
    ('Nanglamoraghat', '016-UBDDIB', 26.985, 94.776667),
    ('Desangpani', '015-UBDDIB', 27.046667, 94.912222),
]
RUNS = {
    'uniform_raw': 'c600_d4_r30_0705_432h_raw.p01.hdf',
    'ERA5_grid': 'c600_d4_gridera5full_r30_0705_432h.p01.hdf',
    'IMERG_grid': 'c600_d4_gridimerg_r30_0705_432h.p01.hdf',
    'IMERG_wide_channel': 'c600_d4_gridw4imerg_r30_0705_432h.p01.hdf',
}
obs = pq.read_table(ROOT/'docs/research/2026-11-16/cwc_hourly_latest.parquet',
    filters=[('station_code', 'in', [s[1] for s in STATIONS]),
             ('datetime', '>=', pd.Timestamp('2026-07-05')),
             ('datetime', '<', pd.Timestamp('2026-07-24')),
             ('datatype_code', '=', 'HHS')]).to_pandas()
obs['utc'] = obs.datetime-pd.Timedelta(hours=5, minutes=30)
obs.to_csv(RES/'cwc_four_gauges_july2026.csv', index=False)
transform = Transformer.from_crs(4326, 32646, always_xy=True)
start = pd.Timestamp('2026-07-18 00:00')
end = pd.Timestamp('2026-07-20 03:30')
rows = []
fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
for si, (name, code, lat, lon) in enumerate(STATIONS):
    o = obs[(obs.station_code == code) & obs.value.between(70, 140)].sort_values('utc')
    assert not o.utc.duplicated().any(), f'duplicate gauge timestamps: {code}'
    ox = pd.DatetimeIndex(o.utc).as_unit('ns').astype('int64').to_numpy()
    observed_rise = float(np.interp(end.value, ox, o.value)-np.interp(start.value, ox, o.value))
    ax = axes[si]
    ax.plot(o.utc, o.value, color='black', linewidth=2, label='CWC-derived archive')
    for label, filename in RUNS.items():
        with h5py.File(RES/filename) as f:
            g = f[f'Geometry/2D Flow Areas/{AREA}']
            xy = g['Cells Center Coordinate'][:]
            indexes = g['Cells FacePoint Indexes'][:]
            real = np.flatnonzero((indexes >= 0).sum(axis=1) >= 3)
            vertices = g['FacePoints Coordinate'][:]
            x, y = transform.transform(lon, lat)
            distance = np.linalg.norm(xy-[x, y], axis=1)
            nearest = int(real[np.argmin(distance[real])])
            nearby = real[distance[real] <= 1000]
            owners = [int(cid) for cid in nearby if Polygon(vertices[indexes[cid][indexes[cid]>=0]]).covers(Point(x, y))]
            assert len(owners) == 1, (name, label, owners)
            owner = owners[0]
            times = pd.to_datetime([v.decode() for v in f[f'{TS}/Time Date Stamp'][:]], format='%d%b%Y %H:%M:%S')
            tx = times.as_unit('ns').astype('int64').to_numpy()
            assert tx[0] <= start.value < end.value <= tx[-1]
            assert ox[0] <= start.value < end.value <= ox[-1]
            # h5py fancy indexing requires sorted indexes.
            wse = f[f'{TS}/2D Flow Areas/{AREA}/Water Surface'][:, nearby]
            a = np.array([np.interp(start.value, tx, wse[:, j]) for j in range(len(nearby))])
            b = np.array([np.interp(end.value, tx, wse[:, j]) for j in range(len(nearby))])
            j = int(np.flatnonzero(nearby == owner)[0])
            rows.append(dict(station=name, code=code, run=label, containing_cell=owner,
                nearest_cell=nearest, nearest_distance_m=float(distance[nearest]),
                nearby_cells=int(len(nearby)), observed_rise_m=observed_rise,
                containing_rise_m=float(b[j]-a[j]), containing_end_stage_m=float(b[j]),
                nearby_rise_min_m=float((b-a).min()), nearby_rise_max_m=float((b-a).max()),
                nearby_end_stage_min_m=float(b.min()), nearby_end_stage_max_m=float(b.max())))
            ax.plot(times, wse[:, j], label=label)
            if label == 'uniform_raw':
                ax.fill_between(times, wse.min(axis=1), wse.max(axis=1), alpha=.12,
                    label='Uniform: all cell centres within 1 km')
    ax.set_title(name)
    ax.set_ylabel('Water level (m)')
    ax.grid(alpha=.25)
axes[0].legend(fontsize=8, ncol=2)
axes[-1].set_xlim(pd.Timestamp('2026-07-15'), pd.Timestamp('2026-07-23'))
fig.suptitle('Gauge mapping audit: containing cells, no datum adjustment\nJuly observations cross-checked with CWC; vertical datum remains unresolved', fontsize=12)
fig.autofmt_xdate()
fig.tight_layout(rect=(0, 0, 1, .96))
fig.savefig(RES/'gauge_mapping_audit.png', dpi=160)
df = pd.DataFrame(rows)
df.to_csv(RES/'gauge_mapping_audit.csv', index=False)
print(df.round(3).to_string(index=False))
(RES/'gauge_mapping_audit_metadata.json').write_text(json.dumps(dict(
    rise_start_utc=str(start), rise_end_utc=str(end), radius_m=1000,
    limitations='Only Sivasagar coordinates cross-checked with official CWC metadata; other coordinates from archive. Nearby cells are sensitivity bounds, not verified river cross sections. Absolute elevation datum unresolved. These are existing calibration-period runs, not independent validation.'), indent=2))
