"""Descriptive timing/amplitude audit on saved July runs, no fitting.

Fixed gauge coordinates from script59; compare July15-23 only to exclude
dry-start transients. Largest 24-hour rise permits any timing within that
window. It is exploratory and cannot replace the registered C3 criteria.
"""
from pathlib import Path
import importlib.util
import h5py
import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT/'digital_twin/sivasagar/results'
OUT = RES/'routing_audit'
TS = 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series'
GEO = 'Geometry/2D Flow Areas/Sivasagar_Domain'
spec = importlib.util.spec_from_file_location('ownership', Path(__file__).with_name('52_audit_map_ownership.py'))
ownership = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ownership)
STATIONS = [
    ('Sivasagar', '018-UBDDIB', 26+58/60+54/3600, 94+34/60+44/3600),
    ('Bihubar', '017-UBDDIB', 26.853333, 94.800556),
    ('Nanglamoraghat', '016-UBDDIB', 26.985, 94.776667),
    ('Desangpani', '015-UBDDIB', 27.046667, 94.912222),
]
FILES = dict(
    raw_frozen='c600_d4_r30_0705_432h_raw.p01.hdf',
    imerg='c600_d4_gridimerg_r30_0705_432h.p01.hdf',
    era5='c600_d4_gridera5full_r30_0705_432h.p01.hdf',
    wide_imerg='c600_d4_gridw4imerg_r30_0705_432h.p01.hdf',
    rain_b1='c600_d4_gridrainb1_r30_0705_432h.p01.hdf',
)
obs = pd.read_csv(RES/'cwc_four_gauges_july2026.csv')
obs['utc'] = pd.to_datetime(obs.utc)
grid = pd.date_range('2026-07-15', '2026-07-23', freq='h')
gx = grid.as_unit('ns').astype('int64').to_numpy()
transform = Transformer.from_crs(4326, 32646, always_xy=True)
xy = np.array([transform.transform(lon, lat) for _, _, lat, lon in STATIONS])
rows = []

def describe(values):
    assert len(values) == len(grid)
    rise = values[24:] - values[:-24]
    j = int(np.nanargmax(rise)) + 24
    peak = int(np.nanargmax(values))
    return dict(max_24h_rise_m=float(rise[j-24]), max_rise_end_utc=str(grid[j]),
                range_m=float(np.nanmax(values)-np.nanmin(values)), peak_utc=str(grid[peak]))

for name, code, _, _ in STATIONS:
    o = obs[(obs.station_code == code)&(obs.datatype_code == 'HHS')].sort_values('utc')
    assert not o.utc.duplicated().any()
    ox = pd.DatetimeIndex(o.utc).as_unit('ns').astype('int64').to_numpy()
    assert ox[0] <= gx[0] and ox[-1] >= gx[-1]
    y = np.interp(gx, ox, o.value)
    right = np.searchsorted(ox, gx, side='right').clip(1, len(ox)-1)
    # Do not bridge long observation gaps.
    y[(ox[right]-ox[right-1]) > pd.Timedelta(hours=2).value] = np.nan
    rows.append(dict(station=name, arm='CWC', valid_samples=int(np.isfinite(y).sum()), **describe(y)))

for arm, filename in FILES.items():
    with h5py.File(RES/filename) as f:
        owner, _ = ownership.mesh_owners(f[GEO], xy[:, 0], xy[:, 1])
        assert (owner >= 0).all()
        times = pd.DatetimeIndex(pd.to_datetime([x.decode().strip() for x in f[TS+'/Time Date Stamp'][:]], format='%d%b%Y %H:%M:%S'))
        tx = times.as_unit('ns').astype('int64').to_numpy()
        assert tx[0] <= gx[0] and tx[-1] >= gx[-1]
        centers = f[GEO+'/Cells Center Coordinate'][:]
        real = np.flatnonzero((f[GEO+'/Cells FacePoint Indexes'][:] >= 0).sum(axis=1) >= 3)
        for si, (name, _, _, _) in enumerate(STATIONS):
            ids = real[np.linalg.norm(centers[real]-xy[si], axis=1) <= 1000]
            assert owner[si] in ids
            levels = f[TS+'/2D Flow Areas/Sivasagar_Domain/Water Surface'][:, ids]
            y = np.interp(gx, tx, levels[:, np.flatnonzero(ids == owner[si])[0]])
            envelope = [describe(np.interp(gx, tx, levels[:, j]))['max_24h_rise_m'] for j in range(len(ids))]
            rows.append(dict(station=name, arm=arm, valid_samples=len(y), containing_cell=int(owner[si]),
                             nearby_max_24h_rise_m=float(max(envelope)), **describe(y)))

df = pd.DataFrame(rows)
df.to_csv(OUT/'gauge_rise_timing.csv', index=False)
print(df.to_string(index=False))
