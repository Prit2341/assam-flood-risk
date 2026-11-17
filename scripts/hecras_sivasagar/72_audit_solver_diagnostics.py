"""Audit saved solver diagnostics; hourly samples are not all solver steps.

Reads completed historical results only. Status meanings come from the HDF
dataset's own Description attribute. Does not change any running project.
"""
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'digital_twin/sivasagar/results'
OUT = RES / 'routing_audit'
OUT.mkdir(exist_ok=True)
TS = 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series'
DOMAIN = '2D Flow Areas/Sivasagar_Domain'
FILES = {
    'imerg_control': 'c600_d4_gridimerg_r30_0705_432h.p01.hdf',
    'rain_b1': 'c600_d4_gridrainb1_r30_0705_432h.p01.hdf',
    'raw_frozen': 'c600_d4_r30_0705_432h_raw.p01.hdf',
}
transform = Transformer.from_crs(4326, 32646, always_xy=True)
GAUGES = {
    'Sivasagar': (94 + 34/60 + 44/3600, 26 + 58/60 + 54/3600),
    'Bihubar': (94.800556, 26.853333),
    'Nanglamoraghat': (94.776667, 26.985),
    'Desangpani': (94.912222, 27.046667),
}

rows, summaries = [], {}
for arm, filename in FILES.items():
    path = RES / filename
    if not path.exists():
        summaries[arm] = {'missing': str(path)}
        continue
    with h5py.File(path) as f:
        base = f[TS]
        times = pd.DatetimeIndex(pd.to_datetime([
            x.decode().strip() for x in base['Time Date Stamp'][:]
        ], format='%d%b%Y %H:%M:%S'))
        c = base[DOMAIN + '/Computations']
        status = c['Outer Status'][:].ravel()
        correction = c['Outer Max Water Surface Correction'][:].ravel()
        locations = c['Outer Max Water Surface Correction Cell'][:].ravel().astype(int)
        centers = f['Geometry/' + DOMAIN + '/Cells Center Coordinate'][:]
        params = f['Plan Data/Plan Parameters'].attrs
        equations = np.atleast_1d(params['2D Equation Set'])
        summary = {
            'file': str(path),
            'equation_set': [x.decode() if isinstance(x, bytes) else str(x) for x in equations],
            'saved_samples': len(times),
            'start': str(times[0]), 'end': str(times[-1]),
            'status_description': str(c['Outer Status'].attrs['Description']),
            'status_counts_excluding_initial': {
                str(int(k)): int(v) for k, v in zip(*np.unique(status[1:], return_counts=True))
            },
            'sampled_nonconvergence_fraction': float(np.mean(status[1:] < 0)),
            'max_saved_wse_correction_m': float(np.max(correction[1:])),
            'max_saved_courant': float(np.max(c['Max Courant'][1:])),
            'mass_error_percent': float(f['Results/Unsteady/Summary/Volume Accounting'].attrs['Error Percent']),
            'scope': 'Diagnostics sampled at saved output times; counts are not all internal time steps. Location is the worst correction cell, not every unconverged cell.',
        }
        surge = (times >= '2026-07-18') & (times <= '2026-07-20 04:00')
        summary['surge_sample_count'] = int(surge.sum())
        summary['surge_negative_status_count'] = int(np.sum((status < 0) & surge))
        for i in np.flatnonzero(status < 0):
            cid = locations[i]
            row = dict(arm=arm, utc=str(times[i]), status=int(status[i]),
                       correction_m=float(correction[i]), worst_cell=int(cid),
                       iterations=int(c['Outer Iteration Number'][i, 0]))
            if 0 <= cid < len(centers):
                for name, (lon, lat) in GAUGES.items():
                    xy = np.array(transform.transform(lon, lat))
                    row[name + '_distance_m'] = float(np.linalg.norm(centers[cid] - xy))
            rows.append(row)
        summaries[arm] = summary

pd.DataFrame(rows).to_csv(OUT / 'solver_sampled_nonconvergence.csv', index=False)
(OUT / 'solver_diagnostics_summary.json').write_text(json.dumps(summaries, indent=2))
print(json.dumps(summaries, indent=2))
