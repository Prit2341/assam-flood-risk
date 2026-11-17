"""Inventory saved result fingerprints, avoiding filename-only assumptions.

Terrain classification is only an exact cell-minimum signature, not proof of
identical terrain everywhere. Rain comparisons use cumulative depth at the
same absolute UTC instant and identical cell centers.
"""
from pathlib import Path
import hashlib
import json
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT/'digital_twin/sivasagar/results'
OUT = RES/'routing_audit'
GEO = 'Geometry/2D Flow Areas/Sivasagar_Domain'
TS = 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series'
RAIN = TS+'/2D Flow Areas/Sivasagar_Domain/Cell Cumulative Precipitation Depth'
WHEN = pd.Timestamp('2026-07-18 00:00')

def fingerprint(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def extract(path):
    with h5py.File(path) as f:
        g = f[GEO]
        real = np.flatnonzero((g['Cells FacePoint Indexes'][:] >= 0).sum(axis=1) >= 3)
        times = pd.DatetimeIndex(pd.to_datetime([x.decode().strip() for x in f[TS+'/Time Date Stamp'][:]], format='%d%b%Y %H:%M:%S'))
        if WHEN not in times:
            return None
        rain = f[RAIN][times.get_loc(WHEN), real].astype(float)
        return dict(real_cells=len(real), start=str(times[0]), end=str(times[-1]),
                    cell_centers_sha256=fingerprint(g['Cells Center Coordinate'][:][real]),
                    cell_minimum_sha256=fingerprint(g['Cells Minimum Elevation'][:][real]),
                    rain=rain, rain_depth_mean_mm=float(rain.mean()),
                    rain_depth_spatial_cv=float(rain.std()/rain.mean()) if rain.mean() else None)

references = {
    'raw_uniform': extract(RES/'c600_d4_r30_0705_432h_raw.p01.hdf'),
    'filled_imerg': extract(RES/'c600_d4_gridimerg_r30_0705_432h.p01.hdf'),
    'filled_era5': extract(RES/'c600_d4_gridera5full_r30_0705_432h.p01.hdf'),
}
rows, skipped = [], []
for path in sorted(RES.glob('*.hdf')):
    try:
        row = extract(path)
        if row is None:
            skipped.append(dict(file=path.name, reason='No July18 checkpoint'))
            continue
        rain = row.pop('rain')
        row['file'] = path.name
        for name, ref in references.items():
            same_mesh = row['cell_centers_sha256'] == ref['cell_centers_sha256']
            row[name+'_same_cell_minima'] = same_mesh and row['cell_minimum_sha256'] == ref['cell_minimum_sha256']
            if same_mesh and rain.mean() > 0 and ref['rain'].mean() > 0:
                row[name+'_normalized_rain_max_abs_diff'] = float(np.max(abs(rain/rain.mean()-ref['rain']/ref['rain'].mean())))
        rows.append(row)
    except (KeyError, OSError, ValueError) as error:
        skipped.append(dict(file=path.name, reason=str(error)))

df = pd.DataFrame(rows)
df.to_csv(OUT/'saved_input_inventory.csv', index=False)
matches = df[(df.raw_uniform_same_cell_minima) & (df.filled_imerg_normalized_rain_max_abs_diff < 1e-4)]
summary = dict(checkpoint_utc=str(WHEN), scanned_result_files=len(rows)+len(skipped),
               usable_files=len(rows), skipped=skipped,
               raw_minima_and_imerg_cumulative_pattern_matches=matches.file.tolist(),
               limitations='Scans only results/*.hdf, not other directories or unpublished runs. Cell-minimum signature does not describe full terrain. Cumulative rain pattern at one checkpoint does not prove identical rainfall time series. Empty match list is limited to this inventory.')
(OUT/'saved_input_inventory_summary.json').write_text(json.dumps(summary, indent=2))
print(df[['file','raw_uniform_same_cell_minima','filled_imerg_normalized_rain_max_abs_diff','rain_depth_spatial_cv']].to_string(index=False))
print(json.dumps(summary, indent=2))
