"""Compare stored model precipitation and CWC MPS point observations.

MPS is interpreted as interval rainfall following the archive data dictionary.
Point readings do not validate basin-average rainfall or a runoff coefficient.
"""
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT/'digital_twin/sivasagar/results'
SRC = ROOT/'docs/research/2026-11-16'
TS = 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series'
AUDIT = pd.read_csv(RES/'gauge_mapping_audit.csv')
FILES = {'uniform_raw':'c600_d4_r30_0705_432h_raw.p01.hdf',
         'ERA5_grid':'c600_d4_gridera5full_r30_0705_432h.p01.hdf',
         'IMERG_grid':'c600_d4_gridimerg_r30_0705_432h.p01.hdf'}
rows = []
for station in ['Sivasagar','Bihubar']:
    d = pd.DataFrame(json.loads((SRC/f'cwc_primary_{station.lower()}_hydrology.json').read_text()))
    d = d[d.datatype_code == 'MPS'].copy()
    d['utc'] = pd.to_datetime(d.datetime)-pd.Timedelta(hours=5, minutes=30)
    d['value'] = d.validated_value.combine_first(d.raw_value).astype(float)
    assert not d.utc.duplicated().any()
    for start, end in [('2026-07-18 00:00','2026-07-20 00:00'),
                       ('2026-07-05 00:00','2026-07-18 00:00')]:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        rain = d[(d.utc > start) & (d.utc <= end)]
        expected = int((end-start)/pd.Timedelta(hours=3))
        assert len(rain) == expected, (station, start, len(rain), expected)
        assert rain.value.between(0, 1000).all()
        for label, filename in FILES.items():
            cid = int(AUDIT[(AUDIT.station == station) & (AUDIT.run == label)].containing_cell.iloc[0])
            with h5py.File(RES/filename) as f:
                times = pd.to_datetime([v.decode() for v in f[f'{TS}/Time Date Stamp'][:]], format='%d%b%Y %H:%M:%S')
                i, j = times.get_loc(start), times.get_loc(end)
                applied = f[f'{TS}/2D Flow Areas/Sivasagar_Domain/Cell Cumulative Precipitation Depth']
                mm = float(applied[j,cid]-applied[i,cid])
                rows.append(dict(station=station, run=label, start_utc=str(start), end_utc=str(end),
                    gauge_interval_count=len(rain), gauge_point_rain_mm=float(rain.value.sum()),
                    applied_effective_cell_rain_mm=mm, reconstructed_input_before_r03_mm=mm/.3))
df = pd.DataFrame(rows)
df.to_csv(RES/'gauge_rain_audit.csv', index=False)
print(df.round(3).to_string(index=False))
