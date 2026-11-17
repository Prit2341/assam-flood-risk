"""Saved-output elevation-band budgets; no hourly face-flux integration.

Bands are fixed mesh-cell groups, not surveyed subcatchments. Their net
lateral exchange follows storage change minus applied effective rainfall.
"""
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT/'digital_twin/sivasagar/results'
OUT = R/'routing_audit'
OUT.mkdir(exist_ok=True)
GEO = 'Geometry/2D Flow Areas/Sivasagar_Domain'
TS = 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series'
B = f'{TS}/2D Flow Areas/Sivasagar_Domain'
FILES = dict(uniform_raw='c600_d4_r30_0705_432h_raw.p01.hdf',
             ERA5_grid='c600_d4_gridera5full_r30_0705_432h.p01.hdf',
             IMERG_grid='c600_d4_gridimerg_r30_0705_432h.p01.hdf',
             rain_b1='c600_d4_gridrainb1_r30_0705_432h.p01.hdf')
with h5py.File(R/FILES['IMERG_grid']) as f:
    vertices = f[f'{GEO}/Cells FacePoint Indexes'][:]
    real = np.flatnonzero((vertices >= 0).sum(axis=1) >= 3)
    z = f[f'{GEO}/Cells Minimum Elevation'][:]
groups = {'plain_min_below100':real[z[real] < 100],
          'foothill_min100to300':real[(z[real] >= 100) & (z[real] < 300)],
          'hill_min300plus':real[z[real] >= 300]}
assert sum(map(len,groups.values())) == len(real)
checkpoints = ['2026-07-05 00:00','2026-07-18 00:00','2026-07-20 04:00','2026-07-23 00:00']
rows,closure = [],[]
for label,filename in FILES.items():
    with h5py.File(R/filename) as f:
        assert np.array_equal(vertices,f[f'{GEO}/Cells FacePoint Indexes'][:])
        g = f[GEO]
        areas = g['Cells Surface Area'][:]
        info,curves = g['Cells Volume Elevation Info'][:],g['Cells Volume Elevation Values'][:]
        times = pd.to_datetime([v.decode() for v in f[f'{TS}/Time Date Stamp'][:]],format='%d%b%Y %H:%M:%S')
        states = []
        for when in checkpoints:
            i = times.get_loc(pd.Timestamp(when))
            levels = f[f'{B}/Water Surface'][i]
            volume = np.zeros(len(areas),dtype=float)
            for cid in real:
                start,count = info[cid]
                table = curves[start:start+count]
                level = float(levels[cid])
                if level > table[0,0]:
                    volume[cid] = np.interp(min(level,table[-1,0]),table[:,0],table[:,1])
                    volume[cid] += max(level-table[-1,0],0)*areas[cid]
            rain = f[f'{B}/Cell Cumulative Precipitation Depth'][i]*areas/1000
            reported = float(f[f'{B}/Computations/Volume'][i,0]*1000)
            diff = float(volume[real].sum()-reported)
            closure.append(dict(run=label,time=when,reconstructed_m3=float(volume[real].sum()),reported_m3=reported,difference_m3=diff))
            # Dry-start WSE/property-table tolerances reconstruct a few thousand
            # m3 even when RAS reports zero; retain that initial volume in deltas.
            assert abs(diff) <= max(10000,abs(reported)*.0001), (label,when,diff,reported)
            states.append((volume,rain))
        for k in range(1,len(states)):
            va,ra = states[k-1]
            vb,rb = states[k]
            for name,ids in groups.items():
                delta_v = float((vb[ids]-va[ids]).sum())
                rain_v = float((rb[ids]-ra[ids]).sum())
                # Boundary exports, if any, are included in this net exchange.
                net_export = rain_v-delta_v
                rows.append(dict(run=label,band=name,start=checkpoints[k-1],end=checkpoints[k],
                    cells=len(ids),area_km2=float(areas[ids].sum()/1e6),
                    effective_rain_Mm3=rain_v/1e6,storage_change_Mm3=delta_v/1e6,
                    net_lateral_export_Mm3=net_export/1e6,
                    export_divided_by_interval_rain=net_export/rain_v if rain_v else None,
                    end_storage_Mm3=float(vb[ids].sum()/1e6)))
df = pd.DataFrame(rows)
df.to_csv(OUT/'elevation_band_budgets.csv',index=False)
(OUT/'volume_reconstruction_closure.json').write_text(json.dumps(closure,indent=2))
print(df[df.start == '2026-07-18 00:00'].round(3).to_string(index=False))
print('Max whole-domain reconstruction difference m3:',max(abs(v['difference_m3']) for v in closure))
