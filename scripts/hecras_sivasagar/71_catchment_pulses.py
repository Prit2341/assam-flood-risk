"""Net export pulses from fixed mesh-cell groups near delineated basins.

This is a cell-set water budget, NOT an observed or exact river discharge
hydrograph. Cross-divide/lateral exchanges and coarse boundary cells count.
"""
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
R = ROOT/'digital_twin/sivasagar/results'
OUT = R/'routing_audit'
GEO = 'Geometry/2D Flow Areas/Sivasagar_Domain'
TS = 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series'
B = f'{TS}/2D Flow Areas/Sivasagar_Domain'
basins = gpd.read_file(ROOT/'data/basin_delineation/sivasagar_fabdem/basins_traced.gpkg',layer='basins').to_crs(32646)
files = dict(uniform_raw='c600_d4_r30_0705_432h_raw.p01.hdf',
             ERA5_grid='c600_d4_gridera5full_r30_0705_432h.p01.hdf',
             IMERG_grid='c600_d4_gridimerg_r30_0705_432h.p01.hdf')
all_rows,checks = [],[]
fig,axes = plt.subplots(2,1,figsize=(11,8),sharex=True)
for label,filename in files.items():
    with h5py.File(R/filename) as f:
        g = f[GEO]
        real = np.flatnonzero((g['Cells FacePoint Indexes'][:] >= 0).sum(axis=1) >= 3)
        xy = g['Cells Center Coordinate'][:][real]
        area = g['Cells Surface Area'][:][real]
        levels = f[f'{B}/Water Surface'][:,real]
        times = pd.to_datetime([v.decode() for v in f[f'{TS}/Time Date Stamp'][:]],format='%d%b%Y %H:%M:%S')
        volumes = np.zeros(levels.shape,dtype=float)
        info,table = g['Cells Volume Elevation Info'][:],g['Cells Volume Elevation Values'][:]
        for j,cid in enumerate(real):
            start,count = info[cid]
            curve = table[start:start+count]
            volumes[:,j] = np.interp(levels[:,j],curve[:,0],curve[:,1],left=0)
            volumes[:,j] += np.maximum(levels[:,j]-curve[-1,0],0)*area[j]
        reported = f[f'{B}/Computations/Volume'][:,0].astype(float)*1000
        errors = volumes.sum(axis=1)-reported
        assert np.all(abs(errors) <= np.maximum(10000,abs(reported)*.0001)), (label,float(abs(errors).max()))
        checks.append(dict(run=label,max_storage_reconstruction_difference_m3=float(abs(errors).max())))
        rain = f[f'{B}/Cell Cumulative Precipitation Depth'][:,real].astype(float)*area[None,:]/1000
        for ai,name in enumerate(['Bihubar','Nanglamoraghat']):
            row = basins[basins.gauge == name].iloc[0]
            mask = shapely.contains_xy(row.geometry,xy[:,0],xy[:,1])
            v,r = volumes[:,mask].sum(axis=1),rain[:,mask].sum(axis=1)
            seconds = np.diff(times.as_unit('ns').astype('int64'))/1e9
            net = (np.diff(r)-np.diff(v))/seconds
            frame = pd.DataFrame(dict(time_end_utc=times[1:],run=label,gauge=name,
                net_export_rate_m3s=net,selected_cell_area_km2=float(area[mask].sum()/1e6),
                geometric_basin_area_km2=float(row.geometry.area/1e6)))
            all_rows.append(frame)
            axes[ai].plot(times[1:],pd.Series(net).rolling(6,center=True,min_periods=1).mean(),label=label)
            axes[ai].set_title(f'{name} upstream cell group: net export, six-hour display mean')
            axes[ai].set_ylabel('Net export rate (m³/s)');axes[ai].grid(alpha=.2);axes[ai].legend()
df = pd.concat(all_rows)
df.to_csv(OUT/'catchment_net_export_hourly.csv',index=False)
summaries = []
for (run,gauge),g in df.groupby(['run','gauge']):
    pre = g[(g.time_end_utc >= '2026-07-15') & (g.time_end_utc < '2026-07-18')]
    surge = g[(g.time_end_utc >= '2026-07-18') & (g.time_end_utc <= '2026-07-21')]
    peak = surge.loc[surge.net_export_rate_m3s.idxmax()]
    summaries.append(dict(run=run,gauge=gauge,pre_mean_net_export_m3s=float(pre.net_export_rate_m3s.mean()),
        surge_max_net_export_m3s=float(peak.net_export_rate_m3s),surge_peak_time_utc=str(peak.time_end_utc),
        selected_cell_area_km2=float(g.selected_cell_area_km2.iloc[0]),geometric_basin_area_km2=float(g.geometric_basin_area_km2.iloc[0])))
report = dict(summaries=summaries,storage_closure=checks,
    note='Net external exchange from a fixed cell set approximating a delineated basin, not exact gauge discharge. Includes all lateral crossings, coarse boundary assignment and any incoming flow. Stage peaks cannot be converted into discharge without a rating relation. Display smoothing only; CSV/peak statistics use raw hourly budget.')
(OUT/'catchment_pulse_summary.json').write_text(json.dumps(report,indent=2))
axes[-1].set_xlim(pd.Timestamp('2026-07-15'),pd.Timestamp('2026-07-23'))
fig.suptitle('Saved-model catchment export timing\nCell-set water budgets; these are not measured river discharge hydrographs')
fig.autofmt_xdate();fig.tight_layout();fig.savefig(OUT/'catchment_export_pulses.png',dpi=160)
print(json.dumps(report,indent=2))
