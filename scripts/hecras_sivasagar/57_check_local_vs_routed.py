"""Bound routed water at false-alarm cells from saved HEC-RAS volume tables."""
import contextlib
import io
import json
from pathlib import Path
import runpy

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / 'digital_twin/sivasagar/results'
with contextlib.redirect_stdout(io.StringIO()):
    d = runpy.run_path(str(Path(__file__).with_name('56_trace_excess_water.py')))

with h5py.File(R / 'c600_d4_r30_0705_432h_raw.p01.hdf') as f:
    g = f['Geometry/2D Flow Areas/Sivasagar_Domain']
    info = g['Cells Volume Elevation Info'][:]
    curve = g['Cells Volume Elevation Values'][:]
    area = g['Cells Surface Area'][:]
    real = d['real']
    wse = f[f"{d['base']}/Water Surface"][d['at'], :]
    rain = f[f"{d['base']}/Cell Cumulative Precipitation Depth"][d['at'], :] / 1000
    water = np.zeros(len(area), dtype='float64')
    for cid in real:
        start, count = info[cid]
        tab = curve[start:start+count]
        level = float(wse[cid])
        if level <= tab[0,0]:
            continue
        water[cid] = np.interp(min(level, tab[-1,0]), tab[:,0], tab[:,1])
        if level > tab[-1,0]:
            water[cid] += (level-tab[-1,0])*area[cid]
    local = rain * area
    fa_cells, pixel_count = np.unique(d['fa_cell'], return_counts=True)
    # Water above direct local rain is a lower bound on net routed inflow.
    routed = water[fa_cells] > local[fa_cells] + 1
    deficit = np.maximum(water[fa_cells]-local[fa_cells], 0)
    report = dict(scene='2026-07-17', fa_cells=int(len(fa_cells)),
        fa_pixels_in_cells_with_more_water_than_local_rain_km2=float(pixel_count[routed].sum()*.0004),
        fa_pixels_in_other_cells_km2=float(pixel_count[~routed].sum()*.0004),
        fa_cell_water_m3=float(water[fa_cells].sum()),
        fa_cell_direct_rain_m3=float(local[fa_cells].sum()),
        fa_cell_minimum_net_import_m3=float(deficit.sum()),
        all_real_cell_water_m3=float(water[real].sum()),
        ras_reported_storage_m3=float(f[f"{d['base']}/Computations/Volume"][d['at'],0]*1000),
        method='Per-cell water volume interpolated from HEC-RAS elevation-volume tables; compare against cumulative direct effective rainfall. Water exceeding direct rain proves net lateral import, but water below it may still contain imported water offset by outward flow. False-alarm pixel categories inherit their containing cells; this is not a parcel trace.')
R.joinpath('excess_water_local_vs_routed_0717.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
