"""Reconstruct flux across boundary of July-17 false-alarm cell set."""
import contextlib, io, json, runpy
from pathlib import Path
import h5py, numpy as np

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'digital_twin/sivasagar/results'
with contextlib.redirect_stdout(io.StringIO()):
    d=runpy.run_path(str(Path(__file__).with_name('57_check_local_vs_routed.py')))
target=np.zeros(len(d['water']),bool)
target[d['fa_cells']]=True
at=d['d']['at']
with h5py.File(R/'c600_d4_r30_0705_432h_raw.p01.hdf') as f:
    g=f['Geometry/2D Flow Areas/Sivasagar_Domain']
    b=f['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Sivasagar_Domain']
    cells=g['Faces Cell Indexes'][:]
    boundary=np.flatnonzero(target[cells[:,0]] ^ target[cells[:,1]])
    info=g['Faces Area Elevation Info'][:]
    tab=g['Faces Area Elevation Values'][:]
    lengths=g['Faces NormalUnitVector and Length'][:,2]
    xy=g['Cells Center Coordinate'][:]
    wse=b['Water Surface'][:at+1]
    velocity=b['Face Velocity'][:at+1,boundary]
    totals=dict(east=0.,west=0.,north=0.,south=0.,ghost=0.)
    incoming=outgoing=0.
    for col,fi in enumerate(boundary):
        c0,c1=map(int,cells[fi]);s,n=info[fi]; curve=tab[s:s+n]
        level=(wse[:,c0]+wse[:,c1])/2
        a=np.interp(np.minimum(level,curve[-1,0]),curve[:,0],curve[:,1],left=0)
        a+=np.maximum(level-curve[-1,0],0)*lengths[fi]
        sign=1 if target[c1] else -1
        flux=velocity[:,col]*a*sign
        qin=np.maximum(flux,0);qout=np.maximum(-flux,0)
        vin=float(np.trapezoid(qin,dx=3600));vout=float(np.trapezoid(qout,dx=3600))
        incoming+=vin;outgoing+=vout
        donor=c0 if target[c1] else c1
        receiver=c1 if target[c1] else c0
        if donor>=len(target) or (g['Cells FacePoint Indexes'][donor]>=0).sum()<3:
            direction='ghost'
        else:
            dx,dy=xy[donor]-xy[receiver]
            direction=('east' if dx>0 else 'west') if abs(dx)>=abs(dy) else ('north' if dy>0 else 'south')
        totals[direction]+=vin
    storage_change=float(d['water'][d['fa_cells']].sum())
    direct_rain=float(d['local'][d['fa_cells']].sum())
    report=dict(scene='2026-07-17',boundary_face_count=int(len(boundary)),
        valid_for_directional_inference=False,
        cumulative_in_m3=incoming,cumulative_out_m3=outgoing,net_lateral_m3=incoming-outgoing,
        storage_m3=storage_change,direct_rain_m3=direct_rain,
        implied_net_lateral_from_storage_m3=storage_change-direct_rain,
        closure_difference_m3=(incoming-outgoing)-(storage_change-direct_rain),
        cumulative_in_by_immediate_donor_direction_m3=totals,
        note='REJECTED: hourly face velocity times interpolated conveyance area fails cell water-balance closure. Directional totals are unusable; retain only as a failed method record.')
R.joinpath('excess_water_face_flux_0717.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
