"""Trace two terrain-derived trunks and inspect saved channel-crossing faces.

This is a structural/instantaneous connectivity diagnostic, not a parcel or
integrated-discharge reconstruction. No model or terrain is modified.
"""
from pathlib import Path
import importlib.util
import json
import h5py
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import LineString, Point
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]
D = ROOT/'digital_twin/sivasagar/data'
R = ROOT/'digital_twin/sivasagar/results'
OUT = R/'routing_audit'
OUT.mkdir(exist_ok=True)
D8 = {1:(-1,1),2:(0,1),4:(1,1),8:(1,0),16:(1,-1),32:(0,-1),64:(-1,-1),128:(-1,0)}
spec = importlib.util.spec_from_file_location('ownership',Path(__file__).with_name('52_audit_map_ownership.py'))
own = importlib.util.module_from_spec(spec)
spec.loader.exec_module(own)
with rasterio.open(D/'d8_pointer_full.tif') as src:
    ptr,tr = src.read(1),src.transform
with rasterio.open(D/'d8_accum_full.tif') as src:
    acc = src.read(1)
exit_rc = tuple(json.loads((D/'outlet.json').read_text())['exit_rc'])
to_utm = Transformer.from_crs(4326,32646,always_xy=True)
routes = []
for name,lat,lon in [('Dikhow',26.856,94.801),('Desang',27.047,94.913)]:
    x,y = to_utm.transform(lon,lat)
    col,row = ~tr*(x,y)
    row,col = int(row),int(col)
    candidates = []
    for rr in range(row-17,row+18):
        for cc in range(col-17,col+18):
            xx,yy = tr*(cc+.5,rr+.5)
            if np.hypot(xx-x,yy-y) <= 500:
                candidates.append((float(acc[rr,cc]),rr,cc))
    _,row,col = max(candidates)
    snapped = (row,col)
    upstream,seen = [snapped],{snapped}
    while acc[upstream[-1]]*.0009 >= 100:
        rr,cc = upstream[-1]
        incoming = []
        for dr,dc in D8.values():
            nr,nc = rr+dr,cc+dc
            if 0 <= nr < ptr.shape[0] and 0 <= nc < ptr.shape[1]:
                step = D8.get(int(ptr[nr,nc]))
                if step and (nr+step[0],nc+step[1]) == (rr,cc):
                    incoming.append((float(acc[nr,nc]),nr,nc))
        if not incoming:break
        _,nr,nc = max(incoming)
        if (nr,nc) in seen:raise ValueError('Upstream loop')
        upstream.append((nr,nc));seen.add((nr,nc))
    path = upstream[::-1]
    seen = set(path)
    while path[-1] != exit_rc:
        rr,cc = path[-1]
        step = D8.get(int(ptr[rr,cc]))
        if not step:break
        nxt = rr+step[0],cc+step[1]
        if not (0 <= nxt[0] < ptr.shape[0] and 0 <= nxt[1] < ptr.shape[1]) or nxt in seen:break
        path.append(nxt);seen.add(nxt)
    xy = np.array([tr*(cc+.5,rr+.5) for rr,cc in path])
    distance = np.r_[0,np.hypot(*np.diff(xy,axis=0).T).cumsum()]
    routes.append(dict(name=name,xy=xy,distance=distance,snapped=snapped,
        snapped_upstream_km2=float(acc[snapped]*.0009),reaches_exit=path[-1] == exit_rc))
del ptr,acc
TS = 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series'
GEO = 'Geometry/2D Flow Areas/Sivasagar_Domain'
files = dict(IMERG='c600_d4_gridimerg_r30_0705_432h.p01.hdf',
             wide_IMERG='c600_d4_gridw4imerg_r30_0705_432h.p01.hdf',
             rain_B1='c600_d4_gridrainb1_r30_0705_432h.p01.hdf')
rows,summaries = [],[]
for label,filename in files.items():
    with h5py.File(R/filename) as f:
        g = f[GEO]
        face_cells = g['Faces Cell Indexes'][:]
        face_ids = {tuple(sorted(map(int,pair))):i for i,pair in enumerate(face_cells)}
        corners = g['FacePoints Coordinate'][:]
        edge_points = g['Faces FacePoint Indexes'][:]
        face_z = g['Faces Minimum Elevation'][:]
        times = pd.to_datetime([v.decode() for v in f[f'{TS}/Time Date Stamp'][:]],format='%d%b%Y %H:%M:%S')
        ti = times.get_loc(pd.Timestamp('2026-07-20 04:00'))
        wse = f[f'{TS}/2D Flow Areas/Sivasagar_Domain/Water Surface'][ti]
        velocities = f[f'{TS}/2D Flow Areas/Sivasagar_Domain/Face Velocity'][ti]
        real_mask = (g['Cells FacePoint Indexes'][:] >= 0).sum(axis=1) >= 3
        active = (face_cells >= 0).all(axis=1)
        fc = face_cells[active]
        active_ids = np.flatnonzero(active)
        keep = (real_mask[fc[:,0]] & real_mask[fc[:,1]] &
                (np.maximum(wse[fc[:,0]],wse[fc[:,1]]) > face_z[active_ids]) &
                (np.abs(velocities[active_ids]) > 1e-6))
        fc = fc[keep]
        graph = coo_matrix((np.ones(len(fc)),(fc[:,0],fc[:,1])),shape=(len(wse),len(wse))).tocsr()
        _,components = connected_components(graph,directed=False)
        for route in routes:
            xy = route['xy']
            owners,_ = own.mesh_owners(g,xy[:,0],xy[:,1])
            transitions = np.flatnonzero(owners[1:] != owners[:-1])+1
            matched,unresolved,blocked = 0,0,0
            for i in transitions:
                a,b = int(owners[i-1]),int(owners[i])
                if min(a,b) < 0:continue
                fi = face_ids.get(tuple(sorted((a,b))))
                if fi is None:
                    unresolved += 1
                    continue
                matched += 1
                face_line = LineString(corners[edge_points[fi]])
                segment = LineString(xy[i-1:i+1])
                cross = face_line.intersection(segment)
                if cross.is_empty:
                    unresolved += 1
                    continue
                point = cross if isinstance(cross,Point) else cross.centroid
                terrain = 'dem_burn_w4m.tif' if label == 'wide_IMERG' else 'dem_burn_4m.tif'
                with rasterio.open(D/terrain) as dem:
                    pixel_z = float(next(dem.sample([(point.x,point.y)]))[0])
                closed = float(max(wse[a],wse[b])) <= float(face_z[fi])
                blocked += int(closed)
                rows.append(dict(run=label,river=route['name'],distance_km=float(route['distance'][i]/1000),
                    cell_up=a,cell_down=b,face=int(fi),face_min_z_m=float(face_z[fi]),
                    crossing_raster_nearest_z_m=pixel_z,
                    face_min_minus_crossing_pixel_m=float(face_z[fi]-pixel_z),
                    upstream_cell_wse_m=float(wse[a]),downstream_cell_wse_m=float(wse[b]),
                    face_velocity_mps=float(velocities[fi]),
                    closed_by_face_minimum=bool(closed),x=point.x,y=point.y))
                rows[-1]['same_active_face_component'] = bool(components[a] == components[b])
            summaries.append(dict(run=label,river=route['name'],length_km=float(route['distance'][-1]/1000),
                snapped_upstream_km2=route['snapped_upstream_km2'],reaches_model_exit=route['reaches_exit'],
                all_owner_transitions=len(transitions),matched_face_transitions=matched,
                unresolved_transitions=unresolved,faces_closed_at_peak=blocked))
df = pd.DataFrame(rows)
df.to_csv(OUT/'channel_crossing_faces.csv',index=False)
gpd.GeoDataFrame({'river':[r['name'] for r in routes]},geometry=[LineString(r['xy']) for r in routes],crs=32646).to_file(OUT/'traced_trunks.gpkg',driver='GPKG')
report = dict(summaries=summaries,
    note='Terrain-derived main trunks, >=100km2 upstream head. Snap max accumulation within500m of official rounded Bihubar/Desangpani coordinates. Connectivity uses face minimum and adjacent stages at July20 04UTC, following USACE Mapping Options. Does not measure integrated flow, prove surveyed channel identity, or exclude alternate routes. Nearest raster crossing sample is not the exact interpolated face profile; elevation differences are diagnostic only.')
(OUT/'channel_connectivity_summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
print('Highest crossing face-minus-raster differences:')
print(df.nlargest(12,'face_min_minus_crossing_pixel_m')[['run','river','distance_km','face_min_minus_crossing_pixel_m','closed_by_face_minimum']].round(3).to_string(index=False))
