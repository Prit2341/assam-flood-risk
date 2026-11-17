"""Build the preregistered C3 channel-only mesh experiment using RAS Commander."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import shutil
import geopandas as gpd
import h5py
import numpy as np
import rasterio
import shapely
from shapely.geometry import LineString,Polygon
from ras_commander import init_ras_project
from ras_commander.geom import GeomMesh
from ras_commander.RasMap import RasMap

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'digital_twin/sivasagar'
D = BASE/'data'
SRC = BASE/'hecras_c600_d4_grid_imerg'
parser = argparse.ArgumentParser()
parser.add_argument('--resume',action='store_true',help='Resume this uncomputed experimental mesh only')
parser.add_argument('--spacing',type=int,choices=[50,100],default=50)
parser.add_argument('--label',choices=['channelc3','channelc3fast'],default='channelc3')
args = parser.parse_args()
SPACING = args.spacing
OUT = BASE/('results/channel_c3' if args.label == 'channelc3' else 'results/channel_c3fast')
OUT.mkdir(exist_ok=True)
DST = BASE/f'hecras_c600_d4_grid_{args.label}'
if DST.exists() and not args.resume:raise FileExistsError(f'Refusing to replace {DST}')
if args.resume:
    assert DST.exists() and not (OUT/'build.json').exists(), 'Only resume an unvalidated C3 build'
audit = BASE/'results/routing_audit'
routes = gpd.read_file(audit/'traced_trunks.gpkg')
ref = gpd.read_file(audit/'grwl_reference.geojson').to_crs(32646)
reference = ref[ref.lakeFlag == 0].geometry.union_all()
GEO = 'Geometry/2D Flow Areas/Sivasagar_Domain'
with h5py.File(SRC/'Sivasagar_2D.g01.hdf') as f:
    perimeter = Polygon(f[f'{GEO}/Perimeter'][:])
    old_real = int(((f[f'{GEO}/Cells FacePoint Indexes'][:] >= 0).sum(axis=1) >= 3).sum())
lines = []
with rasterio.open(D/'dem_raw.tif') as z, rasterio.open(D/'d8_accum_full.tif') as acc:
    for _,route in routes.iterrows():
        xy = np.asarray(route.geometry.coords)
        zz = np.array([v[0] for v in z.sample(xy)])
        area = np.array([v[0] for v in acc.sample(xy)])*.0009
        distance = shapely.distance(shapely.points(xy),reference)
        keep = (zz > 0) & (zz < 130) & (area >= 1000) & (distance <= 150)
        indexes = np.flatnonzero(keep)
        chunks = np.split(indexes,np.flatnonzero(np.diff(indexes)>1)+1)
        for chunk in chunks:
            if len(chunk) < 2:continue
            line = LineString(xy[chunk])
            if line.length >= 1000:lines.append(line)
assert lines
line_union = shapely.union_all(lines)
buffered = line_union.buffer(100).intersection(perimeter.buffer(-100))
pieces = list(buffered.geoms) if hasattr(buffered,'geoms') else [buffered]
regions = [Polygon(p.exterior).simplify(10,preserve_topology=True) for p in pieces if p.geom_type == 'Polygon' and p.area > 0]
assert regions and all(p.is_valid for p in regions)
actual = shapely.union_all(regions)
extra_fraction = (actual.area-buffered.area)/buffered.area
assert extra_fraction <= .25, f'Filled buffer islands add too much area: {extra_fraction}'
gpd.GeoDataFrame({'name':[f'C3_channel_{i:02d}' for i in range(len(regions))]},geometry=regions,crs=32646).to_file(OUT/'refinement_regions.gpkg',driver='GPKG')
gpd.GeoDataFrame({'name':['screened_trunks']},geometry=[line_union],crs=32646).to_file(OUT/'screened_trunks.gpkg',driver='GPKG')
if not args.resume:
    shutil.copytree(SRC,DST,ignore=shutil.ignore_patterns('*.p01.hdf','*.b01','*.bco01','*.ic.o01','*.x01','*.log'))
ras = init_ras_project(str(DST),'7.0.1')
gfile = DST/'Sivasagar_2D.g01'
api_globals = GeomMesh.generate.__wrapped__.__globals__
api_globals['_load_dlls'](None)
native_types = api_globals['_imports']()
def write_native_regions(hdf_path,ns):
    import System
    from RasMapperLib import PointM
    native = ns['RASGeometry'](str(hdf_path))
    layer = native.MeshRegions
    if layer.FeatureCount():
        assert layer.FeatureCount() == len(regions)
        return
    # The installed wrapper writes Spacing dx/dy. HEC-RAS7.0.1 expects
    # Cell Size X/Y and its native polygon serialization. Use its writer.
    for i,polygon in enumerate(regions):
        points = ns['PointMs']()
        for x,y in polygon.exterior.coords:points.Add(PointM(float(x),float(y),0.0))
        row = layer.AddFeature(ns['Polygon'](points))
        row['Name'] = f'C3_channel_{i:02d}'
        for key,val in [('Cell Size X',SPACING),('Cell Size Y',SPACING),('Cell Shift DX',0),
                        ('Cell Shift DY',0),('Perimeter Spacing',SPACING),('Far Spacing',600)]:
            row[key] = System.Single(val)
        row['Near Repeats'] = System.Int32(1)
        row['Generate as Hexagon'] = False
        row['Enforce 1 Cell Protection Radius'] = False
    assert layer.Save(), 'Native refinement layer save failed'
    check = ns['RASGeometry'](str(hdf_path))
    assert check.MeshRegions.FeatureCount() == len(regions), 'Native region definitions did not round-trip'
    print('Native region writer round-trip passed:',len(regions),flush=True)
with h5py.File(str(gfile)+'.hdf','a') as hf:
    if 'Geometry/Refinement Regions' in hf:
        del hf['Geometry/Refinement Regions']  # remove only the failed naming probe
write_native_regions(str(gfile)+'.hdf',native_types)
region_fids = list(range(len(regions)))

# ras-commander0.99.1's seed helper passes an EMPTY refinement-region
# index list to RegenerateMeshPoints, ignoring authored regions. Supply
# their verified FIDs in this process only. Preserve its invocation shape.
def seeds_with_regions(geom_hdf_path,ns,fid=0):
    import System
    from System.Reflection import BindingFlags
    from System.Collections.Generic import List as NetList
    write_native_regions(geom_hdf_path,ns)
    geom = ns['RASGeometry'](str(geom_hdf_path))
    assert geom.MeshRegions.FeatureCount() == len(region_fids)
    assert geom.SA2DStructures.FeatureCount() == 0, 'Do not omit SA2D structure enforcement'
    bl_idx,rr_idx,perim_idx = [NetList[System.Int32]() for _ in range(3)]
    for i in range(geom.BreakLines.FeatureCount()):bl_idx.Add(i)
    for i in region_fids:rr_idx.Add(i)
    for i in range(geom.D2FlowArea.FeatureCount()):perim_idx.Add(i)
    pg = ns['PointGenerator'](geom)
    method = pg.GetType().GetMethod('RegenerateMeshPoints',BindingFlags.Instance|BindingFlags.NonPublic)
    assert method is not None
    method.Invoke(pg,System.Array[System.Object]([bl_idx,rr_idx,perim_idx,perim_idx,None,None,False]))
    points = geom.D2FlowArea.Geometry.MeshPoints[fid]
    seeds = ns['PointMs']()
    for i in range(points.PointMCount()):seeds.Add(points.PointM(i))
    print('Seeds including refinement regions:',seeds.Count,flush=True)
    return seeds
GeomMesh.generate.__wrapped__.__globals__['_generate_seeds_via_net'] = seeds_with_regions
spec = importlib.util.spec_from_file_location('setup',Path(__file__).with_name('03_setup_project.py'))
setup = importlib.util.module_from_spec(spec);spec.loader.exec_module(setup)
setup.patch_reseed(ras)
result = GeomMesh.generate(gfile,mesh_name='Sivasagar_Domain',cell_size=600,
    recompile_via_rasexe=False,_require_current_hdf=False,ras_object=ras)
print('Mesh result:',result,flush=True)
assert result.status == 'complete', result
RasMap.set_geometry_association('01',terrain_hdf_path=DST/'Terrain/Terrain.hdf',ras_object=ras,validate=False)
with h5py.File(DST/'Sivasagar_2D.g01.hdf') as f:
    g = f[GEO]
    real = np.flatnonzero((g['Cells FacePoint Indexes'][:] >= 0).sum(axis=1) >= 3)
    xy = g['Cells Center Coordinate'][:]
    vertices = g['Cells FacePoint Indexes'][:]
    corners = g['FacePoints Coordinate'][:]
    cell_area = np.array([Polygon(corners[vertices[c][vertices[c]>=0]]).area for c in real])
    inside = shapely.intersects_xy(actual,xy[real,0],xy[real,1])
    assert len(real) > old_real
    assert inside.any() and np.median(cell_area[inside]) <= 2*SPACING**2, 'Refinement not reflected in mesh cell areas'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(SRC/'rain_grid.nc') == sha(DST/'rain_grid.nc')
for p in (SRC/'Terrain').rglob('*'):
    if p.is_file():assert sha(p) == sha(DST/p.relative_to(SRC))
report = dict(project=str(DST),control=str(SRC),old_real_cells=old_real,new_real_cells=len(real),
    refinement_regions=len(regions),screened_trunk_km=float(line_union.length/1000),
    original_corridor_km2=float(buffered.area/1e6),actual_refined_region_km2=float(actual.area/1e6),
    extra_enclosed_island_area_fraction=float(extra_fraction),
    region_cell_count=int(inside.sum()),median_region_cell_area_m2=float(np.median(cell_area[inside])),
    spacing_m=SPACING,base_spacing_m=600,terrain_identical=True,rain_identical=True,
    rain_sha256=sha(SRC/'rain_grid.nc'),build_pass=True,
    limitation='Screened historical alignment and synthetic channel geometry; no surveyed-bank or bathymetry claim. Compiled hydraulic roughness must be checked after preprocessing/run.')
(OUT/'build.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
