"""Compare modeled D8 burns with independent GRWL river centerlines.

GRWL is historical Landsat geometry, not a surveyed 2026 channel map.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import shapely
from scipy.ndimage import distance_transform_edt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
D = ROOT/'digital_twin/sivasagar/data'
OUT = ROOT/'digital_twin/sivasagar/results/routing_audit'
domain = gpd.read_file(D/'boundary_utm46n.gpkg').geometry.iloc[0]
reference = gpd.read_file(OUT/'grwl_reference.geojson').to_crs(32646)
reference = reference[reference.lakeFlag == 0].copy()
reference['geometry'] = reference.geometry.intersection(domain)
reference = reference[~reference.is_empty]
union = reference.geometry.union_all()
routes = gpd.read_file(OUT/'traced_trunks.gpkg')
with rasterio.open(D/'stream_mask.tif') as src:
    mask,tr = src.read(1)>0,src.transform
    h,w = src.shape
dist = distance_transform_edt(~mask)*30
reference_samples = []
for _,row in reference.iterrows():
    geom = row.geometry
    pieces = list(geom.geoms) if hasattr(geom,'geoms') else [geom]
    for piece in pieces:
        if piece.geom_type != 'LineString':continue
        for along in np.arange(0,piece.length,30):
            p = piece.interpolate(along)
            c,r = ~tr*(p.x,p.y)
            r,c = int(r),int(c)
            if 0 <= r < h and 0 <= c < w:
                reference_samples.append(dict(x=p.x,y=p.y,width_med_m=float(row.width_med_),
                    distance_to_any_model_stream_m=float(dist[r,c])))
ref = pd.DataFrame(reference_samples)
ref.to_csv(OUT/'grwl_to_model_stream_samples.csv',index=False)
del dist,mask
summary = []
points_for_plot = []
with rasterio.open(D/'dem_raw.tif') as terrain, rasterio.open(D/'d8_accum_full.tif') as acc:
    for _,route in routes.iterrows():
        xy = np.asarray(route.geometry.coords)
        z = np.array([v[0] for v in terrain.sample(xy)])
        area = np.array([v[0] for v in acc.sample(xy)])*.0009
        select = (z > 0) & (z < 130) & (area >= 1000)
        positions = xy[select]
        dd = shapely.distance(shapely.points(positions),union)
        summary.append(dict(river=route.river,samples=len(dd),distance_median_m=float(np.median(dd)),
            distance_p90_m=float(np.percentile(dd,90)),fraction_farther300m=float(np.mean(dd>300)),
            fraction_farther1000m=float(np.mean(dd>1000))))
        pd.DataFrame(dict(x=positions[:,0],y=positions[:,1],distance_to_GRWL_m=dd)).to_csv(OUT/f'{route.river.lower()}_model_to_grwl.csv',index=False)
        points_for_plot.append((positions,dd))
report = dict(reference_to_any_burned_stream=dict(samples=len(ref),
    distance_median_m=float(ref.distance_to_any_model_stream_m.median()),
    distance_p90_m=float(ref.distance_to_any_model_stream_m.quantile(.9)),
    fraction_farther300m=float((ref.distance_to_any_model_stream_m>300).mean())),
    modeled_main_trunks=summary,
    limitations='GRWL historical Landsat river geometry includes only its mapped rivers; channel migration and coverage gaps possible. Model comparison restricted to rawterrain0..130m and upstreamarea>=1000km2. Distance to ANY burned stream may match a different tributary. Samples approximately30m apart, no current-survey claim.')
(OUT/'channel_alignment_summary.json').write_text(json.dumps(report,indent=2))
fig,axes = plt.subplots(1,2,figsize=(12,7))
for ax,(_,route),(positions,dd) in zip(axes,routes.iterrows(),points_for_plot):
    gpd.GeoSeries([domain],crs=32646).boundary.plot(ax=ax,color='gray',linewidth=.5)
    reference.plot(ax=ax,color='royalblue',linewidth=1,label='Historical GRWL rivers')
    ax.plot(*np.asarray(route.geometry.coords).T,color='tomato',linewidth=.8,label='D8 modeled trunk')
    bad=positions[dd>300]
    ax.scatter(bad[:,0],bad[:,1],s=2,color='black',label='D8 >300 m from GRWL')
    ax.set_title(route.river);ax.set_aspect('equal');ax.ticklabel_format(style='plain');ax.legend(fontsize=7)
fig.suptitle('Modeled terrain-derived trunks versus historical observed river geometry\nDistances screen alignment; they do not establish the 2026 channel position')
fig.tight_layout();fig.savefig(OUT/'channel_alignment.png',dpi=160)
print(json.dumps(report,indent=2))
