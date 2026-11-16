"""Full-grid paired calibration check for registered A2 ridge-alignment pilot.

Containing real-cell horizontal mapping for BOTH arms; legacy nearest scores
are descriptive. No alteration to historical result files or production scorer.
"""
from pathlib import Path
import importlib.util
import json
import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from pyproj import Transformer
from shapely.geometry import Polygon
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "digital_twin/sivasagar"
R = BASE / "results"
AREA = "Sivasagar_Domain"
GEO = f"Geometry/2D Flow Areas/{AREA}"
TS = "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series"
spec = importlib.util.spec_from_file_location("score", Path(__file__).with_name("05_score.py"))
score = importlib.util.module_from_spec(spec); spec.loader.exec_module(score)


def main():
    tag, when = score.SCENES["2026-07-17"]
    tif = next(score.GFM.glob(f"*{tag}*_flood.tif"))
    pre_tif = next(score.GFM.glob("*pre_2026-07-05*_flood.tif"))
    with rasterio.open(tif) as src:
        obs = src.read(1).ravel()
        shape, tr, crs = src.shape, src.transform, src.crs
    with rasterio.open(pre_tif) as src:
        if src.shape != shape or src.transform != tr or src.crs != crs:
            raise ValueError("Pre-water and target grids differ")
        pre = src.read(1).ravel()
    z = score.raw_z(tif, 0)
    rr, cc = np.mgrid[0:shape[0], 0:shape[1]]
    px, py = tr*(cc.ravel()+.5, rr.ravel()+.5)
    ux, uy = Transformer.from_crs(crs, 32646, always_xy=True).transform(px, py)
    del rr, cc, px, py
    files = dict(control=R/"c600_d4_r30_0705_432h_raw.p01.hdf",
                 ridge_a2=R/"c600_d4_r30_0705_312h_ridge_a2.p01.hdf")
    maps, legacy, metadata = {}, {}, {}
    for label, path in files.items():
        with h5py.File(path, "r") as f:
            g = f[GEO]
            verts = g["Cells FacePoint Indexes"][:]
            points = g["FacePoints Coordinate"][:]
            xx, yy = Transformer.from_crs(32646, crs, always_xy=True).transform(points[:, 0], points[:, 1])
            projected = np.c_[xx, yy]
            real = np.flatnonzero((verts >= 0).sum(axis=1) >= 3)
            polygons = [(Polygon(projected[row[row >= 0]]), int(cid))
                        for cid, row in zip(real, verts[real])]
            owners = rasterize(polygons, out_shape=shape, transform=tr,
                               fill=-1, dtype="int32", all_touched=False).ravel()
            stamps = pd.to_datetime([s.decode() for s in f[f"{TS}/Time Date Stamp"][:]],
                                    format="%d%b%Y %H:%M:%S")
            ti = int(np.argmin(np.abs(stamps-pd.Timestamp(when))))
            if abs(stamps[ti]-pd.Timestamp(when)) > pd.Timedelta(hours=1):
                raise ValueError("No appropriately timed model output")
            wse = f[f"{TS}/2D Flow Areas/{AREA}/Water Surface"][ti]
            if not np.isfinite(wse[real]).all():
                raise ValueError(f"Nonfinite real-cell water surface in {label}")
            xy = g["Cells Center Coordinate"][:]
            nearest = cKDTree(xy).query(np.c_[ux, uy])[1]
            maps[label] = (owners >= 0, wse[np.maximum(owners, 0)]-z > .1)
            legacy[label] = wse[nearest]-z > .1
            n = g["Cells Center Manning's n"][:]
            metadata[label] = dict(file=str(path), output_time=str(stamps[ti]),
                                   real_cells=int(len(real)),
                                   output_start=str(stamps[0]), output_end=str(stamps[-1]),
                                   mass_error_percent=float(f["Results/Unsteady/Summary/Volume Accounting"].attrs["Error Percent"]),
                                   manning_min=float(n[real].min()), manning_max=float(n[real].max()))
            if not np.allclose(n[real], .04):
                raise ValueError(f"Roughness changed in {label}; comparison is confounded")
        print("mapped", label, flush=True)
    valid = (obs != 255) & (pre != 1) & np.isfinite(z)
    covered = maps["control"][0] & maps["ridge_a2"][0]
    paired = valid & covered
    output = []
    for label in files:
        for method, model in [("containing", maps[label][1]), ("nearest_descriptive", legacy[label])]:
            o = obs == 1
            h, miss, fa = [int(np.count_nonzero(v)) for v in
                           [paired & o & model, paired & o & ~model, paired & ~o & model]]
            output.append(dict(arm=label, mapping=method, valid_km2=paired.sum()*4e-4,
                uncovered_valid_fraction=float((valid & ~covered).sum()/max(valid.sum(), 1)),
                hits_km2=h*4e-4, misses_km2=miss*4e-4, false_alarms_km2=fa*4e-4,
                CSI=h/max(h+miss+fa, 1), POD=h/max(h+miss, 1),
                area_ratio=(h+fa)/max(h+miss, 1)))
    df = pd.DataFrame(output)
    primary = df[df.mapping == "containing"].set_index("arm")
    delta = float(primary.loc["ridge_a2", "CSI"]-primary.loc["control", "CSI"])
    hit_ratio = float(primary.loc["ridge_a2", "hits_km2"]/primary.loc["control", "hits_km2"])
    report = dict(scene="2026-07-17", scene_role="calibration/exploratory",
        metadata=metadata, primary_CSI_gain=delta, primary_hit_retention=hit_ratio,
        pilot_pass=bool(delta >= .01 and hit_ratio >= .9),
        G1_pass=False, G1_note="One calibration scene cannot establish G1")
    df.to_csv(R/"ridge_a2_pilot_scores.csv", index=False)
    (R/"ridge_a2_pilot_verdict.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(df.to_string(index=False)); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
