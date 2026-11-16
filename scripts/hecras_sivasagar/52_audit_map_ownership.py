"""Audit nearest-centre flood mapping against actual HEC-RAS cell ownership.

Research/registration: docs/teacher_improvement_review_2026-11-16.md, A1.
Read-only on existing model outputs; writes new diagnostic CSV/JSON files only.
Does not alter historical scores, observations, models or thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio
import shapely
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "digital_twin/sivasagar"
DATA = BASE / "data"
RESULTS = BASE / "results"
AREA = "Sivasagar_Domain"
GEOM = f"Geometry/2D Flow Areas/{AREA}"
TS = "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series"
DEPTH = 0.1  # frozen pixel-depth definition, metres


def mesh_owners(group, x, y):
    """Return actual polygon owner IDs; artificial boundary cells have 2 nodes."""
    corners = group["FacePoints Coordinate"][:]
    vertex_ids = group["Cells FacePoint Indexes"][:]
    cell_ids = np.flatnonzero((vertex_ids >= 0).sum(axis=1) >= 3)
    polys = [Polygon(corners[row[row >= 0]]) for row in vertex_ids[cell_ids]]
    if not np.all(shapely.is_valid(polys)):
        raise ValueError("Invalid real-cell polygons; do not repair geometry silently")
    found = STRtree(polys).query(shapely.points(x, y), predicate="within")
    owners = np.full(x.size, -1, dtype=np.int32)
    if found.shape[1]:
        if np.unique(found[0]).size != found.shape[1]:
            raise ValueError("Overlapping cell polygons at sampled pixel centres")
        owners[found[0]] = cell_ids[found[1]]
    return owners, cell_ids


def confusion(obs, model, valid, pixel_area):
    h = int(np.count_nonzero(valid & obs & model))
    m = int(np.count_nonzero(valid & obs & ~model))
    fa = int(np.count_nonzero(valid & ~obs & model))
    return dict(hits_px=h, misses_px=m, false_alarms_px=fa,
                hits_km2=h * pixel_area, misses_km2=m * pixel_area,
                false_alarms_km2=fa * pixel_area,
                observed_km2=(h + m) * pixel_area,
                model_km2=(h + fa) * pixel_area,
                CSI=h / max(h + m + fa, 1), POD=h / max(h + m, 1),
                area_ratio=(h + fa) / max(h + m, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--ratios", nargs="+", type=int, default=[15, 30, 50])
    args = ap.parse_args()
    if args.stride < 1:
        raise ValueError("stride must be positive")
    passes = pd.read_csv(DATA / "gfm_domain/passes.csv")
    pre_row = passes[passes.key.str.startswith("20260705")].iloc[0]
    held = passes[~passes.key.str.startswith("20260705")]
    with rasterio.open(DATA / "gfm_domain" / pre_row.file) as ref:
        height, width = ref.shape
    with np.load(DATA / f"gfm_domain_pixmap_{height}x{width}.npz") as cache:
        idx = cache["idx"][::args.stride]
        x = cache["ux"][::args.stride]
        y = cache["uy"][::args.stride]
    with rasterio.open(DATA / "dem_raw.tif") as ref:
        terrain = ref.read(1)
        rows, cols = rasterio.transform.rowcol(ref.transform, x, y)
        rows, cols = np.asarray(rows), np.asarray(cols)
        inside = ((rows >= 0) & (rows < ref.height) &
                  (cols >= 0) & (cols < ref.width))
        z = np.full(x.size, np.nan, dtype=np.float32)
        z[inside] = terrain[rows[inside], cols[inside]]
        if ref.nodata is not None:
            z[z == ref.nodata] = np.nan
    del terrain
    with rasterio.open(DATA / "gfm_domain" / pre_row.file) as ref:
        pre = ref.read(1).ravel()[idx]
    pixel_area = 4e-4 * args.stride  # area estimate for deterministic subsample
    records, dry_records, mesh_signature = [], [], None
    owner = nearest = real_ids = None
    ownership_report = {}
    for ratio in args.ratios:
        hdf = RESULTS / f"c600_d4_r{ratio}_0705_1176h_raw.p01.hdf"
        with h5py.File(hdf, "r") as f:
            g = f[GEOM]
            xy = g["Cells Center Coordinate"][:]
            signature = hashlib.sha256(xy.tobytes() +
                g["Cells FacePoint Indexes"][:].tobytes() +
                g["FacePoints Coordinate"][:].tobytes()).hexdigest()
            if mesh_signature is None:
                mesh_signature = signature
                owner, real_ids = mesh_owners(g, x, y)
                nearest = cKDTree(xy).query(np.c_[x, y])[1]
                covered = owner >= 0
                real_flag = np.zeros(len(xy), bool)
                real_flag[real_ids] = True
                ownership_report = dict(sampled_domain_pixels=len(idx),
                    containing_coverage=float(covered.mean()),
                    changed_owner_fraction=float((owner[covered] != nearest[covered]).mean()),
                    nearest_artificial_fraction=float((~real_flag[nearest]).mean()),
                    real_cell_count=len(real_ids), mesh_sha256=signature)
                print("ownership", ownership_report, flush=True)
            elif signature != mesh_signature:
                raise ValueError("Meshes differ; ownership must be rebuilt per mesh")
            times = pd.to_datetime([v.decode() for v in f[f"{TS}/Time Date Stamp"][:]],
                                   format="%d%b%Y %H:%M:%S")
            wse = f[f"{TS}/2D Flow Areas/{AREA}/Water Surface"]
            zmin = g["Cells Minimum Elevation"][:]
            w0 = wse[0]
            dv = (owner >= 0) & np.isfinite(z) & (pre != 1)
            for method, ids in [("nearest", nearest), ("containing", np.maximum(owner, 0))]:
                md = w0[ids] - z > DEPTH
                dry_records.append(dict(ratio=ratio, method=method,
                    initial_time=str(times[0]),
                    initial_max_real_cell_depth_m=float(np.nanmax(w0[real_ids] - zmin[real_ids])),
                    mapped_flood_km2=float(np.count_nonzero(dv & md) * pixel_area)))
            for _, scene in held.iterrows():
                when = pd.Timestamp(scene.pass_time).tz_convert(None)
                if not (times[0] <= when <= times[-1]):
                    raise ValueError(f"Scene outside simulation: {scene.key}")
                ti = int(np.argmin(np.abs(times - when)))
                levels = wse[ti]
                with rasterio.open(DATA / "gfm_domain" / scene.file) as ref:
                    obs = ref.read(1).ravel()[idx]
                valid = (obs != 255) & (pre != 1) & np.isfinite(z)
                paired = valid & (owner >= 0)
                old = levels[nearest] - z > DEPTH
                new = levels[np.maximum(owner, 0)] - z > DEPTH
                changed = float(np.count_nonzero(paired & (old != new)) / max(paired.sum(), 1))
                for method, model in [("nearest", old), ("containing", new)]:
                    row = dict(ratio=ratio, pass_key=scene.key, method=method,
                        requested_time=str(when), output_time=str(times[ti]),
                        stride=args.stride, valid_km2_estimate=float(paired.sum() * pixel_area),
                        unmatched_valid_fraction=float((valid & (owner < 0)).sum() / max(valid.sum(), 1)),
                        changed_wet_fraction=changed,
                        **confusion(obs == 1, model, paired, pixel_area))
                    records.append(row)
        print("completed ratio", ratio, flush=True)
    scores = pd.DataFrame(records)
    summaries = []
    for (ratio, method), group in scores.groupby(["ratio", "method"]):
        h, m, fa = group[["hits_px", "misses_px", "false_alarms_px"]].sum()
        summaries.append(dict(ratio=int(ratio), method=method,
            CSI=float(h / max(h + m + fa, 1)), POD=float(h / max(h + m, 1)),
            area_ratio=float((h + fa) / max(h + m, 1)),
            hits_km2_estimate=float(h * pixel_area), misses_km2_estimate=float(m * pixel_area),
            false_alarms_km2_estimate=float(fa * pixel_area)))
    report = dict(stride=args.stride, sampling="every stride-th existing domain pixel",
                  ownership=ownership_report, pooled=summaries, initial_state=dry_records)
    prefix = RESULTS / f"mapping_ownership_audit_s{args.stride}"
    scores.to_csv(prefix.with_suffix(".csv"), index=False)
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
