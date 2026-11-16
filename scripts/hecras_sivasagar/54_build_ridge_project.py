"""Build isolated A2 ridge-aligned mesh; never overwrite baseline projects.

Registered in docs/teacher_improvement_review_2026-11-16.md.
Actual terrain heights, plan physics and rainfall are copied unchanged.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import shutil
import geopandas as gpd
import h5py
import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import LineString
from ras_commander import init_ras_project
from ras_commander.geom import GeomStorage, GeomMesh
from ras_commander.RasMap import RasMap

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "digital_twin/sivasagar"
AREA = "Sivasagar_Domain"
NAME = "Sivasagar_2D"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="resume this experiment's existing clone without appending breaklines")
    args = ap.parse_args()
    src = BASE / "hecras_c600_d4_raw"
    dst = BASE / "hecras_c600_d4_ridges_a2"
    if dst.exists() and not args.resume:
        raise FileExistsError(f"Refusing to replace existing experimental project: {dst}")
    ridges = gpd.read_file(BASE / "data/ridge_a2/ridge_candidates.gpkg")
    if ridges.crs.to_epsg() != 32646 or len(ridges) == 0:
        raise ValueError("Missing ridge candidates or incorrect CRS")
    if not ridges.is_valid.all() or not ridges.is_simple.all():
        raise ValueError("Ridge candidates have invalid/self-crossing geometry")
    if not args.resume:
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
            "*.p01.hdf", "*.b01", "*.bco01", "*.ic.o01", "*.x01", "*.log"))
    gfile = dst / f"{NAME}.g01"
    breaks = [dict(name=f"A2_ridge_{i:02d}", coords=list(row.geometry.coords),
                   cell_size_near=150, cell_size_far=600)
              for i, row in ridges.iterrows()]
    if not args.resume:
        GeomStorage.set_breaklines(gfile, AREA, breaks)
    elif gfile.read_text().count("BreakLine Name=A2_ridge_") != len(breaks):
        raise ValueError("Resume clone does not have the expected A2 breaklines")
    ras = init_ras_project(str(dst), "7.0.1")
    spec = importlib.util.spec_from_file_location("setup_sivasagar", Path(__file__).with_name("03_setup_project.py"))
    setup = importlib.util.module_from_spec(spec); spec.loader.exec_module(setup)
    setup.patch_reseed(ras)
    res = GeomMesh.generate(str(gfile), mesh_name=AREA, cell_size=600,
                           recompile_via_rasexe=True, ras_object=ras)
    print("mesh", res, flush=True)
    if res.status != "complete":
        raise RuntimeError("Ridge-aligned mesh generation did not complete")
    RasMap.set_geometry_association("01", terrain_hdf_path=dst / "Terrain/Terrain.hdf",
                                   ras_object=ras, validate=False)
    with h5py.File(dst / f"{NAME}.g01.hdf", "r") as f:
        g = f[f"Geometry/2D Flow Areas/{AREA}"]
        coords = g["FacePoints Coordinate"][:]
        edge_ids = g["Faces FacePoint Indexes"][:]
        edges = [LineString(coords[e]) for e in edge_ids]
    samples = np.array([line.interpolate(t).coords[0] for line in ridges.geometry
                        for t in np.arange(0, line.length, 30)])
    pts = shapely.points(samples)
    edge_array = np.array(edges, dtype=object)
    nearest = STRtree(edges).nearest(pts)
    distance = shapely.distance(pts, edge_array[nearest])
    report = dict(project=str(dst), cell_count=res.cell_count, face_count=res.face_count,
                  ridge_count=len(ridges), ridge_km=float(ridges.length.sum()/1000),
                  base_cell_m=600, near_cell_m=150, far_cell_m=600,
                  alignment_fraction_within_30m=float(np.mean(distance <= 30)),
                  alignment_distance_median_m=float(np.median(distance)),
                  alignment_distance_p95_m=float(np.percentile(distance, 95)),
                  terrain_changed=False)
    (BASE / "data/ridge_a2/mesh_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if report["alignment_fraction_within_30m"] < .9:
        raise RuntimeError("Breaklines not adequately enforced; do not run physical test")


if __name__ == "__main__":
    main()
