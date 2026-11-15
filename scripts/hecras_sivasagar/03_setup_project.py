"""
Sivasagar HEC-RAS teacher, step 3: build a headless HEC-RAS 2D project (pattern of Arunachal setup_project_namsai.py).

  python scripts/hecras_sivasagar/03_setup_project.py --cell 500 --depth 4          # new project hecras_c500_d4
  python scripts/hecras_sivasagar/03_setup_project.py --cell 500 --depth 8 --from-depth 4   # copy mesh, swap terrain only

Stages: project (template) -> terrain (DEM variant dem_burn_<d>m.tif or dem_filled.tif for depth 0) -> plan+flow files ->
perimeter + outlet BC line -> mesh. Uniform Manning n, no infiltration (decision D7); runoff given as rain x ratio (step 4).
The 'swap terrain' path copies a finished project (mesh included) and rebuilds only Terrain.hdf; HEC-RAS recomputes the
property tables at run time when the terrain is newer than g01.hdf (verified in the Namsai channel test, 2026-10-31).
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hecras_io as rio  # noqa: E402

from ras_commander import init_ras_project  # noqa: E402
from ras_commander.RasPrj import create_project_from_template  # noqa: E402
from ras_commander.RasMap import RasMap  # noqa: E402
from ras_commander.terrain.RasTerrain import RasTerrain  # noqa: E402
from ras_commander.geom import GeomStorage, GeomBcLines, GeomMesh  # noqa: E402

ROOT = HERE.parents[1]
DATA = ROOT / "digital_twin" / "sivasagar" / "data"
CRS = 32646
RAS_VERSION = "7.0.1"
NAME = "Sivasagar_2D"
AREA = "Sivasagar_Domain"
OUTLET_BC = "Outlet_West"
MANNING_N = 0.04          # baseline uniform, not calibrated
SIM_HOURS = 24            # placeholder window; 04_run_event.py rewrites window and rain
SIM_START = "2026-07-05 00:00"


def dirs(cell, depth):
    d = ROOT / "digital_twin" / "sivasagar" / f"hecras_c{cell}_d{depth}"
    return d, d / f"{NAME}.prj", d / f"{NAME}.g01", d / "Terrain"


def dem_for(depth):
    return DATA / ("dem_filled.tif" if depth == 0 else f"dem_burn_{depth}m.tif")


def make_terrain(hdir, tdir, depth):
    hdf = RasTerrain.create_terrain_from_rasters(input_rasters=[dem_for(depth)], output_folder=tdir, terrain_name="Terrain",
                                                 units="Meters", hecras_version=RAS_VERSION)
    return hdf


def patch_reseed(ras_obj):
    """ras-commander 0.99.1 stub fix (Arunachal setup_geometry._patch_reseed_after_perimeter_fix, Chroma manual-0062)."""
    from ras_commander.geom import GeomMesh as GM
    g = GM.generate.__wrapped__.__globals__
    _patch_text_perimeter, _recompile, _seeds = g["_patch_text_perimeter"], g["_recompile_geometry_hdf_via_rasexe"], g["_generate_seeds_via_net"]

    def real(text_path, hdf_path, current_perim, cell_size, fid, mesh_name, ns, hecras_dir=None):
        _patch_text_perimeter(text_path, current_perim, mesh_name=mesh_name)
        _recompile(text_path, hdf_path, ras_object=ras_obj)
        return _seeds(str(hdf_path), ns, fid=fid)

    g["_reseed_after_perimeter_fix"] = real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, required=True)
    ap.add_argument("--depth", type=int, required=True, help="channel burn depth m (0 = none)")
    ap.add_argument("--simplify", type=float, default=None, help="perimeter simplify tolerance m (default 100 if cell<=300 else 150)")
    ap.add_argument("--from-depth", type=int, default=None, help="copy this finished project and swap the terrain only")
    a = ap.parse_args()
    hdir, prj, g01, tdir = dirs(a.cell, a.depth)
    outlet = json.loads((DATA / "outlet.json").read_text())

    if a.from_depth is not None:
        src, *_ = dirs(a.cell, a.from_depth)
        if hdir.exists():
            shutil.rmtree(hdir)
        shutil.copytree(src, hdir, ignore=shutil.ignore_patterns("*.p01.hdf", "*.u01.hdf", "*.log", "*.b01", "*.bco01", "*.ic.o01", "*.x01"))
        for f in tdir.glob("*"):
            f.unlink()
        make_terrain(hdir, tdir, a.depth)
        print(f"terrain swapped to depth {a.depth} m in {hdir}")
        return

    if hdir.exists():
        shutil.rmtree(hdir)
    create_project_from_template(hdir, NAME, version="7.0", target_crs=f"EPSG:{CRS}")
    txt = prj.read_text().replace("English Units", "SI Units")
    txt = txt.replace("Geom File=g01", "Current Plan=p01\nGeom File=g01\nUnsteady File=u01\nPlan File=p01", 1)
    prj.write_text(txt)

    hdf = make_terrain(hdir, tdir, a.depth)
    RasMap.add_terrain_layer(terrain_hdf=hdf, rasmap_path=hdir / f"{NAME}.rasmap", layer_name="Terrain",
                             projection_prj=tdir / "Projection.prj")
    p01, u01 = hdir / f"{NAME}.p01", hdir / f"{NAME}.u01"
    rio.write_plan_file(HERE / "templates" / "plan.p01.template", p01, title=NAME, short_id="Sivasagar", start=SIM_START,
                        end=str(np.datetime64(SIM_START) + np.timedelta64(SIM_HOURS, "h")).replace("T", " "), comp_interval="10SEC")
    rio.write_flow_file(HERE / "templates" / "flow.u01.template", u01, flow_area=AREA, outlet_bc=OUTLET_BC,
                        friction_slope=outlet["friction_slope"], n_hours=SIM_HOURS, title=f"{AREA}_Flow")

    raw = gpd.read_file(DATA / "boundary_utm46n.gpkg").geometry.iloc[0]
    geom = Polygon(raw.exterior).buffer(-a.cell / 2).buffer(a.cell / 2)
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda p: p.area)
    geom = geom.simplify(a.simplify or (100 if a.cell <= 300 else 150), preserve_topology=True)
    print(f"perimeter: {len(geom.exterior.coords)} vertices, {geom.area / 1e6:.1f} km2 vs {raw.area / 1e6:.1f} km2 raw")
    GeomStorage.set_2d_flow_area_perimeter(geom_file=str(g01), flow_area_name=AREA, geometry=geom,
                                           point_generation_data=(None, None, a.cell, a.cell))
    lines = g01.read_text().splitlines(keepends=True)
    lines = ["Storage Area Type= 1\n" if l.strip() == "Storage Area Type= 0" else l for l in lines]
    t = "".join(lines)
    t = re.sub(r"Storage Area Mannings=.*", f"Storage Area Mannings={MANNING_N}", t)
    g01.write_text(t)
    coords = np.array(list(geom.exterior.coords))
    c = int(np.argmin(np.hypot(coords[:, 0] - outlet["outlet_x"], coords[:, 1] - outlet["outlet_y"])))
    print("BC centre distance to outlet cell (m):", round(float(np.hypot(*(coords[c] - (outlet["outlet_x"], outlet["outlet_y"])))), 1))
    bc = [tuple(x) for x in coords[max(0, c - 3):c + 4]]
    GeomBcLines.add_bc_lines(geom_file=str(g01), lines=[{"name": OUTLET_BC, "storage_area": AREA, "coordinates": bc}], replace_existing=True)

    ras_obj = init_ras_project(str(hdir), RAS_VERSION)
    patch_reseed(ras_obj)
    for attempt in (1, 2):
        t0 = time.time()
        res = GeomMesh.generate(str(g01), recompile_via_rasexe=True, ras_object=ras_obj)
        print(f"mesh attempt {attempt}: {res.cell_count} cells, {res.face_count} faces, {time.time() - t0:.0f}s, status={res.status}, "
              f"fixes={getattr(res, 'fixes_applied', None)}", flush=True)
        if res.status == "complete":
            break
        RasMap.set_geometry_association("01", terrain_hdf_path=tdir / "Terrain.hdf", ras_object=ras_obj, validate=False)
    print("terrain association:", RasMap.get_geometry_association("01", ras_object=ras_obj)["terrain_raw_filename"])


if __name__ == "__main__":
    main()
