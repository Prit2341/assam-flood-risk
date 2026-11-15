"""Clone a finished gridded-rain project and swap ONLY the terrain raster (03_setup_project.py --from-depth pattern; HEC-RAS recomputes
the cell/face property tables at run time when Terrain.hdf is newer than g01.hdf, verified in the Namsai channel test 2026-10-31).
  python -u scripts/hecras_sivasagar/36_swap_terrain.py hecras_c600_d4_grid_imerg hecras_c600_w4_grid_imerg dem_burn_w4m.tif"""
import shutil, sys, os, time
from pathlib import Path
from ras_commander.terrain.RasTerrain import RasTerrain
S = Path(__file__).resolve().parents[2] / "digital_twin" / "sivasagar"
src, dst, dem = S / sys.argv[1], S / sys.argv[2], S / "data" / sys.argv[3]
if dst.exists(): shutil.rmtree(dst)
shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.p01.hdf", "_compute_p01.log", "*.bco01", "*.ic.o01"))
for f in (dst / "Terrain").glob("*"): f.unlink()
hdf = RasTerrain.create_terrain_from_rasters(input_rasters=[dem], output_folder=dst / "Terrain", terrain_name="Terrain", units="Meters", hecras_version="7.0.1")
time.sleep(1); os.utime(hdf)        # make sure terrain is newer than g01.hdf
print("terrain", hdf, "g01.hdf mtime", time.ctime((dst / "Sivasagar_2D.g01.hdf").stat().st_mtime), "terrain mtime", time.ctime(Path(hdf).stat().st_mtime))
print(sorted(p.name for p in (dst / "Terrain").iterdir()))
