"""
Step 9: run the gridded-rain project (hecras_c600_d4_grid, built by hand from hecras_c600_d4 + set_gridded_precipitation) for N hours from 2026-07-05 00:00.
The 2D-area 'Precipitation Hydrograph' boundary block was REMOVED from the u01 by hand (HEC-RAS refuses rain in both places); .u01 now has Met BC Precipitation Mode=Gridded only. Result copied to results/c600_d4_gridrain_r30_0705_<N>h_<tag>.p01.hdf
Run: python -u scripts/hecras_sivasagar/09_run_grid.py 48 test
"""
import shutil, sys, time
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import hecras_io as rio
from ras_commander import init_ras_project, RasCmdr
ROOT = HERE.parents[1]; h = ROOT / "digital_twin" / "sivasagar" / "hecras_c600_d4_grid"; RES = ROOT / "digital_twin" / "sivasagar" / "results"
hours, tag = int(sys.argv[1]), sys.argv[2]
t0 = pd.Timestamp("2026-07-05 00:00"); rio.set_simulation_window(h / "Sivasagar_2D.p01", t0, t0 + pd.Timedelta(hours=hours))
ras = init_ras_project(str(h), "7.0.1"); t = time.time()
ok = RasCmdr.compute_plan("01", ras_object=ras, verify=True, dialog_watchdog=False); wall = time.time() - t
out = RES / f"c600_d4_gridrain_r30_0705_{hours}h_{tag}.p01.hdf"; shutil.copy(h / "Sivasagar_2D.p01.hdf", out)
print("done", ok, round(wall, 1), out.name)
