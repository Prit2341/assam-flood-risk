"""Run a cloned gridded-rain project (33_grid_rain_project.py) from 2026-07-05 00:00 for N hours.
  python -u scripts/hecras_sivasagar/34_run_grid_project.py imerg 432
Out: results/c600_d4_grid<name>_r30_0705_<N>h.p01.hdf ; wall time appended to results/run_log.csv"""
import shutil, sys, time
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import hecras_io as rio
from ras_commander import init_ras_project, RasCmdr
ROOT = HERE.parents[1]; name, hours = sys.argv[1], int(sys.argv[2])
h = ROOT / "digital_twin" / "sivasagar" / f"hecras_c600_d4_grid_{name}"; RES = ROOT / "digital_twin" / "sivasagar" / "results"
t0 = pd.Timestamp(sys.argv[3] if len(sys.argv) > 3 else "2026-07-05 00:00"); rio.set_simulation_window(h / "Sivasagar_2D.p01", t0, t0 + pd.Timedelta(hours=hours))
ras = init_ras_project(str(h), "7.0.1"); t = time.time()
ok = RasCmdr.compute_plan("01", ras_object=ras, verify=True, dialog_watchdog=False); wall = time.time() - t
out = RES / f"c600_d4_grid{name}_r30_{t0:%m%d}_{hours}h.p01.hdf"; shutil.copy(h / "Sivasagar_2D.p01.hdf", out)
pd.DataFrame([dict(name=out.stem, cell=600, depth=4, ratio=0.3, start=str(t0), hours=hours, wall_s=round(wall, 1), ok=ok)]).to_csv(
    RES / "run_log.csv", mode="a", header=False, index=False)
print("done", ok, round(wall, 1), out.name, out.stat().st_size)
