"""Exit-boundary test: same model as c600_d4 / runoff 30%, normal-depth exit slope x10 (0.00314 instead of 0.000314), 5 Jul 00:00 + 312 h (to 18 Jul 00:00), uniform domain-mean rain.
Project dir digital_twin/sivasagar/hecras_c600_d4_slope was copied from hecras_c600_d4 and its u01 'Friction Slope' edited by hand. Result: results/c600_d4_r30_0705_312h_slope10x.p01.hdf"""
import shutil, sys, time
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import hecras_io as rio
from ras_commander import init_ras_project, RasCmdr
ROOT = HERE.parents[1]; h = ROOT / "digital_twin" / "sivasagar" / "hecras_c600_d4_slope"; D = ROOT / "digital_twin" / "sivasagar"
t0 = pd.Timestamp("2026-07-05 00:00"); hours = 312
rain = pd.read_csv(D / "data" / "era5land_domain_hourly_2026-07-01_2026-08-26.csv", parse_dates=["timestamp"]).set_index("timestamp").tp_mm
vals = (rain[t0:t0 + pd.Timedelta(hours=hours)].values * 0.3).tolist(); assert len(vals) == hours + 1
rio.set_simulation_window(h / "Sivasagar_2D.p01", t0, t0 + pd.Timedelta(hours=hours)); rio.rewrite_precipitation(h / "Sivasagar_2D.u01", vals)
ras = init_ras_project(str(h), "7.0.1"); t = time.time()
ok = RasCmdr.compute_plan("01", ras_object=ras, verify=True, dialog_watchdog=False)
out = D / "results" / "c600_d4_r30_0705_312h_slope10x.p01.hdf"; shutil.copy(h / "Sivasagar_2D.p01.hdf", out); print("done", ok, round(time.time() - t, 1))
