"""
Sivasagar HEC-RAS teacher, step 4: run one event on a finished project.

  python scripts/hecras_sivasagar/04_run_event.py --cell 600 --depth 4 --ratio 0.3 --start "2026-07-05 00:00" --hours 72 [--tag t]

Rain (decisions D3, D5): domain-mean ERA5-Land hourly (02_pull_rain.py), multiplied by a constant runoff ratio (no infiltration in
HEC-RAS; ratio is the bracketed unknown). HEC-RAS convention (own pulse test 2026-10-31): value k covers the interval ENDING at stamp
t0 + k h and value 0 is unused, so the hydrograph is the ERA5 rows stamped t0 .. t0 + hours (hours + 1 values), no shift needed because
ERA5(-Land) hourly accumulations are stamped at the end of the hour (assumption; not independently verified here).
Result HDF is copied to digital_twin/sivasagar/results/ with a descriptive name; run time is logged to results/run_log.csv.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hecras_io as rio  # noqa: E402

from ras_commander import init_ras_project, RasCmdr  # noqa: E402

ROOT = HERE.parents[1]
DATA = ROOT / "digital_twin" / "sivasagar" / "data"
RES = ROOT / "digital_twin" / "sivasagar" / "results"
NAME = "Sivasagar_2D"
RAIN_CSV = DATA / "era5land_domain_hourly_2026-07-01_2026-08-26.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, required=True)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--hours", type=int, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--proj", default=None, help="project folder name under digital_twin/sivasagar (default hecras_c<cell>_d<depth>)")
    a = ap.parse_args()
    hdir = ROOT / "digital_twin" / "sivasagar" / (a.proj or f"hecras_c{a.cell}_d{a.depth}")
    t0 = pd.Timestamp(a.start)
    t1 = t0 + pd.Timedelta(hours=a.hours)
    rain = pd.read_csv(RAIN_CSV, parse_dates=["timestamp"]).set_index("timestamp").tp_mm
    seg = rain[t0:t1]
    assert len(seg) == a.hours + 1, (len(seg), a.hours + 1)
    vals = (seg.values * a.ratio).tolist()
    rio.set_simulation_window(hdir / f"{NAME}.p01", t0, t1)
    rio.rewrite_precipitation(hdir / f"{NAME}.u01", vals)
    print(f"rain applied: {sum(vals[1:]):.1f} mm over {a.hours} h (ratio {a.ratio}, raw {seg.values[1:].sum():.1f} mm, first value unused)")

    ras = init_ras_project(str(hdir), "7.0.1")
    t = time.time()
    ok = RasCmdr.compute_plan("01", ras_object=ras, verify=True, dialog_watchdog=False)
    wall = time.time() - t
    RES.mkdir(parents=True, exist_ok=True)
    name = f"c{a.cell}_d{a.depth}_r{int(a.ratio * 100)}_{t0:%m%d}_{a.hours}h{('_' + a.tag) if a.tag else ''}"
    out = RES / f"{name}.p01.hdf"
    shutil.copy(hdir / f"{NAME}.p01.hdf", out)
    with h5py.File(out, "r") as f:
        ts = f["Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time Date Stamp"].shape[0]
    row = dict(name=name, cell=a.cell, depth=a.depth, ratio=a.ratio, start=str(t0), hours=a.hours, wall_s=round(wall, 1), ok=ok, outputs=ts)
    log = RES / "run_log.csv"
    pd.DataFrame([row]).to_csv(log, mode="a", header=not log.exists(), index=False)
    print(row)


if __name__ == "__main__":
    main()
