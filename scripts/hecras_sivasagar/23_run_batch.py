"""Run the surrogate training events on the frozen teacher (hecras_c600_d4, runoff 0.3, cold start) and keep compact outputs.
Per event: water-surface elevation of the 26,038 real cells at every output hour (float32, T x N) + the applied hourly rain -> surrogate/runs/<id>.npz;
the HDF is deleted after extraction. Resumable (skips ids with an npz); appends results/batch_log.csv.
  python -u scripts/hecras_sivasagar/23_run_batch.py [max_events]"""
import sys, time
from pathlib import Path
import h5py, numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import hecras_io as rio
from ras_commander import init_ras_project, RasCmdr
import os
ROOT = HERE.parents[1]; D = ROOT / "digital_twin/sivasagar"; NAME = "Sivasagar_2D"; NREAL = 26038
# optional overrides (added 2026-11-06, defaults = frozen teacher): BATCH_PROJ (project folder), BATCH_RATIO, BATCH_RUNS (output subfolder of surrogate/)
PROJ = os.environ.get("BATCH_PROJ", "hecras_c600_d4"); RATIO = float(os.environ.get("BATCH_RATIO", "0.3")); RUNS = os.environ.get("BATCH_RUNS", "runs")
TS = "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series"
runs = D / "surrogate" / RUNS; runs.mkdir(parents=True, exist_ok=True)
ev = pd.read_csv(D / "surrogate/events.csv", parse_dates=["start"])
import glob
rain = pd.concat([pd.read_csv(f, parse_dates=["timestamp"]) for f in sorted(glob.glob(str(D / "data/era5land_domain_hourly_*.csv")))]).drop_duplicates("timestamp").set_index("timestamp").tp_mm.sort_index()
h = D / PROJ; ras = init_ras_project(str(h), "7.0.1")
todo = [r for _, r in ev.iterrows() if not (runs / f"{r.id}.npz").exists()][: int(sys.argv[1]) if len(sys.argv) > 1 else None]
print(len(todo), "events to run on", PROJ, "ratio", RATIO, "->", runs, flush=True)
for r in todo:
    t0 = r.start; t1 = t0 + pd.Timedelta(hours=int(r.hours)); seg = rain[t0:t1]; assert len(seg) == r.hours + 1
    vals = (seg.values * RATIO * r.scale).tolist()
    rio.set_simulation_window(h / f"{NAME}.p01", t0, t1); rio.rewrite_precipitation(h / f"{NAME}.u01", vals)
    t = time.time(); ok = RasCmdr.compute_plan("01", ras_object=ras, verify=True, dialog_watchdog=False); wall = time.time() - t
    good = bool(ok)
    if good:
        with h5py.File(h / f"{NAME}.p01.hdf", "r") as f:
            wse = f[f"{TS}/2D Flow Areas/{NAME.split('_')[0]}_Domain/Water Surface"][:, :NREAL].astype("float32")
            nt = wse.shape[0]
        np.savez_compressed(runs / f"{r.id}.npz", wse=wse, rain_mm=np.array(vals, "float32"), start=str(t0), scale=r.scale, ratio=RATIO)
    pd.DataFrame([dict(id=r.id, start=str(t0), hours=int(r.hours), scale=r.scale, split=r.split, wall_s=round(wall), ok=good, outputs=nt if good else 0)]).to_csv(
        D / f"results/batch_log_{RUNS}.csv" if RUNS != "runs" else D / "results/batch_log.csv", mode="a", header=True if RUNS != "runs" and not (D / f"results/batch_log_{RUNS}.csv").exists() else not (D / "results/batch_log.csv").exists(), index=False)
    print(r.id, t0, "scale", r.scale, "wall", round(wall), "ok", good, flush=True)
