"""Calibration batch (rule s3): 3 channel depths x 3 runoff ratios, 2026-07-05 00:00 + 432 h, scored on the 2026-07-17 scene ONLY.
Held-out scenes (07-22, 08-20) are not scored here. Appends to digital_twin/sivasagar/results/calibration.csv."""
import subprocess, sys, importlib.util
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "digital_twin" / "sivasagar" / "results"
spec = importlib.util.spec_from_file_location("sc", HERE / "05_score.py"); sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
cell = 600
for depth in (4, 2, 8):
    for ratio in (0.3, 0.15, 0.5):
        name = f"c{cell}_d{depth}_r{int(ratio*100)}_0705_432h_cal"
        if not (RES / f"{name}.p01.hdf").exists():
            subprocess.run([sys.executable, "-u", str(HERE / "04_run_event.py"), "--cell", str(cell), "--depth", str(depth), "--ratio", str(ratio),
                            "--start", "2026-07-05 00:00", "--hours", "432", "--tag", "cal"], check=True)
        r = sc.score(RES / f"{name}.p01.hdf", "2026-07-17"); r.update(depth=depth, ratio=ratio)
        pd.DataFrame([r]).to_csv(RES / "calibration.csv", mode="a", header=not (RES / "calibration.csv").exists(), index=False)
        print(r, flush=True)
