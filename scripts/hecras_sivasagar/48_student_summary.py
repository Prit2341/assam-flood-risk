"""Student baseline summary vs the pre-registered rule (Chroma manual-0248/0252).
Reads surrogate/extent_<tag>_test.csv (28_extent_eval.py) for the listed runs, derives hits = csi*(T+S)/(1+csi), POD = hits/T, precision = hits/S,
and applies: ROBUST only if mean over the converged runs has POD >= 0.60 and area ratio <= 1.5 and every run has POD >= 0.55.
  python scripts/hecras_sivasagar/48_student_summary.py
"""
from pathlib import Path
import pandas as pd
S = Path(__file__).resolve().parents[2] / "digital_twin/sivasagar/surrogate"
runs = {"orig (direct_cont)": "direct_cont", "s1b (seed 1, bias init)": "direct_cont_s1b", "s2b (seed 2, bias init)": "direct_cont_s2b", "m1 (mild weights)": "direct_cont_m1", "f1 (sub-cell feats, seed 5)": "direct_cont_f1", "f2 (sub-cell feats, seed 6)": "direct_cont_f2"}
rows = []
for name, tag in runs.items():
    f = S / f"extent_{tag}_test.csv"
    if not f.exists():
        print("missing", f.name); continue
    d = pd.read_csv(f); d["hits"] = d.csi * (d.teacher_km2 + d.student_km2) / (1 + d.csi)
    d["pod"] = d.hits / d.teacher_km2.clip(lower=1e-9); d["prec"] = d.hits / d.student_km2.clip(lower=1e-9)
    rows.append(dict(run=name, POD=d.pod.mean(), precision=d.prec.mean(), CSI=d.csi.mean(), area_ratio=d.area_ratio.mean(), POD_share_ge_0_6=(d.pod >= 0.6).mean()))
df = pd.DataFrame(rows).round(3); print(df.to_string(index=False))
core = df[df.run.str.startswith(("orig", "s1b", "s2b"))]
if len(core) == 3:
    ok = core.POD.mean() >= 0.60 and core.area_ratio.mean() <= 1.5 and (core.POD >= 0.55).all()
    print(f"\nmean over orig, s1b, s2b: POD {core.POD.mean():.3f} (range {core.POD.min():.3f}-{core.POD.max():.3f}), ratio {core.area_ratio.mean():.2f}, CSI {core.CSI.mean():.3f} (range {core.CSI.min():.3f}-{core.CSI.max():.3f})")
    print("BASELINE ROBUST (pre-registered rule):", ok)
