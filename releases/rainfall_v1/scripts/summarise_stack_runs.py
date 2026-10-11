"""
Summarise the CNN-into-base stacking runs: seed-to-seed spread (Nagaon) and the
control-vs-stacked gain for every district that finished.
Usage: python scripts/summarise_stack_runs.py
"""
import glob
import os

import pandas as pd

D = "data/rainfall_nowcast"
pd.set_option("display.width", 250)
MET = ["corr_all", "rmse_all", "corr_rainy", "bias_rainy", "auc_gate_heavy", "heavy_cov", "heavy_IS"]


def gain(df):
    p = df.pivot(index="lead", columns="arm", values=MET)
    out = pd.DataFrame(index=p.index)
    for c in MET:
        out[f"{c}_ctrl"] = p[(c, "control")]
        out[f"{c}_stk"] = p[(c, "stacked")]
    return p, out


runs = {}
for f in sorted(glob.glob(os.path.join(D, "stack_cnn_base_*_summary*.csv"))):
    name = os.path.basename(f)[len("stack_cnn_base_"):-len(".csv")].replace("_summary", "")
    runs[name] = pd.read_csv(f)
print("runs found:", list(runs))

# seed-to-seed (Nagaon): the stacked-minus-control gain per seed
print("\n== Nagaon: stacked minus control, per seed (positive = better for corr, negative = better for rmse/IS)")
rows = []
for name in ("nagaon", "nagaon_s1"):
    if name in runs:
        p = runs[name].pivot(index="lead", columns="arm", values=MET)
        for c in ("corr_all", "corr_rainy", "rmse_all", "heavy_cov", "heavy_IS", "bias_rainy"):
            g = (p[(c, "stacked")] - p[(c, "control")]).round(3)
            rows.append(pd.DataFrame({"run": name, "metric": c, **{f"h{h}": g[h] for h in g.index}}, index=[0]))
if rows:
    print(pd.concat(rows).to_string(index=False))

# stacked minus control, every district
print("\n== corr_all gain (stacked - control) and heavy_cov gain, all runs")
tab = []
for name, df in runs.items():
    p = df.pivot(index="lead", columns="arm", values=["corr_all", "heavy_cov", "rmse_all"])
    tab.append(pd.DataFrame({
        "run": name,
        **{f"dcorr_h{h}": round(p[("corr_all", "stacked")][h] - p[("corr_all", "control")][h], 3) for h in p.index},
        "mean_dcorr": round((p[("corr_all", "stacked")] - p[("corr_all", "control")]).mean(), 3),
        "mean_dcov": round((p[("heavy_cov", "stacked")] - p[("heavy_cov", "control")]).mean(), 3),
        "mean_drmse_pct": round(100 * ((p[("rmse_all", "stacked")] / p[("rmse_all", "control")]) - 1).mean(), 1),
        "n_heavy": int(df["n_heavy"].iloc[0])}, index=[0]))
print(pd.concat(tab).to_string(index=False))
