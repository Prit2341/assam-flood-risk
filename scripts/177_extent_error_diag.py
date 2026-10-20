"""
Why does H5 over- (and under-) predict flooded extent on some dates? Descriptive diagnostic, no model change.
Uses the Amendment 4 H5 LOYO held-out predictions on table.parquet (2000 sampled cells per scene).
1. Per scene: observed flooded share, predicted (mean p), log ratio, and drivers (Q/Q90 tributary and main stem,
   rising/falling limb = Q / 7-day max Q, rain r1/r5/r30, soil moisture, month, satellite observed share).
2. Spearman of the log ratio with each driver over scenes where observed or predicted >= 2%.
3. For the 10 largest over-prediction scenes: where the excess sum(p - y) sits (by flood history ff, HAND, land cover).

Run:  python -u scripts/177_extent_error_diag.py
Out:  data/dynamic_susceptibility/diag/extent_error_{scenes,drivers,excess}.csv
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s173", os.path.join(HERE, "173_amendment4.py"))
a4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a4)
a3, dc = a4.a3, a4.dc
OUT = os.path.join(dc.DYN, "diag")
DW = {0: "water", 1: "trees", 2: "grass", 3: "flooded veg", 4: "crops", 5: "shrub", 6: "built", 7: "bare", 8: "snow"}


def main():
    os.makedirs(OUT, exist_ok=True)
    T = a3.a2.load_amended()
    T["p"] = np.load(os.path.join(dc.DYN, "amend4_h5_loyo_preds.npz"))["p"]
    T = T.dropna(subset=["p"])
    hist = {}
    for Y in sorted(T["year"].unique()):          # ff for each row from the test-year history files (years != Y)
        for d in dc.DISTRICTS:
            f = os.path.join(dc.DYN, "h5_history", f"{d}_{Y}.npz")
            m = ((T["year"] == Y) & (T["district"] == d)).values
            if m.any():
                hist.setdefault("ff", np.full(len(T), np.nan))[m] = np.load(f)["ff"].ravel()[T.loc[m, "cell"].values]
    T["ff"] = hist["ff"]
    obs = pd.concat([pd.read_csv(os.path.join(dc.DYN, "labels", d, "labels_index.csv")).assign(district=d)
                     for d in dc.DISTRICTS])
    obs["date"] = pd.to_datetime(obs["date"])
    S = T.groupby(["district", "date"]).agg(obs_pct=("y", "mean"), pred_pct=("p", "mean"), q_trib=("q_trib", "first"),
                                             q_main=("q_main", "first"), qt7=("q_trib_max7", "first"),
                                             qm7=("q_main_max7", "first"), r1=("r1", "mean"), r5=("r5", "mean"),
                                             r30=("r30", "mean"), sm1=("sm1", "mean")).reset_index()
    S = S.merge(obs[["district", "date", "obs_frac"]], on=["district", "date"], how="left")
    S[["obs_pct", "pred_pct"]] *= 100
    S["limb_trib"] = S["q_trib"] / S["qt7"]           # < 1: falling limb (flow below its 7-day max)
    S["limb_main"] = S["q_main"] / S["qm7"]
    S["month"] = S["date"].dt.month
    S["log_ratio"] = np.log10((S["pred_pct"] + 0.5) / (S["obs_pct"] + 0.5))
    S.to_csv(os.path.join(OUT, "extent_error_scenes.csv"), index=False)

    act = S[(S["obs_pct"] >= 2) | (S["pred_pct"] >= 2)]
    rows = []
    for d, g in [("all", act)] + list(act.groupby("district")):
        for c in ["q_trib", "q_main", "limb_trib", "limb_main", "r1", "r5", "r30", "sm1", "month", "obs_frac"]:
            rho = spearmanr(g[c], g["log_ratio"], nan_policy="omit").correlation
            rows.append({"district": d, "driver": c, "rho": rho, "n": len(g)})
    R = pd.DataFrame(rows).pivot(index="driver", columns="district", values="rho")
    R.to_csv(os.path.join(OUT, "extent_error_drivers.csv"))
    print(f"Spearman(log pred/obs, driver), {len(act)} active scenes:\n{R.round(2)}\n")

    act = act.assign(limb=np.where(act["limb_main"] < 0.97, "falling", "rising/peak"))
    print(act.groupby("limb")[["obs_pct", "pred_pct"]].agg(["mean", "size"]).round(2), "\n")

    top = act.assign(excess=act["pred_pct"] - act["obs_pct"]).nlargest(10, "excess")
    print("10 largest over-predictions:\n", top[["district", "date", "obs_pct", "pred_pct", "q_trib", "q_main",
                                                    "limb_main", "r5", "obs_frac"]].round(2).to_string(index=False))
    sel = T.merge(top[["district", "date"]], on=["district", "date"])
    sel["exc"] = sel["p"] - sel["y"]
    sel["ff_bin"] = pd.cut(sel["ff"], [-.01, 0, .02, .05, .1, .2, 1.01], labels=["0", "0-2%", "2-5%", "5-10%", "10-20%", ">20%"])
    sel["hand_bin"] = pd.cut(sel["hand"], [-1, 1, 2, 5, 1e4], labels=["<1m", "1-2m", "2-5m", ">5m"])
    sel["lc"] = sel["lulc"].map(DW)
    parts = []
    for b in ["ff_bin", "hand_bin", "lc"]:
        g = sel.groupby(b, observed=True).agg(cells=("p", "size"), pred=("p", "mean"), obs=("y", "mean"), excess=("exc", "sum"))
        g["share_of_excess"] = g["excess"] / sel["exc"].sum()
        g = g.reset_index().rename(columns={b: "bin"}).assign(by=b)
        parts.append(g)
        print(f"\nExcess by {b}:\n", g.drop(columns="by").round(3).to_string(index=False))
    pd.concat(parts).to_csv(os.path.join(OUT, "extent_error_excess.csv"), index=False)


if __name__ == "__main__":
    main()
