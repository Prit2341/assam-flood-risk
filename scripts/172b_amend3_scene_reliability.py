"""Amendment 3 (c), descriptive, added AFTER the results: scene-level reliability binned on a quantity that is not
the outcome. C2 conditions on observed flooded fraction (regression to the mean), so it cannot show over/under-
prediction. Here scenes are binned by (i) predicted scene mean (quintiles) and (ii) scene max(q_trib, q_main) quintiles."""
import importlib.util, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s172", os.path.join(HERE, "172_amendment3.py"))
a3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a3)
dc = a3.dc
T = a3.a2.load_amended()
T["p"] = np.load(os.path.join(dc.DYN, "amend3_h2_loyo_preds.npz"))["p"]
T["pc"] = np.load(os.path.join(dc.DYN, "amend3_h2_cal_preds.npz"))["p"]
T["qx"] = T[["q_trib", "q_main"]].max(axis=1)
G = T.groupby(["district", "date"]).agg(p=("p", "mean"), pc=("pc", "mean"), y=("y", "mean"), qx=("qx", "first")).reset_index()
rows = []
for d, g in G.groupby("district"):
    for by in ("p", "qx"):
        b = pd.qcut(g[by].rank(method="first"), 5, labels=False)
        for k, h in g.groupby(b):
            rows.append({"district": d, "binned_by": "pred_mean" if by == "p" else "Q/Q90", "quintile": int(k) + 1,
                         "n_scenes": len(h), "obs": h.y.mean(), "pred": h.p.mean(), "pred_cal": h.pc.mean(),
                         "ratio": h.p.mean() / h.y.mean(), "ratio_cal": h.pc.mean() / h.y.mean()})
R = pd.DataFrame(rows)
R.to_csv(os.path.join(dc.DYN, "eval_amend3_cal_scene_bins.csv"), index=False)
pd.set_option("display.width", 200)
print(R.round(4).to_string(index=False))
