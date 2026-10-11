"""
Rainfall v1.1 = v1 point forecast unchanged + a CALIBRATED heavy-hour alert threshold.

Problem (Chroma manual-0130): v1 issues a heavy-hour alert when q50_final > the district heavy threshold
(train p99.5). That gives POD 0.74-0.90 but FAR 0.57-0.64, so CSI (0.33-0.40) is BELOW persistence
(0.40-0.57) in all 4 districts.

Fix (standard categorical-forecast post-processing, Wilks 2011 Statistical Methods in the Atmospheric
Sciences, ch. 8: the decision threshold of a continuous forecast is a free parameter chosen on independent
data, e.g. to maximise CSI or to give frequency bias ~1):
  * inner split: fit the stacked v1 model on < 2022, predict 2022-2023 (validation; the cached CNN feature
    for all rows < 2024 is out-of-fold, so no leakage);
  * choose tau on validation: arm "csi" = argmax CSI; arm "fb1" = frequency bias closest to 1;
  * apply tau to the 2024+ test predictions of the frozen v1 model (trained < 2024) and score against
    persistence with the same event definition (observed > heavy threshold).
Pre-registered (written before running): v1.1 adopted if its heavy-hour CSI >= persistence CSI in >= 3 of
4 districts on 2024+; the point forecast and its MSE skill are unchanged by construction.

Run: python -u scripts/125_v1_1_heavy_threshold.py all
Out: data/rainfall_nowcast/v1_1_heavy_threshold_summary.csv
"""
import importlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m = importlib.import_module("20_base_rainfall")
OUT = os.path.join(m.ROOT, "data", "rainfall_nowcast")
BASE_FEATS = list(m.FEATURE_COLS)
DISTRICTS = ["Sivasagar", "Marigaon", "Nagaon", "Karimganj"]


def cont(o, f):
    h = int((o & f).sum()); mi = int((o & ~f).sum()); fa = int((~o & f).sum())
    return {"POD": h / (h + mi) if h + mi else np.nan, "FAR": fa / (h + fa) if h + fa else np.nan,
            "CSI": h / (h + mi + fa) if h + mi + fa else np.nan, "FB": (h + fa) / (h + mi) if h + mi else np.nan}


def run(d):
    raw = m.load_hourly(d)
    df = m.build_features(raw, d)
    heavy = m.compute_heavy_threshold(df[df["datetime"] < pd.Timestamp(m.TEST_SPLIT_DATE)])
    cnn = pd.read_parquet(os.path.join(OUT, f"cnn6_oof_features_{d.lower()}.parquet")).reset_index() \
        .rename(columns={"index": "datetime"})
    df = df.merge(cnn[["datetime", "cnn_h1"]], on="datetime", how="left").rename(columns={"cnn_h1": "cnn_pred"})
    df = df.dropna(subset=["tp_mm", "cnn_pred"]).reset_index(drop=True)
    df["y_occurrence"] = (df["tp_mm"] > m.RAIN_THR_MM).astype(int)
    tr = df[df["datetime"] < "2022-01-01"].reset_index(drop=True)
    va = df[(df["datetime"] >= "2022-01-01") & (df["datetime"] < "2024-01-01")].reset_index(drop=True)
    m.FEATURE_COLS = BASE_FEATS + ["cnn_pred"]
    out_va, _ = m.run_split(tr, va, heavy)
    m.FEATURE_COLS = BASE_FEATS
    yv, fv = out_va["tp_mm"].values, out_va["q50_final"].values
    grid = np.unique(np.quantile(fv[fv > 0.5], np.linspace(0, 1, 400)))
    rows = [{"tau": t, **cont(yv > heavy, fv > t)} for t in grid]
    g = pd.DataFrame(rows).dropna(subset=["CSI"])
    tau_csi = float(g.loc[g["CSI"].idxmax(), "tau"])
    tau_fb1 = float(g.loc[(g["FB"] - 1).abs().idxmin(), "tau"])
    te = pd.read_csv(os.path.join(OUT, f"v1_rolling1h_{d.lower()}_hourly.csv"))
    y, f, p = te["tp_mm"].values, te["q50_final"].values, te["persistence"].values
    res = []
    for arm, t in (("v1_default", heavy), ("v1.1_csi", tau_csi), ("v1.1_fb1", tau_fb1), ("persistence", heavy)):
        fc = (p if arm == "persistence" else f) > t
        res.append({"district": d, "arm": arm, "tau_mm_h": t, "heavy_thr_mm_h": heavy,
                    "n_obs_heavy": int((y > heavy).sum()), **cont(y > heavy, fc)})
    val = {"district": d, "arm": "validation_2022_23_best", "tau_mm_h": tau_csi, "heavy_thr_mm_h": heavy,
           "n_obs_heavy": int((yv > heavy).sum()), **cont(yv > heavy, fv > tau_csi)}
    r = pd.DataFrame(res + [val])
    print(r.round(3).to_string(index=False), flush=True)
    return r


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "all"
    r = pd.concat([run(d) for d in (DISTRICTS if a == "all" else [a])])
    r.to_csv(os.path.join(OUT, "v1_1_heavy_threshold_summary.csv"), index=False)
    t = r[r["arm"].isin(["v1.1_csi", "persistence"])].pivot(index="district", columns="arm", values="CSI")
    n = int((t["v1.1_csi"] >= t["persistence"]).sum())
    print(f"\nPRE-REGISTERED: v1.1_csi CSI >= persistence in {n}/{len(t)} districts -> "
          f"{'ADOPT v1.1' if n >= 3 else 'DO NOT ADOPT'}")
