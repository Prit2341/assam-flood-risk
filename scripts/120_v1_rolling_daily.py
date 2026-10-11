"""
Rainfall v1 (frozen, releases/rainfall_v1) evaluated as an HOURLY-UPDATED 1-hour-ahead prediction,
scored hourly and aggregated to daily totals, against reference forecasts.

WHY: Prit's framing (2026-10-09): at every hour the model gets the real observed inputs up to that hour
and predicts the next hour, so a whole day is covered by 24 successive 1-h predictions. An earlier
check used the base model only and reported daily corr ~0.99 with no baseline. A daily total of 1-h-ahead
predictions is nearly the observed total shifted by one hour, so ANY reasonable 1-h predictor
(persistence included) will score high there. Skill must be shown against reference forecasts.

Verification design (standard practice, cited):
  * Reference forecasts persistence and climatology, skill score SS = 1 - MSE_model / MSE_ref
    (Murphy 1988, Mon. Wea. Rev. 116:2417; Wilks 2011, Statistical Methods in the Atmospheric Sciences,
    ch. 8; WMO/WWRP JWGFVR forecast verification guidance).
  * Categorical scores for heavy hours / heavy days: POD, FAR, CSI from the 2x2 contingency table
    (Wilks 2011 ch. 8; Schaefer 1990, Wea. Forecasting 5:570 for CSI).
  * Daily categories use IMD's rainfall intensity classes (India Meteorological Department terminology:
    moderate 15.6-64.4, heavy 64.5-115.5 mm/day).
  * Heavy hour = above the district's train p99.5 rainy-hour threshold (same as v1, 20_base_rainfall.py).

v1 = base model + out-of-fold CNN feature (cached cnn6_oof_features_<d>.parquet, built by script 70),
stacked arm, lead 1. Nothing is retrained differently from script 70: same features, same split
(train < 2024, test >= 2024), same run_split.

Run: python -u scripts/120_v1_rolling_daily.py <District>|all
Out: data/rainfall_nowcast/v1_rolling1h_<d>_hourly.csv, v1_rolling_daily_summary.csv
"""
import importlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m = importlib.import_module("20_base_rainfall")
ROOT = m.ROOT
OUT = os.path.join(ROOT, "data", "rainfall_nowcast")
TEST_START = pd.Timestamp(m.TEST_SPLIT_DATE)
BASE_FEATS = list(m.FEATURE_COLS)
DISTRICTS = ["Sivasagar", "Marigaon", "Nagaon", "Karimganj"]
IMD_MODERATE, IMD_HEAVY = 15.6, 64.5


def cont(obs_evt, fc_evt):
    h = int((obs_evt & fc_evt).sum()); mi = int((obs_evt & ~fc_evt).sum()); fa = int((~obs_evt & fc_evt).sum())
    return {"n_obs": int(obs_evt.sum()), "hits": h,
            "POD": h / (h + mi) if h + mi else np.nan,
            "FAR": fa / (h + fa) if h + fa else np.nan,
            "CSI": h / (h + mi + fa) if h + mi + fa else np.nan}


def ss(y, f, ref):
    return 1 - np.mean((y - f) ** 2) / np.mean((y - ref) ** 2)


def run(district):
    raw = m.load_hourly(district)
    df = m.build_features(raw, district)
    heavy_thr = m.compute_heavy_threshold(df[df["datetime"] < TEST_START])
    cnn = pd.read_parquet(os.path.join(OUT, f"cnn6_oof_features_{district.lower()}.parquet"))
    cnn = cnn.reset_index().rename(columns={"index": "datetime"})
    df = df.merge(cnn[["datetime", "cnn_h1"]], on="datetime", how="left").rename(columns={"cnn_h1": "cnn_pred"})
    df = df.dropna(subset=["tp_mm", "cnn_pred"]).reset_index(drop=True)
    df["y_occurrence"] = (df["tp_mm"] > m.RAIN_THR_MM).astype(int)
    tr = df[df["datetime"] < TEST_START].reset_index(drop=True)
    te = df[df["datetime"] >= TEST_START].reset_index(drop=True)

    m.FEATURE_COLS = BASE_FEATS + ["cnn_pred"]
    out, _ = m.run_split(tr, te, heavy_thr)
    m.FEATURE_COLS = BASE_FEATS
    out = out.reset_index(drop=True)
    out["persistence"] = te["tp_lag1h"].values
    clim = tr.assign(mo=tr["datetime"].dt.month, hr=tr["datetime"].dt.hour).groupby(["mo", "hr"])["tp_mm"].mean()
    out["climatology"] = clim.reindex(list(zip(out["datetime"].dt.month, out["datetime"].dt.hour))).values
    out = out.dropna(subset=["persistence", "climatology"])
    out[["datetime", "tp_mm", "q50_final", "q10_ens", "q90_ens", "persistence", "climatology"]].to_csv(
        os.path.join(OUT, f"v1_rolling1h_{district.lower()}_hourly.csv"), index=False)

    y = out["tp_mm"].values
    rows = []
    for name in ("q50_final", "persistence", "climatology"):
        f = out[name].values
        r = {"district": district, "scale": "hourly", "forecast": name,
             "corr": np.corrcoef(y, f)[0, 1], "rmse": float(np.sqrt(np.mean((y - f) ** 2))),
             "bias_pct": 100 * (f.sum() / y.sum() - 1),
             "SS_vs_persistence": ss(y, f, out["persistence"].values),
             "SS_vs_climatology": ss(y, f, out["climatology"].values)}
        r.update({f"heavy_{k}": v for k, v in cont(y > heavy_thr, f > heavy_thr).items()})
        rows.append(r)

    out["day"] = out["datetime"].dt.floor("D")
    g = out.groupby("day").agg(n=("tp_mm", "size"), obs=("tp_mm", "sum"), q50_final=("q50_final", "sum"),
                               persistence=("persistence", "sum"), climatology=("climatology", "sum"))
    g = g[g["n"] == 24]
    yd = g["obs"].values
    for name in ("q50_final", "persistence", "climatology"):
        f = g[name].values
        r = {"district": district, "scale": "daily", "forecast": name, "n_days": len(g),
             "corr": np.corrcoef(yd, f)[0, 1], "rmse": float(np.sqrt(np.mean((yd - f) ** 2))),
             "bias_pct": 100 * (f.sum() / yd.sum() - 1),
             "SS_vs_persistence": ss(yd, f, g["persistence"].values),
             "SS_vs_climatology": ss(yd, f, g["climatology"].values)}
        r.update({f"modplus_{k}": v for k, v in cont(yd >= IMD_MODERATE, f >= IMD_MODERATE).items()})
        r.update({f"imdheavy_{k}": v for k, v in cont(yd >= IMD_HEAVY, f >= IMD_HEAVY).items()})
        rows.append(r)
    res = pd.DataFrame(rows)
    res["heavy_thr_mm_h"] = heavy_thr
    print(res.round(3).to_string(index=False), flush=True)
    return res


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    ds = DISTRICTS if arg == "all" else [arg]
    res = pd.concat([run(d) for d in ds])
    res.to_csv(os.path.join(OUT, "v1_rolling_daily_summary.csv" if arg == "all"
                            else f"v1_rolling_daily_{arg.lower()}.csv"), index=False)
