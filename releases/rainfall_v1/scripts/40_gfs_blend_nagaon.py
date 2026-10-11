"""
GFS-blend layer for the Nagaon rainfall model -- point forecast AND heavy-rain
coverage, with an out-of-sample fitting period.

WHY THIS DESIGN (see Chroma manual-0065 + advisor review 2026-09-19):
  * GFS archive starts 2021-05-01. Base model is trained on ERA5 2000-2023, so
    its 2021-2023 predictions are IN-SAMPLE and would make the model look
    better than it is, under-weighting GFS. So ONE base model is trained on
    2000-01 .. 2021-04 and used for BOTH stages (blend-fit and final test) --
    no two-model mismatch.
      base-train : datetime <  2021-05-01   (ERA5 only)
      blend-fit  : 2021-05-01 <= datetime < 2024-01-01  (base OUT-of-sample)
      test       : datetime >= 2024-01-01    (same window as existing results)
  * Three arms scored on the SAME test rows, so the GFS effect is isolated
    from the effect of merely recalibrating on the fit period:
      base : base model as-is (its own q10_ens/q90_ens interval)
      ctrl : recalibrated on the fit period, NO GFS
      gfs  : recalibrated on the fit period, WITH GFS
  * Timestamps: GFS precipitation_surface = "average rate since the previous
    forecast step" (dataset attrs); GEE ERA5 total_precipitation = "1 hour
    ending at the validity time" (GEE docs). Both end-of-hour -> GFS value at
    valid time V pairs with ERA5 stamp V (offset 0). --shift 1 = sensitivity.
  * Lead-dependent linear blend of the point forecast (few parameters, per
    Imhoff et al. 2023 whose blend weights vary by lead time). Heavy events:
    logistic gate2 on [logit(base gate prob), GFS rain, GFS precipitable
    water]; intervals for gated rows rebuilt with the SAME CSG-Gamma + local
    GPD mean-of-2 construction as 20_base_rainfall.py, conditioned on the
    arm's own point forecast.
  * lagH = GFS run must be >= H hours old at issue time (delivery delay).
    5h is an ASSUMPTION, not verified in this project.

Run (project root):  python -u scripts/40_gfs_blend_nagaon.py [--shift 0|1]
"""
import os
import sys
import importlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m = importlib.import_module("20_base_rainfall")
ROOT = m.ROOT

DISTRICT = "Nagaon"
BASE_TRAIN_END = pd.Timestamp("2021-05-01")
FIT_END = pd.Timestamp("2024-01-01")
LEADS = [1, 2, 3, 4, 5, 6]
LAGS = [0, 5]
GFS_FILES = [
    "data/gfs_test/nagaon_gfs_2021-05-01_2023-12-31.parquet",
    "data/gfs_test/nagaon_gfs_2024-01-01_2026-03-31.parquet",
]
OUT_CSV = "data/rainfall_nowcast/gfs_blend_nagaon_summary.csv"


def load_gfs():
    g = pd.concat([pd.read_parquet(os.path.join(ROOT, f)) for f in GFS_FILES], ignore_index=True)
    g["gfs_mm"] = g["precipitation_surface"] * 3600.0
    g = g.drop_duplicates(["init_time", "lead_h"]).set_index(["init_time", "lead_h"]).sort_index()
    return g[["gfs_mm", "precipitable_water_atmosphere"]].rename(
        columns={"precipitable_water_atmosphere": "gfs_pw"})


def add_gfs(d, g, h, lag, shift):
    """GFS rain/PW valid at issue+h+shift from the latest init <= issue-lag."""
    issue = d["datetime"] - pd.Timedelta(hours=1)
    init = (issue - pd.Timedelta(hours=lag)).dt.floor("6h")
    lead = (issue + pd.Timedelta(hours=h + shift) - init) / pd.Timedelta(hours=1)
    key = pd.MultiIndex.from_arrays([init.values, lead.astype(int).values])
    vals = g.reindex(key)
    d = d.copy()
    d["gfs_mm"] = vals["gfs_mm"].values
    d["gfs_pw"] = vals["gfs_pw"].values
    return d


def point_metrics(actual, pred):
    rainy = actual > m.RAIN_THR_MM
    return {
        "corr_all": np.corrcoef(actual, pred)[0, 1],
        "rmse_all": float(np.sqrt(mean_squared_error(actual, pred))),
        "corr_rainy": np.corrcoef(actual[rainy], pred[rainy])[0, 1],
        "bias_rainy": float((pred[rainy].mean() - actual[rainy].mean()) / actual[rainy].mean()),
    }


def build_interval(d_fit, d_te, point_fit, point_te, gated_te, heavy_thr):
    """Same construction as 20_base_rainfall.run_split heavy branch, but the
    CSG-Gamma heavy fit is done on the fit-period rows (out-of-sample base)."""
    heavy_fit = d_fit["tp_mm"].values > heavy_thr
    rain_fit = d_fit["tp_mm"].values > m.RAIN_THR_MM
    fit_mask = heavy_fit & rain_fit
    light_mask = (~heavy_fit) & rain_fit
    shape_l, lam_l = m.fit_csg_gamma(np.clip(point_fit[light_mask], 0, None), d_fit["tp_mm"].values[light_mask])
    if fit_mask.sum() >= 10:
        shape_h, lam_h = m.fit_csg_gamma(np.clip(point_fit[fit_mask], 0, None), d_fit["tp_mm"].values[fit_mask])
    else:
        shape_h, lam_h = shape_l, lam_l
    p = np.clip(d_te["p_rain"].values, 1e-4, 1 - 1e-4)
    q10 = d_te["q10_mm"].values.copy()
    q90 = d_te["q90_mm"].values.copy()
    if gated_te.sum() > 0:
        q10_g, q90_g = m.csg_quantiles_gamma(0.10, 0.90, p[gated_te], shape_h, lam_h, np.clip(point_te[gated_te], 0, None))
        gs, gc = d_te["local_gpd_shape"].iloc[0], d_te["local_gpd_scale"].iloc[0]
        if np.isfinite(gs):
            q10_p, q90_p = m.gpd_quantiles(0.10, 0.90, p[gated_te], gs, gc, heavy_thr)
        else:
            q10_p, q90_p = q10_g, q90_g
        q10[gated_te] = (q10_g + q10_p) / 2.0
        q90[gated_te] = (q90_g + q90_p) / 2.0
    return q10, np.maximum(q90, point_te)


def heavy_cov(actual, q10, q90, heavy_thr):
    hv = actual > heavy_thr
    if hv.sum() == 0:
        return float("nan"), float("nan"), 0
    return (float(np.mean((actual[hv] >= q10[hv]) & (actual[hv] <= q90[hv]))),
            float(np.mean(q90[hv] - q10[hv])), int(hv.sum()))


def main():
    shift = int(sys.argv[sys.argv.index("--shift") + 1]) if "--shift" in sys.argv else 0
    print(f"GFS timestamp shift: {shift}h", flush=True)
    g = load_gfs()

    raw = m.load_hourly(DISTRICT)
    df = m.build_features(raw, DISTRICT)
    train1 = df[df["datetime"] < BASE_TRAIN_END]
    heavy_thr = m.compute_heavy_threshold(train1)
    print(f"Base-train rows (h=1): {len(train1):,}  heavy threshold (base-train p99.5): {heavy_thr:.2f} mm/h",
          flush=True)

    rows = []
    for h in LEADS:
        dfh = df.copy()
        dfh["tp_mm"] = df["tp_mm"].shift(-(h - 1))
        dfh = dfh.dropna(subset=["tp_mm"]).reset_index(drop=True)
        dfh["y_occurrence"] = (dfh["tp_mm"] > m.RAIN_THR_MM).astype(int)
        tr = dfh[dfh["datetime"] < BASE_TRAIN_END].reset_index(drop=True)
        te_all = dfh[dfh["datetime"] >= BASE_TRAIN_END].reset_index(drop=True)
        out, models = m.run_split(tr, te_all, heavy_thr)
        gate_p = models[2].predict_proba(te_all[m.FEATURE_COLS])[:, 1]
        out = out.reset_index(drop=True)
        out["gate_p"] = gate_p
        out["gated"] = out["gated"].astype(bool)
        assert (out["datetime"].values == te_all["datetime"].values).all()
        print(f"h={h}: base trained on {len(tr):,} rows, scored on {len(te_all):,}", flush=True)

        for lag in LAGS:
            d = add_gfs(out, g, h, lag, shift)
            d = d.dropna(subset=["gfs_mm", "gfs_pw"]).reset_index(drop=True)
            fit = d[d["datetime"] < FIT_END].reset_index(drop=True)
            te = d[d["datetime"] >= FIT_END].reset_index(drop=True)
            a_te, a_fit = te["tp_mm"].values, fit["tp_mm"].values

            # ---- point forecast: linear blend, fit on fit-period (out-of-sample base)
            Xf = np.c_[np.ones(len(fit)), fit["q50_final"], fit["gfs_mm"]]
            beta = np.linalg.lstsq(Xf, a_fit, rcond=None)[0]
            pt_gfs_fit = np.clip(Xf @ beta, 0, None)
            pt_gfs_te = np.clip(np.c_[np.ones(len(te)), te["q50_final"], te["gfs_mm"]] @ beta, 0, None)
            pt_base_te = te["q50_final"].values
            pt_base_fit = fit["q50_final"].values

            # ---- heavy gate2: logistic on [logit(base gate prob), GFS rain, GFS PW]
            def gfeat(z, use_gfs):
                lg = np.log(np.clip(z["gate_p"], 1e-4, 1 - 1e-4) / (1 - np.clip(z["gate_p"], 1e-4, 1 - 1e-4)))
                cols = [lg] + ([z["gfs_mm"], z["gfs_pw"]] if use_gfs else [])
                return np.c_[tuple(np.asarray(c) for c in cols)]
            yf = (a_fit > heavy_thr).astype(int)
            yt = (a_te > heavy_thr).astype(int)
            gate2 = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
            gate2.fit(gfeat(fit, True), yf)
            s_gfs = gate2.predict_proba(gfeat(te, True))[:, 1]
            s_base = te["gate_p"].values
            frac = te["gated"].mean()                       # same gated share as base gate
            k = max(1, int(round(frac * len(te))))
            gated_gfs = np.zeros(len(te), bool)
            gated_gfs[np.argsort(-s_gfs)[:k]] = True
            gated_base = te["gated"].values

            # ---- intervals per arm
            q10_b, q90_b = te["q10_ens"].values, te["q90_ens"].values                     # base as-is
            q10_c, q90_c = build_interval(fit, te, pt_base_fit, pt_base_te, gated_base, heavy_thr)  # ctrl
            # gfs arm: gate2 chooses which rows get the wide heavy interval
            fit_g = fit.copy()
            q10_g, q90_g = build_interval(fit_g, te, pt_gfs_fit, pt_gfs_te, gated_gfs, heavy_thr)

            arms = {"base": (pt_base_te, q10_b, q90_b, gated_base),
                    "ctrl": (pt_base_te, q10_c, q90_c, gated_base),
                    "gfs": (pt_gfs_te, q10_g, q90_g, gated_gfs)}
            for arm, (pt, q10, q90, gated) in arms.items():
                cov, wid, nh = heavy_cov(a_te, q10, q90, heavy_thr)
                pm = point_metrics(a_te, pt)
                rows.append({"lag_h": lag, "lead": h, "arm": arm, **pm,
                             "heavy_cov": cov, "heavy_width_mm": wid, "n_heavy": nh,
                             "heavy_recall_at_gate": float(gated[yt == 1].mean()) if nh else np.nan,
                             "gated_share": float(gated.mean()),
                             "auc_heavy": (roc_auc_score(yt, {"base": s_base, "ctrl": s_base, "gfs": s_gfs}[arm])
                                           if 0 < nh < len(yt) else np.nan),
                             "n_fit": len(fit), "n_test": len(te),
                             "blend_a": beta[0], "blend_b_model": beta[1], "blend_c_gfs": beta[2]})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(ROOT, OUT_CSV), index=False)
    pd.set_option("display.width", 250)
    cols = ["lag_h", "lead", "arm", "corr_all", "rmse_all", "corr_rainy", "bias_rainy",
            "heavy_cov", "heavy_width_mm", "heavy_recall_at_gate", "auc_heavy", "n_heavy"]
    print(res[cols].round(3).to_string(index=False))
    print("Saved", OUT_CSV)


if __name__ == "__main__":
    main()
