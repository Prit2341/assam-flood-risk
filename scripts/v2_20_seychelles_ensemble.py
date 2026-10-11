"""
v2 step 2 -- the Seychelles production ensemble AS WRITTEN, evaluated against Assam v1.

Source (all read in full 2026-10-09): approach_b_nwp/110_model_b_26yr.py (hard gate),
101_production_ensemble.py (CSG-Gamma, Local GPD, RFA GPD), 111_adaptive_ensemble_26yr.py
(SLSQP weights + 4-member blend), 119_live_4model_ensemble.py (production blend), and
3D_approach/10_pinn_26yr.py (the PINN member, see v2_10_pinn_diagnostic.py).

COPIED AS WRITTEN, INCLUDING KNOWN BUGS (Chroma manual-0106/0107/0111):
  * point forecast q50_final = where(gated, ext_pred, q50_mm): HARD gate (v1 uses a soft gate).
  * p_rain isotonic-calibrated, fitted on in-sample training predictions.
  * light branch interval = CSG-Gamma(p_calib, point = q50_mm), not the quantile regressions.
  * heavy branch interval = SLSQP-weighted sum of CSG-Gamma, Local GPD, Regional GPD, PINN;
    weights >=0, sum to 1, minimise MSE of Q90 proxies on TRAINING heavy rows; the PINN proxy is
    1.5 x its point prediction and its test interval is 0.5x / 1.5x its point prediction.
  * Regional GPD: Hosking kappa passed unchanged as scipy `c` (sign bug) and discordancy
    (N/2)*Mahalanobis^2 (inflated ~40x) -- NOT fixed here, by instruction.
  * PINN member merged on the row datetime; it is a same-hour diagnostic model.
  * no CNN stack.

NECESSARY ADAPTATIONS (not choices): v1's split (train < 2024, test >= 2024) and 6 leads so v1 and
v2 are comparable (Seychelles used 8 walk-forward folds); district adaptive heavy threshold (not a
fixed 10 mm); Assam features/base code (20_base_rainfall.py, which is itself the Seychelles port).

v2.1 (fix 1, 2026-10-09): arms v2_1_pinn / v2_1_nopinn = the v2 arms with the SOFT-gate point forecast.
Intervals are identical to v2 by construction. Pre-registered fix-1 criteria in Chroma (v2.1 entry).

Arms per (district, lead), all on the SAME rows:
  v1_control   current base model, soft gate, Gamma+LocalGPD mean-of-2 interval
  v1_stacked   v1 production (base + out-of-fold CNN feature)
  v2_pinn      Seychelles as written, 4 members (with the PINN)
  v2_nopinn    Seychelles as written, 3 members (PINN removed; exposes same-hour PINN leakage)
v2 arms share the hard-gate point forecast.

PRE-REGISTERED CRITERIA (Chroma manual-0111), v2_pinn vs v1_stacked over 24 cells, BOTH required:
  (a) heavy-event interval score lower for v2 in >= 18/24 cells
  (b) overall 80% interval coverage closer to 0.80 for v2 in >= 18/24 cells
Secondary (report only): point forecast metrics, SLSQP weights, h=1 leak signature, v2_nopinn.

Run:  python -u scripts/v2_20_seychelles_ensemble.py Nagaon
      python -u scripts/v2_20_seychelles_ensemble.py summary
"""
import importlib
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m = importlib.import_module("20_base_rainfall")
rg = importlib.import_module("100_regional_gpd")
d90 = importlib.import_module("90_distill_features")
ROOT = m.ROOT
LEADS = [1, 2, 3, 4, 5, 6]
TEST_START = pd.Timestamp(m.TEST_SPLIT_DATE)
BASE_FEATS = list(m.FEATURE_COLS)
OUT_DIR = os.path.join(ROOT, "data", "rainfall_nowcast")
REG_DIR = os.path.join(ROOT, "data", "icechunk_regional_tp")


def optimize_weights(train_actual, preds_dict):
    """Seychelles 111: SLSQP, sum(w)=1, 0<=w<=1, minimise MSE(actual - P w)."""
    names = list(preds_dict.keys())
    P = np.column_stack([preds_dict[k] for k in names])
    obj = lambda w: np.mean((train_actual - P @ np.array(w)) ** 2)
    n = len(names)
    res = minimize(obj, np.ones(n) / n, method="SLSQP", bounds=tuple((0.0, 1.0) for _ in range(n)),
                   constraints=({"type": "eq", "fun": lambda w: np.sum(w) - 1.0}))
    return {names[i]: float(res.x[i]) for i in range(n)}


def interval_score(y, lo, hi, a=0.2):
    return (hi - lo) + (2 / a) * np.maximum(lo - y, 0) + (2 / a) * np.maximum(y - hi, 0)


def metrics(y, q50, lo, hi, thr, arm, district, h, extra=None):
    rn = y > m.RAIN_THR_MM
    hv = y > thr
    r = {"district": district, "lead": h, "arm": arm,
         "corr_all": np.corrcoef(y, q50)[0, 1],
         "rmse_all": float(np.sqrt(mean_squared_error(y, q50))),
         "corr_rainy": np.corrcoef(y[rn], q50[rn])[0, 1],
         "bias_rainy": float((q50[rn].mean() - y[rn].mean()) / y[rn].mean()),
         "overall_cov": float(np.mean((y >= lo) & (y <= hi))),
         "overall_width": float(np.mean(hi - lo)),
         "rainy_cov": float(np.mean((y[rn] >= lo[rn]) & (y[rn] <= hi[rn]))),
         "heavy_cov": float(np.mean((y[hv] >= lo[hv]) & (y[hv] <= hi[hv]))),
         "heavy_width": float(np.mean(hi[hv] - lo[hv])),
         "heavy_IS": float(np.mean(interval_score(y[hv], lo[hv], hi[hv]))),
         "n_heavy": int(hv.sum()), "n_test": len(y)}
    if extra:
        r.update(extra)
    return r


def seychelles_arm(tr, te, out, models, thr, reg_train, drop, with_pinn):
    """Return (q50_final_hard, q10_ens, q90_ens, weights) exactly as Seychelles 111/110/119 do."""
    occ, q_models, gate_clf, ext_reg = models
    feats = BASE_FEATS
    y = out["tp_mm"].values
    point_pred = out["q50_mm"].values
    gated = out["gated"].values.astype(bool)
    is_rain = out["p_rain"].values > 0.5

    # --- 110: hard gate point forecast ---
    ext_pred = np.zeros(len(te))
    if (tr["tp_mm"] > thr).sum() > 5:
        ext_pred = ext_reg.predict(te[feats])
    q50_hard = np.where(gated, ext_pred, point_pred)
    q50_hard = np.where(is_rain, np.clip(q50_hard, 0, None), 0.0)

    # --- 111: isotonic p_rain on in-sample training predictions ---
    p_tr_raw = occ.predict_proba(tr[feats])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True).fit(p_tr_raw, tr["y_occurrence"].values)
    p_cal = np.clip(iso.predict(out["p_rain"].values), 1e-4, 1 - 1e-4)

    train_rain = tr[tr["y_occurrence"] == 1]
    tpp = np.clip(q_models[0.5].predict(train_rain[feats]), 0, None)
    ta = train_rain["tp_mm"].values
    hv_tr = ta > thr
    shape_l, lam_l = m.fit_csg_gamma(tpp[~hv_tr], ta[~hv_tr])
    shape_h, lam_h = m.fit_csg_gamma(tpp[hv_tr], ta[hv_tr]) if hv_tr.sum() >= 10 else (shape_l, lam_l)
    lgpd = m.fit_gpd_tail(ta, thr)
    rfa = rg.fit_regional_gpd(reg_train, tr["tp_mm"].values, thr, drop, "sey")      # (kappa, scale, diag), as written

    # --- SLSQP weights on TRAINING heavy rows (Q90 proxies) ---
    weights = None
    if hv_tr.sum() > 0:
        p_tr_h = np.clip(iso.predict(occ.predict_proba(train_rain[feats][hv_tr])[:, 1]), 1e-4, 1 - 1e-4)
        q90_csg = m.csg_quantiles_gamma(0.10, 0.90, p_tr_h, shape_h, lam_h, tpp[hv_tr])[1]
        q90_l = m.gpd_quantiles(0.10, 0.90, p_tr_h, lgpd[0], lgpd[1], thr)[1] if lgpd else q90_csg
        q90_r = m.gpd_quantiles(0.10, 0.90, p_tr_h, rfa[0], rfa[1], thr)[1] if rfa else q90_csg
        pdict = {"csg": q90_csg, "lgpd": q90_l, "rgpd": q90_r}
        if with_pinn:
            pt = train_rain["pinn_tp_mm"].values[hv_tr]
            pdict["pinn"] = np.where(np.isnan(pt), tpp[hv_tr], pt) * 1.5
        weights = optimize_weights(ta[hv_tr], pdict)
        if any(np.isnan(v) for v in weights.values()):
            weights = {k: 1.0 / len(pdict) for k in pdict}
    else:
        weights = {k: 1.0 / (4 if with_pinn else 3) for k in (["csg", "lgpd", "rgpd"] + (["pinn"] if with_pinn else []))}

    # --- test intervals ---
    q10 = np.zeros(len(te))
    q90 = np.zeros(len(te))
    ng = ~gated
    if ng.sum() > 0:
        q10[ng], q90[ng] = m.csg_quantiles_gamma(0.10, 0.90, p_cal[ng], shape_l, lam_l, point_pred[ng])
    if gated.sum() > 0:
        q10_g, q90_g = m.csg_quantiles_gamma(0.10, 0.90, p_cal[gated], shape_h, lam_h, point_pred[gated])
        q10_l, q90_l = (m.gpd_quantiles(0.10, 0.90, p_cal[gated], lgpd[0], lgpd[1], thr) if lgpd else (q10_g, q90_g))
        q10_r, q90_r = (m.gpd_quantiles(0.10, 0.90, p_cal[gated], rfa[0], rfa[1], thr) if rfa else (q10_g, q90_g))
        w = weights
        q10[gated] = w["csg"] * q10_g + w["lgpd"] * q10_l + w["rgpd"] * q10_r
        q90[gated] = w["csg"] * q90_g + w["lgpd"] * q90_l + w["rgpd"] * q90_r
        if with_pinn:
            pt = te["pinn_tp_mm"].values[gated]
            pt = np.where(np.isnan(pt), point_pred[gated], pt)
            q10[gated] += w["pinn"] * pt * 0.5
            q90[gated] += w["pinn"] * pt * 1.5
    return q50_hard, q10, q90, weights


def run_district(district):
    t_all = time.time()
    raw = m.load_hourly(district)
    df = m.build_features(raw, district)
    thr = m.compute_heavy_threshold(df[df["datetime"] < TEST_START])
    print(f"{district}: heavy threshold {thr:.2f} mm/h", flush=True)

    pinn = pd.read_parquet(os.path.join(OUT_DIR, f"v2_pinn_diag_{district.lower()}.parquet"))
    pinn["datetime"] = pd.to_datetime(pinn["datetime"])
    df = df.merge(pinn, on="datetime", how="left").merge(d90.load_cnn_feats(district), on="datetime", how="left")

    sp = m.PATH_SPELLING[district]["pressure"]
    reg = pd.read_parquet(os.path.join(REG_DIR, f"{sp}_regional_tp.parquet"))
    reg["datetime"] = pd.to_datetime(reg["datetime"])
    reg_train = reg[reg["datetime"] < TEST_START]
    lat0, lon0 = m.DISTRICTS[district]
    centre = reg_train[(reg_train.latitude == lat0) & (reg_train.longitude == lon0)].set_index("datetime")["tp_mm"]
    loc = df[df["datetime"] < TEST_START].set_index("datetime")["tp_mm"]
    j = centre.to_frame("g").join(loc.rename("l"), how="inner").dropna()
    drop = (lat0, lon0) if np.corrcoef(j["g"], j["l"])[0, 1] > 0.99 else None

    rows = []
    for h in LEADS:
        dfh = df.copy()
        dfh["tp_mm"] = df["tp_mm"].shift(-(h - 1))
        dfh["cnn_pred"] = df[f"cnn_h{h}"]
        dfh = dfh.dropna(subset=["tp_mm", "cnn_pred"]).reset_index(drop=True)
        dfh["y_occurrence"] = (dfh["tp_mm"] > m.RAIN_THR_MM).astype(int)
        tr = dfh[dfh["datetime"] < TEST_START].reset_index(drop=True)
        te = dfh[dfh["datetime"] >= TEST_START].reset_index(drop=True)
        y = te["tp_mm"].values

        # v1 arms (current production code, soft gate)
        for arm, feats in (("v1_control", BASE_FEATS), ("v1_stacked", BASE_FEATS + ["cnn_pred"])):
            m.FEATURE_COLS = feats
            o, _ = m.run_split(tr, te, thr)
            rows.append(metrics(y, o["q50_final"].values, o["q10_ens"].values, o["q90_ens"].values,
                                thr, arm, district, h))
        m.FEATURE_COLS = BASE_FEATS

        # v2 arms (Seychelles as written; base features, no CNN)
        out, models = m.run_split(tr, te, thr)
        for arm, with_pinn in (("v2_pinn", True), ("v2_nopinn", False)):
            q50h, q10, q90, w = seychelles_arm(tr, te, out, models, thr, reg_train, drop, with_pinn)
            extra = {f"w_{k}": v for k, v in w.items()}
            rows.append(metrics(y, q50h, q10, q90, thr, arm, district, h, extra))
            # v2.1 = fix 1: SOFT-gate point forecast (v1's run_split q50_final); intervals identical to v2
            rows.append(metrics(y, out["q50_final"].values, q10, q90, thr, arm.replace("v2_", "v2_1_"),
                                district, h, extra))
        r = [x for x in rows if x["lead"] == h and x["district"] == district]
        print(f"h={h}: " + " | ".join(
            f"{x['arm']}: ovcov={x['overall_cov']:.3f} hvcov={x['heavy_cov']:.3f} IS={x['heavy_IS']:.1f} corr={x['corr_all']:.3f}"
            for x in r), flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"v2_{district.lower()}_summary.csv"), index=False)
    print(f"{district} done in {(time.time()-t_all)/60:.1f} min", flush=True)


def verdict(res, a, b, label):
    wis = res.pivot_table(index=["district", "lead"], columns="arm", values="heavy_IS")
    wcv = res.pivot_table(index=["district", "lead"], columns="arm", values="overall_cov")
    n = len(wis)
    need = int(np.ceil(0.75 * n))
    ca = int((wis[a] < wis[b]).sum())
    cb = int(((wcv[a] - 0.8).abs() < (wcv[b] - 0.8).abs()).sum())
    ok = ca >= need and cb >= need
    print(f"\n{label}: {a} vs {b}\n  (a) heavy interval score lower: {ca}/{n} (need >={need})   mean IS {wis[a].mean():.2f} vs {wis[b].mean():.2f}"
          f"\n  (b) overall coverage closer to 0.80: {cb}/{n} (need >={need})   mean cov {wcv[a].mean():.3f} vs {wcv[b].mean():.3f}"
          f"\n  -> {'PASS (v2 good)' if ok else 'FAIL (not good)'}")
    return ok


def fix1_report(res):
    """Pre-registered fix-1 criteria (Chroma v2.1 entry)."""
    pt = res.pivot_table(index=["district", "lead"], columns="arm", values=["corr_all", "bias_rainy", "rmse_all"])
    n = len(pt)
    need = int(np.ceil(0.75 * n))
    f1a = int((pt["bias_rainy"]["v2_1_pinn"].abs() <= 0.15).sum())
    f1b = int((pt["corr_all"]["v2_1_pinn"] > pt["corr_all"]["v2_pinn"]).sum())
    diffs = max((pt[c]["v2_1_pinn"] - pt[c]["v1_control"]).abs().max() for c in ("corr_all", "bias_rainy", "rmse_all"))
    f1c = diffs < 1e-9
    ok = f1a >= need and f1b >= need and f1c
    v2_a = int((pt["bias_rainy"]["v2_pinn"].abs() <= 0.15).sum())
    print(f"\nFIX 1 (v2.1) PRE-REGISTERED CRITERIA, {n} cells:")
    print(f"  F1-a |rainy bias| <= 0.15: {f1a}/{n} (need >={need})   [v2 hard gate: {v2_a}/{n}]")
    print(f"  F1-b corr_all better than v2: {f1b}/{n} (need >={need})   mean "
          f"{pt['corr_all']['v2_1_pinn'].mean():.3f} vs {pt['corr_all']['v2_pinn'].mean():.3f}")
    print(f"  F1-c v2.1 point == v1_control: max abs diff {diffs:.2e} (need <1e-9)")
    print(f"  -> fix 1 {'ACCEPTED' if ok else 'NOT accepted'}")


def summary():
    files = [os.path.join(OUT_DIR, f"v2_{d.lower()}_summary.csv") for d in m.DISTRICTS]
    res = pd.concat([pd.read_csv(f) for f in files if os.path.exists(f)], ignore_index=True)
    print(f"districts present: {sorted(res['district'].unique())}")
    verdict(res, "v2_pinn", "v1_stacked", "PRE-REGISTERED VERDICT")
    verdict(res, "v2_nopinn", "v1_stacked", "(secondary) without the PINN member")
    verdict(res, "v1_stacked", "v1_control", "(reference) v1 stack vs v1 control")
    if "v2_1_pinn" in set(res["arm"]):
        fix1_report(res)
        wr = res.pivot_table(index=["district", "lead"], columns="arm", values="rainy_cov")
        wi = res.pivot_table(index=["district", "lead"], columns="arm", values="heavy_IS")
        n = len(wr); need = int(np.ceil(0.75 * n))
        for a in ("v2_pinn", "v2_1_pinn", "v2_nopinn", "v2_1_nopinn"):
            ca = int((wi[a] < wi["v1_stacked"]).sum())
            cb = int(((wr[a] - 0.8).abs() < (wr["v1_stacked"] - 0.8).abs()).sum())
            print(f"  MAIN TEST (a, b') {a:12s} vs v1_stacked: heavy IS lower {ca}/{n}; rainy-hour coverage closer to 0.80 {cb}/{n} "
                  f"(need >={need} both) -> {'PASS' if ca >= need and cb >= need else 'FAIL'}   "
                  f"[mean rainy_cov {wr[a].mean():.3f} vs v1_stacked {wr['v1_stacked'].mean():.3f}]")
    pd.set_option("display.width", 250)
    for col in ("corr_all", "rmse_all", "corr_rainy", "bias_rainy", "overall_cov", "overall_width",
                "heavy_cov", "heavy_width", "heavy_IS"):
        w = res.pivot_table(index=["district", "lead"], columns="arm", values=col)
        print(f"\nmean {col}:\n{w.mean().round(3).to_string()}")
    wis = res.pivot_table(index=["district", "lead"], columns="arm", values="heavy_IS")
    print("\nLEAK SIGNATURE: mean heavy_IS (v1_stacked - v2_pinn) minus (v1_stacked - v2_nopinn) by lead "
          "(positive = PINN member helps; a gain concentrated at h=1 indicates same-hour leakage):")
    print(((wis["v1_stacked"] - wis["v2_pinn"]) - (wis["v1_stacked"] - wis["v2_nopinn"])).groupby("lead").mean().round(2).to_string())
    wcols = [c for c in res.columns if c.startswith("w_")]
    print("\nSLSQP weights (mean over cells), v2_pinn:\n" +
          res[res.arm == "v2_pinn"][wcols].mean().round(3).to_string())
    print("\nv2_pinn PINN weight by lead:\n" + res[res.arm == "v2_pinn"].groupby("lead")["w_pinn"].mean().round(3).to_string())


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "Nagaon"
    if arg == "summary":
        summary()
    elif arg in m.DISTRICTS:
        run_district(arg)
    else:
        print(f"Unknown '{arg}'. Choices: {list(m.DISTRICTS)} or summary")
