"""
Recursive (roll-forward) whole-day rainfall prediction test.

Prit's idea (2026-10-09): predict +1 h from the previous hour's data, feed that prediction back in
as the "previous hour", and keep going to cover a whole day. Does it predict properly?
Design + pre-registered criteria: Chroma manual-0123, TASK.md 2026-10-09.

One h=1 model (20_base_rainfall.fit_models, trained < 2024, soft-gate point forecast) is rolled
forward k = 1..24 from every 2024+ origin hour t0 whose 24 history hours, 24 target hours and 24
feature rows are all present.
At step k (target hour t0+k-1):
  * rain features (tp_lag1h, tp_sum3h/6h/12h/24h, api_7d_lag1h, api_30d_lag1h) are rebuilt from the
    model's OWN earlier predictions (API by its exact recursion y_t = coef * y_{t-1} + tp_t);
  * time features advance to the target hour;
  * every atmospheric feature (CAPE, CIN, TCWV, KX, q850, w500, r850, soil moisture, winds,
    pressure terms, static terrain/land cover) is FROZEN at its origin value. That is the honest
    no-weather-forecast assumption: the future atmosphere is unknown without NWP.

Arms:
  recursive_frozen      the realistic version
  recursive_oracle_atm  UPPER BOUND, not available in practice: the TRUE future atmospheric features
                        at each step, recursion on rain only (separates "missing future atmosphere"
                        from "recursion error")
  persistence           last observed hour, held constant
  climatology           hour-of-day x month mean from the training period
  direct                the shipped direct models h=1..6 (forecast_horizons_<district>.csv)

Regression check: step k=1 of recursive_frozen must equal the shipped v1_control h=1 predictions.

PRE-REGISTERED "RECURSION PREDICTS PROPERLY" = ALL of (Chroma manual-0123):
  R1 lead 6: recursive_frozen corr_all >= direct h=6 corr_all - 0.02 in >= 3/4 districts
  R2 leads 7..24: recursive_frozen corr_all beats BOTH climatology and persistence in >= 3/4
     districts, for >= 14 of the 18 leads
  R3 heavy-hour point hit rate (q50 > district heavy threshold on heavy hours) >= 0.50 at lead 6,
     mean over districts (direct h=6 was 0.19, h=3 0.47)
  R4 24-h total: corr(predicted total, actual total) >= 0.5 averaged over districts and higher than
     the climatology total's
Oracle arm is reported, not a criterion.

Run:  python -u scripts/110_recursive_rollout.py Nagaon
      python -u scripts/110_recursive_rollout.py summary
"""
import importlib
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m = importlib.import_module("20_base_rainfall")
ROOT = m.ROOT
FEATS = list(m.FEATURE_COLS)
TEST_START = pd.Timestamp(m.TEST_SPLIT_DATE)
OUT_DIR = os.path.join(ROOT, "data", "rainfall_nowcast")
K = 24
RAIN_COLS = ["tp_lag1h", "tp_sum3h", "tp_sum6h", "tp_sum12h", "tp_sum24h", "api_7d_lag1h", "api_30d_lag1h"]
TIME_COLS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "month"]
API_COEF = {"api_7d_lag1h": 0.9933, "api_30d_lag1h": 0.9979}
ci = {c: FEATS.index(c) for c in FEATS}


def point_forecast(models, X, thr, n_heavy_train):
    """Exactly the soft-gate q50_final of 20_base_rainfall.run_split."""
    occ, q_models, gate_clf, ext_reg = models
    Xd = pd.DataFrame(X, columns=FEATS)
    p = occ.predict_proba(Xd)[:, 1]
    is_rain = p > 0.5
    q50 = np.where(is_rain, q_models[0.5].predict(Xd), 0.0)
    gp = gate_clf.predict_proba(Xd)[:, 1]
    gated = gp > m.GATE_PROBABILITY_CUTOFF
    ext = ext_reg.predict(Xd) if n_heavy_train > 5 else np.zeros(len(Xd))
    gw = np.clip((gp - m.GATE_PROBABILITY_CUTOFF) / (1.0 - m.GATE_PROBABILITY_CUTOFF), 0.0, 1.0)
    blended = gw * ext + (1.0 - gw) * q50
    out = np.where(gated, blended, q50)
    return np.where(is_rain, np.clip(out, 0, None), 0.0)


def time_feats(dts):
    hour = dts.hour.values
    doy = dts.dayofyear.values
    return {"hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
            "doy_sin": np.sin(2 * np.pi * doy / 365), "doy_cos": np.cos(2 * np.pi * doy / 365),
            "month": dts.month.values.astype(float)}


def rollout(models, thr, nht, X0, row_idx, dfv, hist, t0s, oracle):
    N = len(row_idx)
    P = np.zeros((N, K))
    api = {c: X0[:, ci[c]].copy() for c in API_COEF}
    seq = hist.copy()                                    # (N, 24) actual history
    for k in range(1, K + 1):
        X = dfv[row_idx + (k - 1)].copy() if oracle else X0.copy()
        if k > 1:
            for c, coef in API_COEF.items():
                api[c] = coef * api[c] + P[:, k - 2]
        X[:, ci["tp_lag1h"]] = seq[:, -1]
        X[:, ci["tp_sum3h"]] = seq[:, -3:].sum(1)
        X[:, ci["tp_sum6h"]] = seq[:, -6:].sum(1)
        X[:, ci["tp_sum12h"]] = seq[:, -12:].sum(1)
        X[:, ci["tp_sum24h"]] = seq[:, -24:].sum(1)
        for c in API_COEF:
            X[:, ci[c]] = api[c]
        tf = time_feats(t0s + pd.Timedelta(hours=k - 1))
        for c in TIME_COLS:
            X[:, ci[c]] = tf[c]
        P[:, k - 1] = point_forecast(models, X, thr, nht)
        seq = np.concatenate([seq, P[:, k - 1:k]], axis=1)
    return P


def run_district(district):
    t_all = time.time()
    raw = m.load_hourly(district)
    df = m.build_features(raw, district)
    surf = m.load_surface(district).set_index("datetime")["tp_mm"]
    tps = surf.reindex(pd.date_range(surf.index.min(), surf.index.max(), freq="h"))
    train = df[df["datetime"] < TEST_START].reset_index(drop=True)
    thr = m.compute_heavy_threshold(train)
    nht = int((train["tp_mm"] > thr).sum())
    m.FEATURE_COLS = FEATS
    models = m.fit_models(train, thr)
    print(f"{district}: heavy thr {thr:.2f}, train rows {len(train):,}", flush=True)

    dts = pd.DatetimeIndex(df["datetime"])
    dfv = df[FEATS].values.astype(float)
    n = len(df)
    ok = np.zeros(n, dtype=bool)
    ok[: n - (K - 1)] = (dts[K - 1:] - dts[: n - (K - 1)]) == pd.Timedelta(hours=K - 1)   # contiguous rows
    ok &= (dts >= TEST_START)
    idx = np.where(ok)[0]
    t0s = dts[idx]
    tpv = tps.values
    pos = tps.index.get_indexer(t0s)
    hist = np.stack([tpv[pos + j] for j in range(-K, 0)], axis=1)
    Y = np.stack([tpv[pos + j] for j in range(K)], axis=1)
    good = ~np.isnan(hist).any(1) & ~np.isnan(Y).any(1)
    idx, t0s, hist, Y = idx[good], t0s[good], hist[good], Y[good]
    X0 = dfv[idx]
    print(f"origins: {len(idx):,} (of {int((dts >= TEST_START).sum()):,} test rows)", flush=True)

    P_f = rollout(models, thr, nht, X0, idx, dfv, hist, t0s, oracle=False)
    P_o = rollout(models, thr, nht, X0, idx, dfv, hist, t0s, oracle=True)

    # baselines
    tr_s = tps[tps.index < TEST_START].dropna()
    clim = tr_s.groupby([tr_s.index.hour, tr_s.index.month]).mean()
    C = np.zeros((len(idx), K))
    for k in range(K):
        tt = t0s + pd.Timedelta(hours=k)
        C[:, k] = clim.reindex(list(zip(tt.hour, tt.month))).values
    Pe = np.repeat(hist[:, -1:], K, axis=1)

    # direct (shipped) h=1..6 on the same origins
    fh = pd.read_csv(os.path.join(OUT_DIR, f"forecast_horizons_{district.lower()}.csv"), parse_dates=["datetime"])
    D = np.full((len(idx), K), np.nan)
    for h in range(1, 7):
        s = fh[fh["lead_hour"] == h].set_index("datetime")["q50_final"]
        D[:, h - 1] = s.reindex(t0s).values

    # regression check: recursive_frozen step 1 must equal a FRESH run_split h=1 forecast (training is
    # deterministic: two fresh runs are identical). The stored forecast_horizons CSV differs from a fresh
    # run by a small drift (Nagaon: mean 0.004 mm, corr 0.8960 vs 0.8961), so it is reported, not asserted.
    tst = df[df["datetime"] >= TEST_START].reset_index(drop=True)
    fresh, _ = m.run_split(train, tst, thr)
    fs = pd.Series(fresh["q50_final"].values, index=pd.DatetimeIndex(tst["datetime"]))
    d1 = np.abs(P_f[:, 0] - fs.reindex(t0s).values)
    print(f"REGRESSION CHECK step 1 vs fresh run_split: max abs diff {np.nanmax(d1):.2e} over {len(d1):,} rows", flush=True)
    assert np.nanmax(d1) < 1e-6, "rollout step 1 does not reproduce the h=1 model"
    ds = np.abs(P_f[:, 0] - D[:, 0])
    print(f"info: vs stored forecast_horizons CSV: mean abs diff {np.nanmean(ds):.4f} mm, max {np.nanmax(ds):.3f} "
          f"(stored file drift, not a rollout error)", flush=True)

    rows = []
    arms = {"recursive_frozen": P_f, "recursive_oracle_atm": P_o, "persistence": Pe, "climatology": C, "direct": D}
    for arm, Pm in arms.items():
        for k in range(1, K + 1):
            y, p = Y[:, k - 1], Pm[:, k - 1]
            v = ~np.isnan(p)
            if v.sum() < 100:
                continue
            y, p = y[v], p[v]
            rn, hv = y > m.RAIN_THR_MM, y > thr
            rows.append({"district": district, "arm": arm, "lead": k, "n": int(v.sum()), "n_heavy": int(hv.sum()),
                         "corr_all": np.corrcoef(y, p)[0, 1] if p.std() > 0 else np.nan,
                         "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
                         "corr_rainy": np.corrcoef(y[rn], p[rn])[0, 1] if p[rn].std() > 0 else np.nan,
                         "bias_rainy": float((p[rn].mean() - y[rn].mean()) / y[rn].mean()),
                         "heavy_hit": float((p[hv] > thr).mean()),
                         "heavy_pred_over_actual": float(p[hv].mean() / y[hv].mean())})
        if arm != "direct":
            ya, pa = Y.sum(1), Pm.sum(1)
            rows.append({"district": district, "arm": arm, "lead": 0, "n": len(ya),   # lead 0 = 24-h total
                         "corr_all": np.corrcoef(ya, pa)[0, 1] if pa.std() > 0 else np.nan,
                         "rmse": float(np.sqrt(np.mean((ya - pa) ** 2))),
                         "bias_rainy": float((pa.mean() - ya.mean()) / ya.mean())})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"recursive_{district.lower()}_summary.csv"), index=False)
    print(f"{district} done in {(time.time()-t_all)/60:.1f} min", flush=True)


def summary():
    files = [os.path.join(OUT_DIR, f"recursive_{d.lower()}_summary.csv") for d in m.DISTRICTS]
    res = pd.concat([pd.read_csv(f) for f in files if os.path.exists(f)], ignore_index=True)
    nd = res["district"].nunique()
    print(f"districts present: {sorted(res['district'].unique())}")
    pd.set_option("display.width", 250)
    w = res[res.lead > 0].pivot_table(index=["district", "lead"], columns="arm", values="corr_all")
    # R1
    r1 = int(((w["recursive_frozen"].xs(6, level="lead")) >= (w["direct"].xs(6, level="lead") - 0.02)).sum())
    # R2
    cnt = 0
    for L in range(7, 25):
        sub = w.xs(L, level="lead")
        beats = int(((sub["recursive_frozen"] > sub["climatology"]) & (sub["recursive_frozen"] > sub["persistence"])).sum())
        cnt += beats >= int(np.ceil(0.75 * nd))
    # R3
    hh = res[(res.lead == 6) & (res.arm == "recursive_frozen")]["heavy_hit"].mean()
    # R4
    t = res[res.lead == 0].pivot_table(index="district", columns="arm", values="corr_all")
    r4a, r4b = t["recursive_frozen"].mean(), t["climatology"].mean()
    print(f"\nPRE-REGISTERED CRITERIA (Chroma manual-0123)")
    print(f"  R1 lead-6 corr within 0.02 of direct, in >=3/4 districts: {r1}/{nd}  -> {'PASS' if r1 >= int(np.ceil(0.75*nd)) else 'FAIL'}")
    print(f"  R2 beats climatology AND persistence at leads 7..24: {cnt}/18 leads (need >=14)  -> {'PASS' if cnt >= 14 else 'FAIL'}")
    print(f"  R3 heavy-hour point hit rate at lead 6: {hh:.3f} (need >=0.50; direct h6 was 0.19)  -> {'PASS' if hh >= 0.5 else 'FAIL'}")
    print(f"  R4 24-h total corr: recursive {r4a:.3f} vs climatology {r4b:.3f} (need >=0.5 and higher)  -> {'PASS' if r4a >= 0.5 and r4a > r4b else 'FAIL'}")
    allp = r1 >= int(np.ceil(0.75 * nd)) and cnt >= 14 and hh >= 0.5 and r4a >= 0.5 and r4a > r4b
    print(f"  => recursion {'PREDICTS PROPERLY' if allp else 'DOES NOT predict properly'} (all four required)")
    for col in ("corr_all", "bias_rainy", "heavy_hit", "heavy_pred_over_actual"):
        tab = res[res.lead.isin([1, 3, 6, 12, 18, 24])].pivot_table(index="lead", columns="arm", values=col)
        print(f"\nmean {col} (over districts) at selected leads:\n{tab.round(3).to_string()}")
    print("\n24-h TOTAL (lead 0): corr / bias by arm (mean over districts):")
    print(res[res.lead == 0].groupby("arm")[["corr_all", "bias_rainy", "rmse"]].mean().round(3).to_string())


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "Nagaon"
    if arg == "summary":
        summary()
    elif arg in m.DISTRICTS:
        run_district(arg)
    else:
        print(f"Unknown '{arg}'. Choices: {list(m.DISTRICTS)} or summary")
