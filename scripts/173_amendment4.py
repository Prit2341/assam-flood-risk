"""
Amendment 4 (exploratory, registered 2026-10-16 in docs/dynamic_susceptibility_method.md before any run).
H5 = H2 + per-pixel label-history features: ff (flood frequency) and b0s (Amendment 3 B0 score Q/t).
Leakage rule: in LOYO fold Y, test rows (year Y) use history from years != Y; training rows of year y use history
from years not in {Y, y}. Outputs eval_amend4_*.csv; Amendments 1-3 are not touched.

Run:  python -u scripts/173_amendment4.py
"""
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s172", os.path.join(HERE, "172_amendment3.py"))
a3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a3)
m, dc, H2 = a3.m, a3.dc, a3.H2
H5 = H2 + ["ff", "b0s"]


def history_features(T):
    """ff[Y], b0s[Y]: per-row feature arrays for LOYO fold Y, following the leakage rule."""
    years = sorted(T["year"].unique())
    ff = {Y: np.full(len(T), np.nan, np.float32) for Y in years}
    bs = {Y: np.full(len(T), np.nan, np.float32) for Y in years}
    for d in dc.DISTRICTS:
        dmask = (T["district"] == d).values
        sd, cells, cix, L = a3.district_setup(T, d)
        yr = sd["year"].values
        W = np.stack([(L[yr == y] == 1).sum(0) for y in years]).astype(np.int32)   # per-year wet counts per pixel
        O = np.stack([(L[yr == y] != 255).sum(0) for y in years]).astype(np.int32)
        Wt, Ot = W.sum(0), O.sum(0)
        ri = {y: np.flatnonzero(dmask & (T["year"] == y).values) for y in years}
        t0 = time.time()
        for i, Y in enumerate(years):
            for j, y in enumerate(years):
                if j < i:
                    continue
                excl = {Y, y}
                w = Wt - sum(W[years.index(e)] for e in excl)
                o = Ot - sum(O[years.index(e)] for e in excl)
                f = np.where(o > 0, w / np.maximum(o, 1), np.nan)
                thr, drv, nobs = a3.b0_fit(sd, L, ~sd["year"].isin(excl).values)
                # Y == y: test rows of fold Y. Otherwise rows of y in fold Y and rows of Y in fold y.
                for fold, rows_year in ((Y, y), (y, Y)) if Y != y else ((Y, Y),):
                    r = ri[rows_year]
                    ff[fold][r] = f[cix[T["cell"].values[r]]]
                    bs[fold][r] = a3.b0_predict(T.iloc[r], sd, cix, thr, drv, nobs)[0]
            print(f"{d}: history for fold {Y} done ({time.time() - t0:.0f}s)", flush=True)
    return ff, bs


def main():
    T = a3.a2.load_amended()
    ph2, th2 = a3.load_preds()
    ff, bs = history_features(T)
    years = sorted(T["year"].unique())
    p5 = np.full(len(T), np.nan)
    th5 = np.zeros(len(T))
    pb = np.full(len(T), np.nan)
    for Y in years:
        X = T[H2].assign(ff=ff[Y], b0s=bs[Y])
        tem = (T["year"] == Y).values
        t0 = time.time()
        mdl = a3.fit_xgb(H5, X[~tem].assign(y=T.loc[~tem, "y"].values))
        p5[tem] = mdl.predict_proba(X.loc[tem, H5])[:, 1]
        th5[tem] = m.best_f1_threshold(T.loc[~tem, "y"].values, mdl.predict_proba(X.loc[~tem, H5])[:, 1])
        pb[tem] = bs[Y][tem]                                   # B0 score for the test year (history from years != Y)
        print(f"H5 LOYO {Y}: {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(os.path.join(dc.DYN, "amend4_h5_loyo_preds.npz"), p=p5, thr=th5, b0=pb)
    imp = pd.Series(mdl.feature_importances_, index=H5).sort_values(ascending=False)
    rows = []
    for d in dc.DISTRICTS:
        dm = (T["district"] == d).values
        keep = dm & ~np.isnan(pb)
        s = T[keep]
        y = s["y"].values
        for a, p, f in (("H2", ph2[keep], ph2[keep] >= th2[keep]), ("H5", p5[keep], p5[keep] >= th5[keep]),
                        ("B0", pb[keep], pb[keep] >= 1)):
            r = a3.scores(s, p, f)
            r["brier"] = float(np.mean((p - y) ** 2)) if a != "B0" else np.nan
            rows.append({"district": d, "arm": a, "coverage": keep.sum() / dm.sum(), **r})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(dc.DYN, "eval_amend4.csv"), index=False)
    v = lambda a, c: R[R.arm == a].set_index("district")[c]
    C = pd.DataFrame([
        {"check": "D1 H5 pooled F1 > H2", "detail": (v("H5", "f1") - v("H2", "f1")).round(4).to_dict(),
         "districts_pass": int((v("H5", "f1") > v("H2", "f1")).sum())},
        {"check": "D2 H5 pooled F1 >= B0", "detail": (v("H5", "f1") - v("B0", "f1")).round(4).to_dict(),
         "districts_pass": int((v("H5", "f1") >= v("B0", "f1")).sum())},
        {"check": "D3 H5 within-scene AUC > H2", "detail": (v("H5", "within_auc") - v("H2", "within_auc")).round(4).to_dict(),
         "districts_pass": int((v("H5", "within_auc") > v("H2", "within_auc")).sum())}])
    C["pass"] = C["districts_pass"] >= 3
    C.to_csv(os.path.join(dc.DYN, "eval_amend4_criteria.csv"), index=False)
    pd.set_option("display.width", 250)
    print(R.round(4).to_string(index=False)); print(C.to_string(index=False))
    print("feature importance (last fold, gain-based, descriptive):", imp.head(8).round(3).to_dict())


if __name__ == "__main__":
    main()
