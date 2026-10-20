"""
Amendment 4, post-hoc forward-chaining check (added after the results; D1-D3 stay as registered).
LOYO history for test year Y includes later years. Here, for each test year Y in 2019..2026, everything uses only
years < Y: H2/H5 training rows, B0 thresholds, and history features (training rows of year y: years < Y except y;
test rows: years < Y). Output eval_amend4_forward.csv.
"""
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s173", os.path.join(HERE, "173_amendment4.py"))
a4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a4)
a3, m, dc, H2, H5 = a4.a3, a4.m, a4.dc, a4.H2, a4.H5
TEST_YEARS = list(range(2019, 2027))


def main():
    T = a3.a2.load_amended()
    years = sorted(T["year"].unique())
    ff = {Y: np.full(len(T), np.nan, np.float32) for Y in TEST_YEARS}
    bs = {Y: np.full(len(T), np.nan, np.float32) for Y in TEST_YEARS}
    for d in dc.DISTRICTS:
        dmask = (T["district"] == d).values
        sd, cells, cix, L = a3.district_setup(T, d)
        yr = sd["year"].values
        W = {y: (L[yr == y] == 1).sum(0).astype(np.int32) for y in years}
        O = {y: (L[yr == y] != 255).sum(0).astype(np.int32) for y in years}
        t0 = time.time()
        for Y in TEST_YEARS:
            past = [y for y in years if y < Y]
            for rows_year in past + [Y]:
                use = [y for y in past if y != rows_year]
                w = sum(W[y] for y in use); o = sum(O[y] for y in use)
                f = np.where(o > 0, w / np.maximum(o, 1), np.nan)
                thr, drv, nobs = a3.b0_fit(sd, L, sd["year"].isin(use).values)
                r = np.flatnonzero(dmask & (T["year"] == rows_year).values)
                ff[Y][r] = f[cix[T["cell"].values[r]]]
                bs[Y][r] = a3.b0_predict(T.iloc[r], sd, cix, thr, drv, nobs)[0]
            print(f"{d}: forward history for {Y} done ({time.time() - t0:.0f}s)", flush=True)
    P = {a: np.full(len(T), np.nan) for a in ("H2", "H5")}
    TH = {a: np.zeros(len(T)) for a in ("H2", "H5")}
    pb = np.full(len(T), np.nan)
    for Y in TEST_YEARS:
        trm, tem = (T["year"] < Y).values, (T["year"] == Y).values
        X = T[H2].assign(ff=ff[Y], b0s=bs[Y], y=T["y"].values)
        for a, cols in (("H2", H2), ("H5", H5)):
            mdl = a3.fit_xgb(cols, X[trm])
            P[a][tem] = mdl.predict_proba(X.loc[tem, cols])[:, 1]
            TH[a][tem] = m.best_f1_threshold(T.loc[trm, "y"].values, mdl.predict_proba(X.loc[trm, cols])[:, 1])
        pb[tem] = bs[Y][tem]
        print(f"forward {Y}: train rows {trm.sum()}", flush=True)
    rows = []
    for d in dc.DISTRICTS:
        keep = (T["district"] == d).values & (T["year"] >= 2019).values & ~np.isnan(pb)
        s = T[keep]
        for a, p, f in (("H2", P["H2"][keep], P["H2"][keep] >= TH["H2"][keep]),
                        ("H5", P["H5"][keep], P["H5"][keep] >= TH["H5"][keep]), ("B0", pb[keep], pb[keep] >= 1)):
            rows.append({"cv": "FORWARD 2019-2026", "district": d, "arm": a, **a3.scores(s, p, f)})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(dc.DYN, "eval_amend4_forward.csv"), index=False)
    v = lambda a, c: R[R.arm == a].set_index("district")[c]
    pd.set_option("display.width", 250)
    print(R.round(4).to_string(index=False))
    for c in ("f1", "within_auc", "auc"):
        print(f"H5-H2 {c}:", (v("H5", c) - v("H2", c)).round(4).to_dict(), "| H5-B0 f1" if c == "f1" else "",
              (v("H5", "f1") - v("B0", "f1")).round(4).to_dict() if c == "f1" else "")


if __name__ == "__main__":
    main()
