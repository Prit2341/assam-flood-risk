"""
Amendment 3 (exploratory, registered 2026-10-15 in docs/dynamic_susceptibility_method.md before any run).
P1-P6 (scripts/170) and Amendment 2 (scripts/171) are not touched; outputs go to eval_amend3_*.csv.

  h2   H2 pooled-LOYO out-of-fold predictions on the K-amended table (saved, reused by b0 and cal)
  b0   (a) per-pixel stage-threshold baseline, simplified from Nevo et al. 2022 HESS 26:4013, vs H2
  ext  (b) leave-extreme-out: per district the top-2 years by labelled-scene max(q_trib, q_main) held out jointly
  cal  (c) nested isotonic calibration of H2 (Niculescu-Mizil & Caruana 2005 ICML)

Run:  python -u scripts/172_amendment3.py h2 | b0 | ext | cal
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location("s171", os.path.join(HERE, "171_amendment2.py"))
a2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a2)
m, dc = a2.m, a2.dc
H1, H2 = a2.H1, a2.H2
MIN_HIST = 10
DRIVERS = ("q_trib", "q_main")
PREDS = os.path.join(dc.DYN, "amend3_h2_loyo_preds.npz")


def out(name):
    return os.path.join(dc.DYN, f"eval_amend3_{name}.csv")


def fit_xgb(cols, tr):
    return XGBClassifier(**m.XGB).fit(tr[cols], tr["y"])


def scores(s, p, flag):
    """m.score for the ranking metrics; F1/CSI from the given binary flags; plus Nevo-style median per-year F1."""
    r = m.score(s, p, np.inf)
    y = s["y"].values
    tp = (flag & (y == 1)).sum(); fp = (flag & (y == 0)).sum(); fn = (~flag & (y == 1)).sum()
    r["csi"] = tp / (tp + fp + fn) if tp + fp + fn else np.nan
    r["f1"] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
    f1y = []
    for _, g in s.assign(f=flag).groupby("year"):
        t = (g["f"] & (g["y"] == 1)).sum(); d = g["f"].sum() + (g["y"] == 1).sum()
        if (g["y"] == 1).sum():
            f1y.append(2 * t / d)
    r["f1_median_year"] = np.median(f1y) if f1y else np.nan
    return r


# ---------------------------------------------------------------- H2 pooled LOYO (saved)
def run_h2():
    T = a2.load_amended()
    p = np.full(len(T), np.nan)
    thr = np.zeros(len(T))
    for yr in sorted(T["year"].unique()):
        tem = (T["year"] == yr).values
        t0 = time.time()
        mdl = fit_xgb(H2, T[~tem])
        p[tem] = mdl.predict_proba(T.loc[tem, H2])[:, 1]
        thr[tem] = m.best_f1_threshold(T.loc[~tem, "y"].values, mdl.predict_proba(T.loc[~tem, H2])[:, 1])
        print(f"H2 LOYO {yr}: {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(PREDS, p=p, thr=thr)


def load_preds():
    if not os.path.exists(PREDS):
        sys.exit("run step h2 first")
    z = np.load(PREDS)
    return z["p"], z["thr"]


# ---------------------------------------------------------------- B0: per-pixel stage threshold
def scene_drivers(T, d):
    """One row per labelled scene of district d: date, year, q_trib, q_main (Q/Q90 at t-1, as in the table)."""
    s = T[T["district"] == d].groupby("date").agg(year=("year", "first"), q_trib=("q_trib", "first"),
                                                 q_main=("q_main", "first"))
    return s.sort_index()


def label_stack(d, dates, cells):
    """uint8 (n_scenes, n_cells): 0 dry, 1 wet, 255 not observed, from the full GFM label rasters."""
    L = np.empty((len(dates), len(cells)), np.uint8)
    for i, t in enumerate(dates):
        L[i] = np.load(os.path.join(dc.DYN, "labels", d, f"{t:%Y-%m-%d}.npz"))["label"].ravel()[cells]
    return L


def fit_thresholds(q, L, chunk=40000):
    """Per pixel, the single threshold t maximising F1 of (q >= t) over that pixel's observed scenes.
    With scenes sorted by q descending and k = index of the lowest scene called wet:
    TP = cumW_k, FP + TP = cumO_k, FN = totW - cumW_k  ->  F1 = 2 cumW_k / (cumO_k + totW).
    Only the last index of a run of tied q is a valid cut; argmax takes the first maximum, i.e. ties go to the
    higher threshold. Pixels never wet in training get t = inf (never wet)."""
    ok = ~np.isnan(q)
    q, L = q[ok], L[ok]
    o = np.argsort(-q, kind="stable")
    qs, L = q[o], L[o]
    cut = np.r_[qs[:-1] != qs[1:], True]
    P = L.shape[1]
    thr, f1, nobs = np.full(P, np.inf), np.zeros(P), np.zeros(P, int)
    for a in range(0, P, chunk):
        Lc = L[:, a:a + chunk]
        cw = np.cumsum(Lc == 1, axis=0, dtype=np.int32)
        co = np.cumsum(Lc != 255, axis=0, dtype=np.int32)
        tw = cw[-1]
        F = np.where(cut[:, None], 2 * cw / np.maximum(co + tw, 1), -1.0)
        k = F.argmax(axis=0)
        best = F[k, np.arange(F.shape[1])]
        wet = tw > 0
        thr[a:a + chunk] = np.where(wet, qs[k], np.inf)
        f1[a:a + chunk] = np.where(wet, best, 0)
        nobs[a:a + chunk] = co[-1]
    return thr, f1, nobs


def b0_fit(sd, L, train_mask):
    """Fit both drivers on training scenes, keep per pixel the one with higher training F1 (ties -> q_trib)."""
    fits = {}
    for drv in DRIVERS:
        q = sd[drv].values.astype(float)
        if np.all(np.isnan(q[train_mask])):
            continue
        fits[drv] = fit_thresholds(q[train_mask], L[train_mask])
    if len(fits) == 1:
        drv = next(iter(fits))
        thr, f1, nobs = fits[drv]
        return thr, np.full(len(thr), DRIVERS.index(drv)), nobs
    (t0, f0, n0), (t1, f1_, n1) = fits["q_trib"], fits["q_main"]
    use_main = f1_ > f0
    return np.where(use_main, t1, t0), use_main.astype(int), np.where(use_main, n1, n0)


def b0_predict(Td, sd, cell_ix, thr, drv, nobs):
    """Score Q/t (0 for never-wet pixels) and flag Q >= t for the rows Td; rows with < MIN_HIST history -> NaN."""
    qm = np.stack([sd[k].reindex(Td["date"]).values for k in DRIVERS], axis=1)
    j = cell_ix[Td["cell"].values]
    q = qm[np.arange(len(Td)), drv[j]]
    t = thr[j]
    p = np.where(np.isinf(t), 0.0, q / np.where(np.isinf(t), 1, t))
    p[(nobs[j] < MIN_HIST) | np.isnan(q)] = np.nan
    return p, p >= 1


def district_setup(T, d):
    sd = scene_drivers(T, d)
    cells = np.unique(T.loc[T["district"] == d, "cell"].values)
    cell_ix = np.full(cells.max() + 1, -1)
    cell_ix[cells] = np.arange(len(cells))
    t0 = time.time()
    L = label_stack(d, sd.index, cells)
    print(f"{d}: {len(sd)} scenes x {len(cells)} cells stacked in {time.time() - t0:.0f}s", flush=True)
    return sd, cells, cell_ix, L


def run_b0():
    T = a2.load_amended()
    ph2, th2 = load_preds()
    pb = np.full(len(T), np.nan)
    fb = np.zeros(len(T), bool)
    meta = []
    for d in dc.DISTRICTS:
        dm = (T["district"] == d).values
        sd, cells, cell_ix, L = district_setup(T, d)
        for yr in sorted(T["year"].unique()):
            trm = (sd["year"] != yr).values
            thr, drv, nobs = b0_fit(sd, L, trm)
            rm = dm & (T["year"] == yr).values
            pb[rm], fb[rm] = b0_predict(T[rm], sd, cell_ix, thr, drv, nobs)
            meta.append({"district": d, "year": yr, "frac_never_wet": float(np.isinf(thr).mean()),
                         "frac_q_main": float(drv.mean()), "frac_hist_lt10": float((nobs < MIN_HIST).mean())})
        print(f"B0 {d} done", flush=True)
    pd.DataFrame(meta).to_csv(out("b0_meta"), index=False)
    rows = []
    for d in dc.DISTRICTS:
        dm = (T["district"] == d).values
        keep = dm & ~np.isnan(pb) & ~np.isnan(ph2)
        s = T[keep]
        rows.append({"district": d, "arm": "B0", "coverage": keep.sum() / dm.sum(), **scores(s, pb[keep], fb[keep])})
        rows.append({"district": d, "arm": "H2", "coverage": keep.sum() / dm.sum(),
                     **scores(s, ph2[keep], ph2[keep] >= th2[keep])})
    R = pd.DataFrame(rows)
    R.to_csv(out("b0"), index=False)
    v = lambda a, c: R[R.arm == a].set_index("district")[c]
    C = pd.DataFrame([
        {"check": "B1 H2 pooled F1 > B0", "detail": (v("H2", "f1") - v("B0", "f1")).round(4).to_dict(),
         "districts_pass": int((v("H2", "f1") > v("B0", "f1")).sum())},
        {"check": "B2 H2 pooled AUC > B0", "detail": (v("H2", "auc") - v("B0", "auc")).round(4).to_dict(),
         "districts_pass": int((v("H2", "auc") > v("B0", "auc")).sum())}])
    C["pass"] = C["districts_pass"] >= 3
    C.to_csv(out("b0_criteria"), index=False)
    print(R.round(3).to_string(index=False)); print(C.to_string(index=False))


# ---------------------------------------------------------------- (b) leave-extreme-out
def extreme_years(T):
    """Per district, the 2 years with the highest max over labelled scenes of max(q_trib, q_main). Discharge only."""
    E = {}
    for d in dc.DISTRICTS:
        sd = scene_drivers(T, d)
        sd["qx"] = sd[list(DRIVERS)].max(axis=1)
        ym = sd.groupby("year")["qx"].max().sort_values(ascending=False)
        E[d] = [int(y) for y in ym.index[:2]]
        print(f"{d}: year max Q/Q90 {ym.round(2).to_dict()} -> E = {E[d]}", flush=True)
    return E


def run_ext():
    T = a2.load_amended()
    ph2, _ = load_preds()
    E = extreme_years(T)
    rows = []
    for d in dc.DISTRICTS:
        tem = ((T["district"] == d) & T["year"].isin(E[d])).values
        trm = ~T["year"].isin(E[d]).values
        te = T[tem]
        preds = {}
        for a, cols in (("H1", H1), ("H2", H2)):
            mdl = fit_xgb(cols, T[trm])
            thr = m.best_f1_threshold(T.loc[trm, "y"].values, mdl.predict_proba(T.loc[trm, cols])[:, 1])
            p = mdl.predict_proba(te[cols])[:, 1]
            preds[a] = (p, p >= thr)
        sd, cells, cell_ix, L = district_setup(T, d)
        thr, drv, nobs = b0_fit(sd, L, ~sd["year"].isin(E[d]).values)
        preds["B0"] = b0_predict(te, sd, cell_ix, thr, drv, nobs)
        preds["A2"] = (te["A2"].values, np.zeros(len(te), bool))
        # out-of-range scenes: held-out scene max(q_trib, q_main) above the district's training-scene maximum
        sd["qx"] = sd[list(DRIVERS)].max(axis=1)
        qmax_tr = sd.loc[~sd["year"].isin(E[d]), "qx"].max()
        oor_dates = sd.index[sd["year"].isin(E[d]) & (sd["qx"] > qmax_tr)]
        keep = ~np.isnan(preds["B0"][0])
        for subset, sm in (("E_all", keep), ("out_of_range", keep & te["date"].isin(oor_dates).values)):
            if sm.sum() == 0 or te["y"].values[sm].mean() in (0, 1):
                rows.append({"district": d, "subset": subset, "E": E[d], "n_scenes_oor": len(oor_dates)})
                continue
            for a, (p, f) in preds.items():
                r = scores(te[sm], p[sm], f[sm])
                if a == "A2":
                    r["f1"] = r["csi"] = np.nan                # no fitted threshold for the AHP index here
                rows.append({"district": d, "subset": subset, "arm": a, "E": E[d], "n_scenes_oor": len(oor_dates),
                             "train_qmax": round(qmax_tr, 3), **r})
        print(f"EXT {d} done: E={E[d]}, out-of-range scenes {len(oor_dates)} (train max Q/Q90 {qmax_tr:.2f})", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(out("ext"), index=False)
    # E1 reference: H2 pooled-LOYO AUC on the full district, from the saved predictions
    ref = {d: roc_auc_score(T.loc[T["district"] == d, "y"], ph2[(T["district"] == d).values]) for d in dc.DISTRICTS}
    ea = R[(R.subset == "E_all")]
    v = lambda a, c: ea[ea.arm == a].set_index("district")[c]
    C = pd.DataFrame([
        {"check": "E1 H2 AUC on E_d >= pooled-LOYO AUC - 0.05",
         "detail": {d: [round(v("H2", "auc")[d], 4), round(ref[d], 4)] for d in dc.DISTRICTS},
         "districts_pass": int(sum(v("H2", "auc")[d] >= ref[d] - 0.05 for d in dc.DISTRICTS))},
        {"check": "E2 H2 F1 on E_d > B0", "detail": (v("H2", "f1") - v("B0", "f1")).round(4).to_dict(),
         "districts_pass": int((v("H2", "f1") > v("B0", "f1")).sum())}])
    C["pass"] = C["districts_pass"] >= 3
    C.to_csv(out("ext_criteria"), index=False)
    print(R.round(3).to_string(index=False)); print(C.to_string(index=False))


# ---------------------------------------------------------------- (c) nested isotonic calibration
def run_cal():
    T = a2.load_amended()
    ph2, _ = load_preds()
    pc = np.full(len(T), np.nan)
    for yr in sorted(T["year"].unique()):
        trm = (T["year"] != yr).values
        Tr = T[trm]
        inner = np.full(len(Tr), np.nan)
        for itr, ite in GroupKFold(n_splits=5).split(Tr, groups=Tr["year"]):
            mdl = fit_xgb(H2, Tr.iloc[itr])
            inner[ite] = mdl.predict_proba(Tr.iloc[ite][H2])[:, 1]
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(inner, Tr["y"].values)
        pc[~trm] = iso.predict(ph2[~trm])
        print(f"CAL outer {yr} done", flush=True)
    np.savez_compressed(os.path.join(dc.DYN, "amend3_h2_cal_preds.npz"), p=pc)
    rows, rel = [], []
    for d in dc.DISTRICTS:
        dm = (T["district"] == d).values
        s = T[dm]
        y = s["y"].values
        for arm, p in (("H2", ph2[dm]), ("H2cal", pc[dm])):
            g = s.assign(p=p).groupby("date").agg(pm=("p", "mean"), ym=("y", "mean"))
            wet, dry = g[g.ym >= 0.05], g[g.ym < 0.01]
            lr = np.log(wet.pm / wet.ym)
            rows.append({"district": d, "arm": arm, "brier": float(np.mean((p - y) ** 2)), "auc": roc_auc_score(y, p),
                         "n_wet_scenes": len(wet), "median_ratio_wet": float(np.exp(np.median(lr))),
                         "median_abs_log_ratio_wet": float(np.median(np.abs(lr))),
                         "dry_mean_pred": float(dry.pm.mean()), "dry_mean_obs": float(dry.ym.mean()), "n_dry_scenes": len(dry),
                         "mean_pred": float(p.mean()), "mean_obs": float(y.mean())})
            b = np.clip((p * 10).astype(int), 0, 9)
            for k in range(10):
                if (b == k).any():
                    rel.append({"district": d, "arm": arm, "bin": k, "n": int((b == k).sum()),
                                "mean_pred": float(p[b == k].mean()), "obs_rate": float(y[b == k].mean())})
    R = pd.DataFrame(rows)
    R.to_csv(out("cal"), index=False)
    pd.DataFrame(rel).to_csv(out("cal_reliability"), index=False)
    v = lambda a, c: R[R.arm == a].set_index("district")[c]
    C = pd.DataFrame([
        {"check": "C1 Brier calibrated < uncalibrated", "detail": (v("H2cal", "brier") - v("H2", "brier")).round(6).to_dict(),
         "districts_pass": int((v("H2cal", "brier") < v("H2", "brier")).sum())},
        {"check": "C2 median |log(pred/obs)| on scenes >=5% obs decreases",
         "detail": {d: [round(v("H2", "median_abs_log_ratio_wet")[d], 3), round(v("H2cal", "median_abs_log_ratio_wet")[d], 3)]
                    for d in dc.DISTRICTS},
         "districts_pass": int((v("H2cal", "median_abs_log_ratio_wet") < v("H2", "median_abs_log_ratio_wet")).sum())}])
    C["pass"] = C["districts_pass"] >= 3
    C.to_csv(out("cal_criteria"), index=False)
    print(R.round(4).to_string(index=False)); print(C.to_string(index=False))


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    {"h2": run_h2, "b0": run_b0, "ext": run_ext, "cal": run_cal}[sys.argv[1]]()
