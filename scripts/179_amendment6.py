"""
Amendment 6 (registered 2026-10-20 in docs/dynamic_susceptibility_method.md before any run).
H6 = H5 + previous-scene state (prev_wet, prev_gap, prev_frac, prev_frac_gap; past-only, gap 1..30 days);
H6v = H6 + discharge volume above Q90 over t-30..t-1 (attribution); P0 = persistence baseline.
Pooled LOYO (H5 history features follow the Amendment 4 rule) + past-only forward check 2019-2026.

Run:  python -u scripts/179_amendment6.py features | loyo | forward | score
Out:  data/dynamic_susceptibility/{amend6_features.npz, amend6_loyo_preds.npz, amend6_forward_preds.npz,
      eval_amend6_*.csv}
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
MAXGAP = 30
PREV = ["prev_wet", "prev_gap", "prev_frac", "prev_frac_gap"]
VOL = ["qv_trib", "qv_main"]
H6, H6V = H5 + PREV, H5 + PREV + VOL
FEAT = os.path.join(dc.DYN, "amend6_features.npz")
HIST = os.path.join(dc.DYN, "amend6_loyo_history.npz")
LOYO = os.path.join(dc.DYN, "amend6_loyo_preds.npz")
FWD = os.path.join(dc.DYN, "amend6_forward_preds.npz")
TEST_YEARS = list(range(2019, 2027))
GLOFAS = {**m.GLOFAS, "Karimganj": ("karimganj_kushiyara", "karimganj_barak")}   # Amendment 2, check K


def out(name):
    return os.path.join(dc.DYN, f"eval_amend6_{name}.csv")


# ---------------------------------------------------------------- features
def prev_features(T):
    pw, pg, pf, pfg = (np.full(len(T), np.nan, np.float32) for _ in range(4))
    for d in dc.DISTRICTS:
        sd, cells, cix, L = a3.district_setup(T, d)          # scenes sorted by date, L (n_scenes, n_cells)
        dates = sd.index.values.astype("datetime64[D]")
        idx = pd.read_csv(os.path.join(dc.DYN, "labels", d, "labels_index.csv"), parse_dates=["date"]).set_index("date")
        frac = idx["flood_frac_obs"].reindex(sd.index).values
        last_state = np.full(L.shape[1], np.nan, np.float32)
        last_day = np.full(L.shape[1], np.datetime64("NaT"), "datetime64[D]")
        dm = (T["district"] == d).values
        rows_by_date = T[dm].groupby("date").indices
        base = np.flatnonzero(dm)
        for i, t in enumerate(sd.index):
            r = base[rows_by_date[t]]
            j = cix[T["cell"].values[r]]
            gap = (dates[i] - last_day[j]).astype(float)
            ok = (gap >= 1) & (gap <= MAXGAP)
            pw[r] = np.where(ok, last_state[j], np.nan)
            pg[r] = np.where(ok, gap, np.nan)
            if i > 0:
                g = float((dates[i] - dates[i - 1]).astype(int))
                if 1 <= g <= MAXGAP:
                    pf[r], pfg[r] = frac[i - 1], g
            seen = L[i] != 255                                   # update state AFTER using it (past-only)
            last_state[seen] = (L[i][seen] == 1)
            last_day[seen] = dates[i]
        print(f"{d}: prev features done; prev_wet coverage {np.mean(~np.isnan(pw[dm])):.2f}", flush=True)
    return pw, pg, pf, pfg


def volume_features(T):
    qv = {k: np.full(len(T), np.nan, np.float32) for k in VOL}
    for d in dc.DISTRICTS:
        dm = (T["district"] == d).values
        for k, tag in zip(VOL, GLOFAS[d]):
            if tag is None:
                continue
            s = pd.read_csv(os.path.join(dc.DATA, "glofas_dyn", f"{tag}.csv"), parse_dates=["date"]).set_index("date")["discharge_m3s"].asfreq("D")
            a = s / s["2015":"2024"].quantile(0.9)                # same Q/Q90 normalisation as scripts/170
            v = (a - 1).clip(lower=0).rolling(30, min_periods=20).sum().shift(1)
            qv[k][dm] = v.reindex(T.loc[dm, "date"]).values
    return qv


def run_features():
    T = a3.a2.load_amended()
    pw, pg, pf, pfg = prev_features(T)
    qv = volume_features(T)
    np.savez_compressed(FEAT, prev_wet=pw, prev_gap=pg, prev_frac=pf, prev_frac_gap=pfg, **qv)
    print("coverage:", {k: round(float(np.mean(~np.isnan(v))), 3) for k, v in
                        dict(prev_wet=pw, prev_frac=pf, **qv).items()})


def load(T):
    z = np.load(FEAT)
    for k in PREV + VOL:
        T[k] = z[k]
    return T


# ---------------------------------------------------------------- LOYO
def run_loyo():
    T = load(a3.a2.load_amended())
    if os.path.exists(HIST):
        z = np.load(HIST)
        years = sorted(T["year"].unique())
        ff = {Y: z[f"ff_{Y}"] for Y in years}; bs = {Y: z[f"bs_{Y}"] for Y in years}
    else:
        ff, bs = a4.history_features(T)
        np.savez_compressed(HIST, **{f"ff_{Y}": v for Y, v in ff.items()}, **{f"bs_{Y}": v for Y, v in bs.items()})
    P = {a: np.full(len(T), np.nan) for a in ("H6", "H6v")}
    ffY = np.full(len(T), np.nan)
    for Y in sorted(T["year"].unique()):
        tem = (T["year"] == Y).values
        X = T[H2 + PREV + VOL].assign(ff=ff[Y], b0s=bs[Y], y=T["y"].values)
        ffY[tem] = ff[Y][tem]
        for a, cols in (("H6", H6), ("H6v", H6V)):
            t0 = time.time()
            P[a][tem] = a3.fit_xgb(cols, X[~tem]).predict_proba(X.loc[tem, cols])[:, 1]
            print(f"LOYO {a} {Y}: {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(LOYO, ff_test=ffY, **P)


# ---------------------------------------------------------------- forward (past-only), as scripts/173b
def run_forward():
    T = load(a3.a2.load_amended())
    years = sorted(T["year"].unique())
    ff = {Y: np.full(len(T), np.nan, np.float32) for Y in TEST_YEARS}
    bs = {Y: np.full(len(T), np.nan, np.float32) for Y in TEST_YEARS}
    for d in dc.DISTRICTS:
        dmask = (T["district"] == d).values
        sd, cells, cix, L = a3.district_setup(T, d)
        yr = sd["year"].values
        W = {y: (L[yr == y] == 1).sum(0).astype(np.int32) for y in years}
        O = {y: (L[yr == y] != 255).sum(0).astype(np.int32) for y in years}
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
        print(f"{d}: forward history done", flush=True)
    P = {a: np.full(len(T), np.nan) for a in ("H5", "H6")}
    ffY = np.full(len(T), np.nan)
    for Y in TEST_YEARS:
        trm, tem = (T["year"] < Y).values, (T["year"] == Y).values
        X = T[H2 + PREV].assign(ff=ff[Y], b0s=bs[Y], y=T["y"].values)
        ffY[tem] = ff[Y][tem]
        for a, cols in (("H5", H5), ("H6", H6)):
            P[a][tem] = a3.fit_xgb(cols, X[trm]).predict_proba(X.loc[tem, cols])[:, 1]
        print(f"forward {Y} done", flush=True)
    np.savez_compressed(FWD, ff_test=ffY, **P)


# ---------------------------------------------------------------- scoring
def scene_set():
    S = pd.read_csv(os.path.join(dc.DYN, "diag", "extent_error_scenes.csv"), parse_dates=["date"])
    return S[(S.obs_pct >= 2) | (S.pred_pct >= 2)][["district", "date"]]


def lr(pred, obs):
    return np.log10((pred + 0.5) / (obs + 0.5))


def extent_table(T, preds, scenes):
    E = T[["district", "date", "y"]].assign(**preds).merge(scenes, on=["district", "date"])
    g = E.groupby(["district", "date"])
    S = g["y"].mean().mul(100).rename("obs").to_frame()
    for a in preds:
        S[a] = g[a].mean() * 100
        S[f"lr_{a}"] = lr(S[a], S["obs"])
    return S.reset_index()


def heldout_thr(T, p):
    """Threshold for year Y = F1-optimal threshold on the pooled out-of-fold predictions of all other years."""
    th = np.zeros(len(T))
    for Y in T["year"].unique():
        o = (T["year"] != Y).values & ~np.isnan(p)
        th[(T["year"] == Y).values] = m.best_f1_threshold(T.loc[o, "y"].values, p[o])
    return th


def summarise(T, preds, scenes, label, thr=True):
    S = extent_table(T, preds, scenes)
    rows = []
    for d in dc.DISTRICTS:
        sd = S[S.district == d]
        dm = (T["district"] == d).values & ~np.isnan(preds["H5"])
        for a, p in preds.items():
            r = {"cv": label, "district": d, "arm": a, "n_scenes": len(sd),
                 "median_abs_lr": sd[f"lr_{a}"].abs().median(), "share_off_2x": (sd[f"lr_{a}"].abs() > np.log10(2)).mean(),
                 "share_over_2x": (sd[f"lr_{a}"] > np.log10(2)).mean(), "share_under_2x": (sd[f"lr_{a}"] < -np.log10(2)).mean(),
                 "pred_over_obs": sd[a].sum() / sd["obs"].sum(), "scene_spearman": sd[[a, "obs"]].corr("spearman").iloc[0, 1]}
            if a != "P0":
                s = T[dm]
                fl = p[dm] >= heldout_thr(T, p)[dm] if thr else np.zeros(dm.sum(), bool)
                sc = a3.scores(s, p[dm], fl)
                r.update(auc=sc["auc"], within_auc=sc["within_auc"], brier=float(np.mean((p[dm] - s["y"].values) ** 2)))
                if thr:
                    r.update(f1=sc["f1"], csi=sc["csi"])
            rows.append(r)
    return pd.DataFrame(rows), S


def run_score():
    T = load(a3.a2.load_amended())
    scenes = scene_set()
    z = np.load(LOYO)
    h5 = np.load(os.path.join(dc.DYN, "amend4_h5_loyo_preds.npz"))["p"]
    p0 = np.where(np.isnan(T["prev_wet"].values), z["ff_test"], T["prev_wet"].values)
    preds = {"H5": h5, "P0": p0, "H6": z["H6"], "H6v": z["H6v"]}
    R, S = summarise(T, preds, scenes, "LOYO")
    S.to_csv(out("loyo_scenes"), index=False)
    # stratified by previous-scene age
    S2 = S.merge(T.groupby(["district", "date"])["prev_frac_gap"].first().reset_index(), on=["district", "date"])
    S2["gap_bin"] = pd.cut(S2["prev_frac_gap"].fillna(99), [0, 5, 12, 30, 1e3], labels=["<=5", "6-12", "13-30", "none"])
    G = S2.groupby("gap_bin", observed=True).agg(n=("obs", "size"), **{f"mae_lr_{a}": (f"lr_{a}", lambda s: s.abs().median()) for a in preds})
    G.to_csv(out("loyo_by_gap"))

    zf = np.load(FWD)
    keep = (T["year"] >= 2019).values
    pf = {a: np.where(keep, zf[a], np.nan) for a in ("H5", "H6")}
    pf["P0"] = np.where(keep, np.where(np.isnan(T["prev_wet"].values), zf["ff_test"], T["prev_wet"].values), np.nan)
    sc_f = scenes[scenes.date.dt.year >= 2019]
    F, _ = summarise(T, pf, sc_f, "FORWARD 2019-2026", thr=False)
    E = pd.concat([R, F])
    E.to_csv(out("eval"), index=False)
    pd.set_option("display.width", 250)
    print(E.round(4).to_string(index=False), "\n\nmedian |log ratio| by previous-scene age (LOYO):\n", G.round(3))

    v = lambda df, a, c: df[df.arm == a].set_index("district")[c]
    crit = []
    def add(name, ok, det):
        crit.append({"criterion": name, "districts_pass": int(ok.sum()), "pass": bool(ok.sum() >= 3), "detail": det})
    add("X1 median|lr| H6 < H5", v(R, "H6", "median_abs_lr") < v(R, "H5", "median_abs_lr"),
        pd.concat([v(R, "H6", "median_abs_lr"), v(R, "H5", "median_abs_lr")], axis=1).round(3).values.tolist())
    add("X2 share off 2x H6 < H5", v(R, "H6", "share_off_2x") < v(R, "H5", "share_off_2x"),
        pd.concat([v(R, "H6", "share_off_2x"), v(R, "H5", "share_off_2x")], axis=1).round(3).values.tolist())
    add("X3a median|lr| H6 < P0", v(R, "H6", "median_abs_lr") < v(R, "P0", "median_abs_lr"),
        v(R, "P0", "median_abs_lr").round(3).tolist())
    add("X3b share off 2x H6 < P0", v(R, "H6", "share_off_2x") < v(R, "P0", "share_off_2x"),
        v(R, "P0", "share_off_2x").round(3).tolist())
    x4 = (v(R, "H6", "auc") >= v(R, "H5", "auc") - 0.01) & (v(R, "H6", "within_auc") >= v(R, "H5", "within_auc") - 0.01)
    crit.append({"criterion": "X4 AUC & within-AUC >= H5-0.01 (all 4)", "districts_pass": int(x4.sum()), "pass": bool(x4.all()),
                 "detail": pd.concat([v(R, "H6", "auc") - v(R, "H5", "auc"), v(R, "H6", "within_auc") - v(R, "H5", "within_auc")], axis=1).round(4).values.tolist()})
    add("FWD X1 median|lr| H6 < H5", v(F, "H6", "median_abs_lr") < v(F, "H5", "median_abs_lr"),
        pd.concat([v(F, "H6", "median_abs_lr"), v(F, "H5", "median_abs_lr")], axis=1).round(3).values.tolist())
    add("FWD X2 share off 2x H6 < H5", v(F, "H6", "share_off_2x") < v(F, "H5", "share_off_2x"),
        pd.concat([v(F, "H6", "share_off_2x"), v(F, "H5", "share_off_2x")], axis=1).round(3).values.tolist())
    C = pd.DataFrame(crit)
    C.to_csv(out("criteria"), index=False)
    print(C.to_string(index=False))


if __name__ == "__main__":
    {"features": run_features, "loyo": run_loyo, "forward": run_forward, "score": run_score}[sys.argv[1]]()
