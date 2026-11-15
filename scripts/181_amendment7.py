"""
Amendment 7 (registered 2026-10-29 in docs/dynamic_susceptibility_method.md before any run).
H7 = H6 + forcing-since-previous-map features (rain_since, rain_max_since, d_rain5, dq_trib, dq_main; past-only,
anchored to the previous scene date); H7r = H6 + rain-only anchored features (attribution); H6s = H6 refit with
random_state=1 (seed-noise control). H6 / P0 are the stored Amendment 6 predictions, unchanged.

Run:  python -u scripts/181_amendment7.py features | loyo | forward | score
Out:  data/dynamic_susceptibility/{amend7_features.npz, amend7_loyo_preds.npz, amend7_forward_preds.npz, eval_amend7_*.csv}
"""
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s179", os.path.join(HERE, "179_amendment6.py"))
a6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a6)
a4, a3, m, dc, H2, H5 = a6.a4, a6.a3, a6.m, a6.dc, a6.H2, a6.H5
NEW_R = ["rain_since", "rain_max_since", "d_rain5"]
NEW = NEW_R + ["dq_trib", "dq_main"]
H6 = a6.H6
H7, H7R = H6 + NEW, H6 + NEW_R
FEAT = os.path.join(dc.DYN, "amend7_features.npz")
LOYO = os.path.join(dc.DYN, "amend7_loyo_preds.npz")
FWD = os.path.join(dc.DYN, "amend7_forward_preds.npz")
TEST_YEARS = a6.TEST_YEARS
MARGIN = 0.01


def out(name):
    return os.path.join(dc.DYN, f"eval_amend7_{name}.csv")


# ---------------------------------------------------------------- features
def run_features():
    """Per-cell rain from the cell's nearest IMERG grid point, exactly as scripts/170 (met_features + nearest-point map);
    discharge is district-level, as in scripts/170."""
    import rasterio
    T = a6.load(a3.a2.load_amended())
    F = {k: np.full(len(T), np.nan, np.float32) for k in NEW}
    for d in dc.DISTRICTS:
        met = pd.read_parquet(os.path.join(dc.DYN, "met", f"{d}_daily_met.parquet"))
        met["date"] = pd.to_datetime(met["date"])
        piv = met.pivot_table(index="date", columns=["lat", "lon"], values="rain_mm").asfreq("D")
        pts = np.array([list(c) for c in piv.columns])
        R = piv.values                                            # (days, points)
        day0 = piv.index[0]
        z = np.load(os.path.join(dc.DYN, "labels", d, "_grid.npz"))
        tr = rasterio.Affine(*z["transform"])
        S = np.load(os.path.join(dc.DYN, "static", f"{d}_static.npz"))
        h, w = S["usable"].shape
        qs = []
        for tag in a6.GLOFAS[d]:
            q = pd.read_csv(os.path.join(dc.DATA, "glofas_dyn", f"{tag}.csv"), parse_dates=["date"]).set_index("date")["discharge_m3s"].asfreq("D")
            qs.append(q / q["2015":"2024"].quantile(0.9))
        dm = (T["district"] == d).values
        cellv = T["cell"].values
        rows_by_date = T[dm].groupby("date").indices
        base = np.flatnonzero(dm)
        n_ok = 0; chk = []
        for t, ix in rows_by_date.items():
            r = base[ix]
            ii = cellv[r]
            lat = tr.f + (ii // w + 0.5) * tr.e; lon = tr.c + (ii % w + 0.5) * tr.a
            pi = np.argmin((lat[:, None] - pts[:, 0]) ** 2 + (lon[:, None] - pts[:, 1]) ** 2, axis=1)
            tt = (t - day0).days
            chk.append(np.nanmax(np.abs(T["r1"].values[r] - R[tt - 1, pi])) < 1e-3)     # model's r1 == rain on day t-1
            gap = T["prev_frac_gap"].values[r[0]]
            if np.isnan(gap):
                continue
            ta, tb = tt - int(gap), tt - 1                         # previous scene day index, t-1
            win = R[ta + 1: tb + 1]                                 # (previous scene day, t-1]
            if len(win):                                            # gap of 1 day: no whole day between the two scenes
                F["rain_since"][r] = np.nansum(win, 0)[pi]
                F["rain_max_since"][r] = np.nanmax(win, 0)[pi]
            else:
                F["rain_since"][r] = 0.0; F["rain_max_since"][r] = 0.0
            F["d_rain5"][r] = (np.nansum(R[tb - 4: tb + 1], 0) - np.nansum(R[ta - 4: ta + 1], 0))[pi]
            a_, b_ = t - pd.Timedelta(days=int(gap)), t - pd.Timedelta(days=1)
            for k, q in zip(("dq_trib", "dq_main"), qs):
                F[k][r] = q.get(b_, np.nan) - q.get(a_, np.nan)
            n_ok += 1
        print(f"{d}: anchored features for {n_ok} scene dates; model r1 == rain[t-1] at the same point on {np.mean(chk):.1%} of scene dates", flush=True)
    np.savez_compressed(FEAT, **F)
    print("coverage:", {k: round(float(np.mean(~np.isnan(v))), 3) for k, v in F.items()})


def load(T):
    T = a6.load(T)
    z = np.load(FEAT)
    for k in NEW:
        T[k] = z[k]
    return T


def fit(cols, tr, seed=0):
    return XGBClassifier(**{**m.XGB, "random_state": seed}).fit(tr[cols], tr["y"])


# ---------------------------------------------------------------- LOYO
def run_loyo():
    T = load(a3.a2.load_amended())
    z = np.load(a6.HIST)
    years = sorted(T["year"].unique())
    ff = {Y: z[f"ff_{Y}"] for Y in years}; bs = {Y: z[f"bs_{Y}"] for Y in years}
    P = {a: np.full(len(T), np.nan) for a in ("H7", "H7r", "H6s")}
    for Y in years:
        tem = (T["year"] == Y).values
        X = T[H2 + a6.PREV + NEW].assign(ff=ff[Y], b0s=bs[Y], y=T["y"].values)
        for a, cols, seed in (("H7", H7, 0), ("H7r", H7R, 0), ("H6s", H6, 1)):
            t0 = time.time()
            P[a][tem] = fit(cols, X[~tem], seed).predict_proba(X.loc[tem, cols])[:, 1]
            print(f"LOYO {a} {Y}: {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(LOYO, **P)                            # checkpoint after every year
    print("LOYO done")


# ---------------------------------------------------------------- forward (past-only), history as scripts/179
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
    P = {a: np.full(len(T), np.nan) for a in ("H7",)}
    for Y in TEST_YEARS:
        trm, tem = (T["year"] < Y).values, (T["year"] == Y).values
        X = T[H2 + a6.PREV + NEW].assign(ff=ff[Y], b0s=bs[Y], y=T["y"].values)
        P["H7"][tem] = fit(H7, X[trm]).predict_proba(X.loc[tem, H7])[:, 1]
        print(f"forward {Y} done", flush=True)
    np.savez_compressed(FWD, **P)


# ---------------------------------------------------------------- scoring
def run_score():
    T = load(a3.a2.load_amended())
    scenes = a6.scene_set()
    z6 = np.load(a6.LOYO); z7 = np.load(LOYO)
    h5 = np.load(os.path.join(dc.DYN, "amend4_h5_loyo_preds.npz"))["p"]
    p0 = np.where(np.isnan(T["prev_wet"].values), z6["ff_test"], T["prev_wet"].values)
    preds = {"H5": h5, "P0": p0, "H6": z6["H6"], "H7": z7["H7"], "H7r": z7["H7r"], "H6s": z7["H6s"]}
    R, S = a6.summarise(T, preds, scenes, "LOYO")
    S.to_csv(out("loyo_scenes"), index=False)

    zf6 = np.load(a6.FWD); zf7 = np.load(FWD)
    keep = (T["year"] >= 2019).values
    pf = {"H5": np.where(keep, zf6["H5"], np.nan), "H6": np.where(keep, zf6["H6"], np.nan), "H7": np.where(keep, zf7["H7"], np.nan)}
    pf["P0"] = np.where(keep, np.where(np.isnan(T["prev_wet"].values), zf6["ff_test"], T["prev_wet"].values), np.nan)
    F, SF = a6.summarise(T, pf, scenes[scenes.date.dt.year >= 2019], "FORWARD 2019-2026", thr=False)
    E = pd.concat([R, F]); E.to_csv(out("eval"), index=False)
    pd.set_option("display.width", 250)
    print(E.round(4).to_string(index=False))

    # outcome-conditioned strata (report-only): same definition as the decomposition
    pf_ = T.groupby(["district", "date"])["prev_frac"].first().reset_index().rename(columns={"prev_frac": "prev_frac"})
    X = S.merge(pf_, on=["district", "date"], how="left")
    rt = (X.obs + 0.5) / (X.prev_frac * 100 + 0.5)
    X["trans"] = np.where(X.prev_frac.isna(), "no_prev", np.where(rt >= 2, "rising>=2x", np.where(rt <= 0.5, "falling<=0.5x", "steady")))
    rows = []
    for tr, g in X.groupby("trans"):
        for a in ("H5", "P0", "H6", "H7"):
            rows.append({"trans": tr, "arm": a, "n": len(g), "median_abs_lr": g[f"lr_{a}"].abs().median(),
                         "share_off_2x": (g[f"lr_{a}"].abs() > np.log10(2)).mean(),
                         "share_under_2x": (g[f"lr_{a}"] < -np.log10(2)).mean(), "share_over_2x": (g[f"lr_{a}"] > np.log10(2)).mean()})
    ST = pd.DataFrame(rows); ST.to_csv(out("strata"), index=False)
    print("\nstrata (report-only, outcome-conditioned):\n", ST.round(3).to_string(index=False))

    v = lambda df, a, c: df[df.arm == a].set_index("district")[c]
    crit = []
    def add(name, ok, det, need=3):
        crit.append({"criterion": name, "districts_pass": int(ok.sum()), "pass": bool(ok.sum() >= need), "detail": det})
    def pair(df, a, b, c):
        return pd.concat([v(df, a, c), v(df, b, c)], axis=1).round(3).values.tolist()
    add("Z1 median|lr| H7 < H6 - 0.01", v(R, "H7", "median_abs_lr") < v(R, "H6", "median_abs_lr") - MARGIN, pair(R, "H7", "H6", "median_abs_lr"))
    add("Z2 share off 2x H7 < H6 - 0.01", v(R, "H7", "share_off_2x") < v(R, "H6", "share_off_2x") - MARGIN, pair(R, "H7", "H6", "share_off_2x"))
    add("Z3a median|lr| H7 < P0", v(R, "H7", "median_abs_lr") < v(R, "P0", "median_abs_lr"), pair(R, "H7", "P0", "median_abs_lr"))
    add("Z3b share off 2x H7 < P0", v(R, "H7", "share_off_2x") < v(R, "P0", "share_off_2x"), pair(R, "H7", "P0", "share_off_2x"))
    z4 = (v(R, "H7", "auc") >= v(R, "H6", "auc") - 0.01) & (v(R, "H7", "within_auc") >= v(R, "H6", "within_auc") - 0.01)
    add("Z4 AUC & within-AUC >= H6-0.01 (all 4)", z4, pd.concat([v(R, "H7", "auc") - v(R, "H6", "auc"), v(R, "H7", "within_auc") - v(R, "H6", "within_auc")], axis=1).round(4).values.tolist(), need=4)
    gain = v(R, "H6", "median_abs_lr") - v(R, "H7", "median_abs_lr"); noise = (v(R, "H6s", "median_abs_lr") - v(R, "H6", "median_abs_lr")).abs()
    add("Z5 gain on Z1 metric > |H6s - H6| noise floor", gain > noise, pd.concat([gain, noise], axis=1).round(4).values.tolist())
    add("FWD Z1 median|lr| H7 < H6 - 0.01", v(F, "H7", "median_abs_lr") < v(F, "H6", "median_abs_lr") - MARGIN, pair(F, "H7", "H6", "median_abs_lr"))
    add("FWD Z2 share off 2x H7 < H6 - 0.01", v(F, "H7", "share_off_2x") < v(F, "H6", "share_off_2x") - MARGIN, pair(F, "H7", "H6", "share_off_2x"))
    add("FWD Z3a median|lr| H7 < P0", v(F, "H7", "median_abs_lr") < v(F, "P0", "median_abs_lr"), pair(F, "H7", "P0", "median_abs_lr"))
    add("FWD Z3b share off 2x H7 < P0", v(F, "H7", "share_off_2x") < v(F, "P0", "share_off_2x"), pair(F, "H7", "P0", "share_off_2x"))
    C = pd.DataFrame(crit); C.to_csv(out("criteria"), index=False)
    print("\n", C.to_string(index=False))


if __name__ == "__main__":
    {"features": run_features, "loyo": run_loyo, "forward": run_forward, "score": run_score}[sys.argv[1]]()
