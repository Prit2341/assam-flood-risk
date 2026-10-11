"""
AHP-based dynamic flood susceptibility: build the space-time table, train/evaluate A1, A2, H1, H2, H3,
H3-pure, H4 (+ RF-phase1 reference) under leave-one-year-out (LOYO) and leave-one-district-out (LODO),
score the pre-registered criteria P1-P6 (docs/dynamic_susceptibility_method.md, Amendment 1).

Inputs: scripts 130 (labels), 140 (static), 145 (GloFAS), 150 (IMERG/ERA5-Land), 160 (AHP weights).
Run:  python -u scripts/170_dynamic_susceptibility.py table | eval | sensitivity
Out:  data/dynamic_susceptibility/{table.parquet, eval_*.csv, criteria.csv, sensitivity.csv}
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dyn_common as dc

RNG = np.random.default_rng(0)
N_PER_SCENE = 2000
MIN_OBS_FRAC = 0.20
STATIC_RAW = ["elevation", "slope", "twi", "sti", "tri", "log_acc", "hand", "dist_river", "dist_major",
              "drain_dens", "soil_hsg", "ndvi", "lulc"]
DYN = ["r1", "r3", "r5", "r10", "r30", "sm1", "q_trib", "q_trib_max7", "q_main", "q_main_max7"]
AHP_F = ["slope", "elevation", "rainfall", "lulc", "ndvi", "soil", "drain_dens", "dist_river", "tri", "sti"]
GLOFAS = {"Morigaon": ("morigaon_kopili", "morigaon_brahmaputra"), "Nagaon": ("nagaon_kopili", "nagaon_brahmaputra"),
          "Sivasagar": ("sivasagar_dikhow", "sivasagar_brahmaputra"), "Karimganj": ("karimganj_kushiyara", None)}
LULC_RANK = {0: 5, 1: 1, 2: 1, 3: 5, 4: 3, 5: 1, 6: 4, 7: 3, 8: np.nan}   # Dynamic World classes
XGB = dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
           min_child_weight=5, tree_method="hist", n_jobs=8, random_state=0, eval_metric="logloss")


# ---------------------------------------------------------------- AHP reclassification (method doc table)
def quint(x, valid, high_is_5):
    q = np.nanpercentile(x[valid], [20, 40, 60, 80])
    r = 1 + np.digitize(x, q)
    return (r if high_is_5 else 6 - r).astype(np.float32)


def ahp_static_ranks(S, valid, lulc_rank=LULC_RANK):
    R = {}
    R["slope"] = 5 - np.digitize(S["slope"], [5.45, 12.25, 19.15, 26.45])
    R["dist_river"] = 5 - np.digitize(S["dist_major"], [2.6, 5.2, 7.9, 10.5])
    R["ndvi"] = 5 - np.digitize(S["ndvi"], [0.365, 0.585])
    R["elevation"] = quint(S["elevation"], valid, high_is_5=False)
    R["drain_dens"] = quint(S["drain_dens"], valid, high_is_5=True)
    R["tri"] = quint(S["tri"], valid, high_is_5=False)
    R["sti"] = quint(S["sti"], valid, high_is_5=False)
    R["soil"] = 1 + (S["soil_hsg"] - 1) * 4 / 3
    lu = S["lulc"].copy()
    R["lulc"] = np.vectorize(lambda v: lulc_rank.get(int(v), np.nan) if not np.isnan(v) else np.nan)(lu)
    return {k: v.astype(np.float32) for k, v in R.items()}


def amc_rank(r5):
    """SCS antecedent moisture condition, growing season (NEH-4 Table 4-2): I <35.6, II 35.6-53.3, III >53.3 mm."""
    return np.where(r5 < 35.6, 1, np.where(r5 <= 53.3, 3, 5)).astype(np.float32)


# ---------------------------------------------------------------- dynamic inputs
def met_features(d):
    met = pd.read_parquet(os.path.join(dc.DYN, "met", f"{d}_daily_met.parquet"))
    met["date"] = pd.to_datetime(met["date"])
    out = {}
    for (la, lo), g in met.groupby(["lat", "lon"]):
        g = g.set_index("date").sort_index().asfreq("D")
        r = g["rain_mm"]
        f = pd.DataFrame({f"r{k}": r.rolling(k, min_periods=k).sum().shift(1) for k in (1, 3, 5, 10, 30)})
        f["sm1"] = g["swvl1"].shift(1)
        out[(la, lo)] = f
    return out


def glofas_features(tag):
    if tag is None:
        return None
    if not os.path.exists(os.path.join(dc.DATA, "glofas_dyn", f"{tag}.csv")):
        print(f"   WARNING: GloFAS {tag} missing -> NaN", flush=True)
        return None
    s = pd.read_csv(os.path.join(dc.DATA, "glofas_dyn", f"{tag}.csv"), parse_dates=["date"]).set_index("date")["discharge_m3s"]
    s = s.asfreq("D")
    q90 = s["2015":"2024"].quantile(0.9)
    a = s / q90
    return pd.DataFrame({"q": a.shift(1), "qmax7": a.rolling(7, min_periods=4).max().shift(1)})


# ---------------------------------------------------------------- table
def build_table():
    weights = json.load(open(os.path.join(dc.DYN, "ahp_weights.json")))["weights_10"]
    frames = []
    for d in dc.DISTRICTS:
        S = dict(np.load(os.path.join(dc.DYN, "static", f"{d}_static.npz")))
        usable = S.pop("usable")
        valid = usable & np.all([~np.isnan(S[k]) for k in STATIC_RAW], axis=0)
        R = ahp_static_ranks(S, valid)
        z = np.load(os.path.join(dc.DYN, "labels", d, "_grid.npz"))
        tr = rasterio.Affine(*z["transform"])
        h, w = valid.shape
        rows, cols = np.mgrid[0:h, 0:w]
        lat = tr.f + (rows + 0.5) * tr.e
        lon = tr.c + (cols + 0.5) * tr.a
        met = met_features(d)
        pts = np.array(list(met.keys()))
        near = np.argmin((lat.ravel()[:, None] - pts[:, 0]) ** 2 + (lon.ravel()[:, None] - pts[:, 1]) ** 2, axis=1).reshape(h, w) \
            if len(pts) < 400 else None
        gt, gm = (glofas_features(t) for t in GLOFAS[d])
        # static mean-rain rank for A1: Jun-Sep mean daily IMERG 2016-2025 per met point, district quintiles
        clim = np.array([met[tuple(p)]["r1"][lambda s: s.index.month.isin([6, 7, 8, 9]) & (s.index.year <= 2025)].mean() for p in pts])
        rain_clim = clim[near]
        R_rain_static = quint(rain_clim.astype(np.float32), valid, high_is_5=True)
        idx = pd.read_csv(os.path.join(dc.DYN, "labels", d, "labels_index.csv"))
        idx = idx[idx["obs_frac"] >= MIN_OBS_FRAC]
        n_sc = 0
        for day in idx["date"]:
            t = pd.Timestamp(day)
            lab = np.load(os.path.join(dc.DYN, "labels", d, f"{day}.npz"))["label"]
            obs = (lab != 255) & valid
            ii = np.flatnonzero(obs.ravel())
            if len(ii) < 1000:
                continue
            ii = RNG.choice(ii, min(N_PER_SCENE, len(ii)), replace=False)
            df = pd.DataFrame({"district": d, "date": t, "year": t.year, "cell": ii,
                               "y": lab.ravel()[ii].astype(np.int8)})
            for k in STATIC_RAW:
                df[k] = S[k].ravel()[ii]
            for k in AHP_F:
                if k != "rainfall":
                    df[f"rk_{k}"] = R[k].ravel()[ii]
            df["rk_rain_static"] = R_rain_static.ravel()[ii]
            pi = near.ravel()[ii]
            mf = np.stack([met[tuple(pts[j])].reindex([t]).iloc[0].values for j in range(len(pts))])
            mcols = list(next(iter(met.values())).columns)
            for c_i, c in enumerate(mcols):
                df[c] = mf[pi, c_i]
            for name, g in (("q_trib", gt), ("q_main", gm)):
                if g is None or t not in g.index:
                    df[name], df[f"{name}_max7"] = np.nan, np.nan
                else:
                    df[name], df[f"{name}_max7"] = g.loc[t, "q"], g.loc[t, "qmax7"]
            df["rk_rain_dyn"] = amc_rank(df["r5"].values)
            frames.append(df)
            n_sc += 1
        print(f"{d}: {n_sc} scenes used of {len(idx)} with obs_frac>={MIN_OBS_FRAC}", flush=True)
    T = pd.concat(frames, ignore_index=True)
    w = weights
    base = sum(w[k] * T[f"rk_{k}"] for k in AHP_F if k != "rainfall")
    T["A1"] = base + w["rainfall"] * T["rk_rain_static"]
    T["A2"] = base + w["rainfall"] * T["rk_rain_dyn"]
    T = T.dropna(subset=["r5", "sm1"])
    T.to_parquet(os.path.join(dc.DYN, "table.parquet"), index=False)
    print(T.groupby("district").agg(n=("y", "size"), scenes=("date", "nunique"), prev=("y", "mean"),
                                    years=("year", lambda s: f"{s.min()}-{s.max()}")))
    return T


# ---------------------------------------------------------------- evaluation
def feats(arm):
    rk = [f"rk_{k}" for k in AHP_F if k != "rainfall"]
    return {"H1": STATIC_RAW, "H2": STATIC_RAW + DYN, "H3": rk + ["rk_rain_dyn"] + ["sm1", "q_trib", "q_trib_max7", "q_main", "q_main_max7"],
            "H3pure": rk + ["rk_rain_dyn"], "H4": STATIC_RAW + DYN + ["A2"]}[arm]


def best_f1_threshold(y, p):
    qs = np.unique(np.quantile(p, np.linspace(0.5, 0.999, 200)))
    best, bt = -1, qs[0]
    for t in qs:
        f = p >= t
        tp = (f & (y == 1)).sum(); fp = (f & (y == 0)).sum(); fn = (~f & (y == 1)).sum()
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0
        if f1 > best:
            best, bt = f1, t
    return bt


def score(te, p, thr):
    y = te["y"].values
    out = {"n": len(y), "prev": y.mean(), "auc": roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan,
           "prauc": average_precision_score(y, p) if y.sum() else np.nan}
    f = p >= thr
    tp = (f & (y == 1)).sum(); fp = (f & (y == 0)).sum(); fn = (~f & (y == 1)).sum()
    out["csi"] = tp / (tp + fp + fn) if tp + fp + fn else np.nan
    out["f1"] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
    g = te.assign(p=p).groupby("date")
    within = [roc_auc_score(s["y"], s["p"]) for _, s in g if s["y"].mean() >= 0.01 and s["y"].mean() < 1 and len(s) >= 1000]
    out["within_auc"] = np.mean(within) if within else np.nan
    out["n_scenes_within"] = len(within)
    agg = g.agg(pm=("p", "mean"), ym=("y", "mean"))
    out["spearman"] = spearmanr(agg["pm"], agg["ym"]).correlation if len(agg) >= 5 else np.nan
    out["n_scenes"] = len(agg)
    return out


def predict(arm, tr, te):
    if arm in ("A1", "A2"):
        return tr[arm].values, te[arm].values
    m = XGBClassifier(**XGB)
    m.fit(tr[feats(arm)], tr["y"])
    return m.predict_proba(tr[feats(arm)])[:, 1], m.predict_proba(te[feats(arm)])[:, 1]


ARMS = ["A1", "A2", "H1", "H2", "H3", "H3pure", "H4"]


def evaluate(T):
    rows = []
    for d in dc.DISTRICTS:                               # LODO (LOYO is done in pooled_loyo)
        tr, te = T[T["district"] != d], T[T["district"] == d]
        for arm in ARMS:
            ptr, pte = predict(arm, tr, te)
            thr = best_f1_threshold(tr["y"].values, ptr)
            rows.append({"cv": "LODO", "fold": d, "district": d, "arm": arm, **score(te, pte, thr)})
        print(f"LODO {d} done", flush=True)
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(dc.DYN, "eval_folds.csv"), index=False)
    return E


def pooled_loyo(T):
    """Pooled held-out predictions per district across all LOYO folds (one prediction per row)."""
    preds = {a: np.full(len(T), np.nan) for a in ARMS}
    thr = {a: [] for a in ARMS}
    for yr in sorted(T["year"].unique()):
        trm, tem = (T["year"] != yr).values, (T["year"] == yr).values
        for arm in ARMS:
            ptr, pte = predict(arm, T[trm], T[tem])
            preds[arm][tem] = pte
            thr[arm].append((tem, best_f1_threshold(T[trm]["y"].values, ptr)))
    rows = []
    for d in dc.DISTRICTS:
        dm = (T["district"] == d).values
        for arm in ARMS:
            th = np.zeros(len(T))
            for tem, t in thr[arm]:
                th[tem] = t
            s = T[dm]
            p = preds[arm][dm]
            r = score(s, p, 0)  # placeholder thr, csi recomputed below with per-fold thresholds
            f = p >= th[dm]
            y = s["y"].values
            tp = (f & (y == 1)).sum(); fp = (f & (y == 0)).sum(); fn = (~f & (y == 1)).sum()
            r["csi"] = tp / (tp + fp + fn); r["f1"] = 2 * tp / (2 * tp + fp + fn)
            rows.append({"district": d, "arm": arm, **r})
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(dc.DYN, "eval_loyo_pooled.csv"), index=False)
    np.savez_compressed(os.path.join(dc.DYN, "loyo_preds.npz"), **preds)
    return P


def criteria(P):
    v = lambda arm, col: P[P["arm"] == arm].set_index("district")[col]
    c = []
    def add(name, ok, detail):
        n = int(ok.sum())
        c.append({"criterion": name, "districts_pass": n, "pass": n >= 3, "detail": detail})
    add("P1 H2 AUC > H1", v("H2", "auc") > v("H1", "auc"), (v("H2", "auc") - v("H1", "auc")).round(4).to_dict())
    add("P2 H2 spearman>=0.5 and >=H1+0.2", (v("H2", "spearman") >= 0.5) & (v("H2", "spearman") >= v("H1", "spearman") + 0.2),
        {"H2": v("H2", "spearman").round(3).to_dict(), "H1": v("H1", "spearman").round(3).to_dict()})
    add("P3 H2 within-scene AUC >= H1-0.02", v("H2", "within_auc") >= v("H1", "within_auc") - 0.02,
        (v("H2", "within_auc") - v("H1", "within_auc")).round(4).to_dict())
    add("P4 A2 AUC > A1", v("A2", "auc") > v("A1", "auc"), (v("A2", "auc") - v("A1", "auc")).round(4).to_dict())
    add("P5 H4 AUC > H2", v("H4", "auc") > v("H2", "auc"), (v("H4", "auc") - v("H2", "auc")).round(4).to_dict())
    add("P6 H2 AUC > A2", v("H2", "auc") > v("A2", "auc"), (v("H2", "auc") - v("A2", "auc")).round(4).to_dict())
    C = pd.DataFrame(c)
    C.to_csv(os.path.join(dc.DYN, "criteria.csv"), index=False)
    return C


# ---------------------------------------------------------------- sensitivity (AHP, no training needed)
def sensitivity(T):
    """One-at-a-time +-20% weight perturbation (renormalised) and LULC-mapping variants; pooled AUC per district
    of the dynamic AHP index A2 (Chen et al. 2021, arXiv:2107.13368 for OAT weight sensitivity in AHP)."""
    w0 = json.load(open(os.path.join(dc.DYN, "ahp_weights.json")))["weights_10"]
    col = {k: (f"rk_{k}" if k != "rainfall" else "rk_rain_dyn") for k in AHP_F}

    def auc_by_d(w, lulc_col="rk_lulc"):
        sc = sum(w[k] * (T[lulc_col] if k == "lulc" else T[col[k]]) for k in AHP_F)
        return {d: roc_auc_score(T.loc[g, "y"], sc[g]) for d, g in T.groupby("district").groups.items()}

    rows = [{"variant": "base", **auc_by_d(w0)}]
    for k in AHP_F:
        for f in (0.8, 1.2):
            w = dict(w0); w[k] *= f
            tot = sum(w.values()); w = {a: b / tot for a, b in w.items()}
            rows.append({"variant": f"{k} x{f}", **auc_by_d(w)})
    lu = T["lulc"].round()
    for name, mp in (("flooded_veg=3", {**LULC_RANK, 3: 3}), ("bare=1", {**LULC_RANK, 7: 1}), ("bare=5", {**LULC_RANK, 7: 5})):
        T["_rk_lulc_alt"] = lu.map(mp)
        rows.append({"variant": f"LULC {name}", **auc_by_d(w0, "_rk_lulc_alt")})
    rows.append({"variant": "equal weights", **auc_by_d({k: 0.1 for k in AHP_F})})
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(dc.DYN, "sensitivity.csv"), index=False)
    return S


# ---------------------------------------------------------------- final maps
def final_maps(T, dates):
    """Fit H2 and H4 on ALL rows, then map static (H1), dynamic AHP (A2) and dynamic ML (H2/H4) for given dates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    weights = json.load(open(os.path.join(dc.DYN, "ahp_weights.json")))["weights_10"]
    models = {a: XGBClassifier(**XGB).fit(T[feats(a)], T["y"]) for a in ("H1", "H2", "H4")}
    out_dir = os.path.join(dc.DYN, "maps")
    os.makedirs(out_dir, exist_ok=True)
    for d, day in dates:
        t = pd.Timestamp(day)
        S = dict(np.load(os.path.join(dc.DYN, "static", f"{d}_static.npz")))
        usable = S.pop("usable")
        valid = usable & np.all([~np.isnan(S[k]) for k in STATIC_RAW], axis=0)
        R = ahp_static_ranks(S, valid)
        z = np.load(os.path.join(dc.DYN, "labels", d, "_grid.npz"))
        tr = rasterio.Affine(*z["transform"])
        h, w = valid.shape
        rr, cc = np.mgrid[0:h, 0:w]
        lat, lon = tr.f + (rr + 0.5) * tr.e, tr.c + (cc + 0.5) * tr.a
        met = met_features(d)
        pts = np.array(list(met.keys()))
        from scipy.spatial import cKDTree
        near = cKDTree(pts).query(np.c_[lat.ravel(), lon.ravel()])[1]
        ii = np.flatnonzero(valid.ravel())
        X = pd.DataFrame({k: S[k].ravel()[ii] for k in STATIC_RAW})
        mf = np.stack([met[tuple(p)].reindex([t]).iloc[0].values for p in pts])
        for c_i, c in enumerate(next(iter(met.values())).columns):
            X[c] = mf[near[ii], c_i]
        for name, tag in zip(("q_trib", "q_main"), GLOFAS[d]):
            g = glofas_features(tag)
            X[name], X[f"{name}_max7"] = (np.nan, np.nan) if g is None or t not in g.index else (g.loc[t, "q"], g.loc[t, "qmax7"])
        rk_rain = amc_rank(X["r5"].values)
        X["A2"] = sum(weights[k] * R[k].ravel()[ii] for k in AHP_F if k != "rainfall") + weights["rainfall"] * rk_rain
        maps = {"A2_dynamic_AHP": X["A2"].values}
        for a, mdl in models.items():
            maps[f"{a}"] = mdl.predict_proba(X[feats(a)])[:, 1]
        lab_p = os.path.join(dc.DYN, "labels", d, f"{day}.npz")
        lab = np.load(lab_p)["label"].astype(float) if os.path.exists(lab_p) else None
        fig, axes = plt.subplots(1, len(maps) + (lab is not None), figsize=(5 * (len(maps) + 1), 5))
        for ax, (k, v) in zip(axes, maps.items()):
            arr = np.full(h * w, np.nan); arr[ii] = v; arr = arr.reshape(h, w)
            dc.save_tif(os.path.join(out_dir, f"{d}_{day}_{k}.tif"), arr, tr)
            ax.imshow(arr, cmap="viridis"); ax.set_title(f"{k}"); ax.axis("off")
        if lab is not None:
            lab[lab == 255] = np.nan
            axes[-1].imshow(lab, cmap="coolwarm", vmin=0, vmax=1); axes[-1].set_title("GFM observed"); axes[-1].axis("off")
        fig.suptitle(f"{d} {day}  r5={X['r5'].mean():.0f} mm  sm1={X['sm1'].mean():.2f}  q_trib={X['q_trib'].iloc[0]:.2f}")
        fig.savefig(os.path.join(out_dir, f"{d}_{day}.png"), dpi=70, bbox_inches="tight")
        plt.close(fig)
        print(f"map {d} {day} done", flush=True)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "eval"
    pd.set_option("display.width", 250)
    if step == "table":
        build_table()
    elif step == "eval":
        T = pd.read_parquet(os.path.join(dc.DYN, "table.parquet"))
        P = pooled_loyo(T)
        print(P.round(3).to_string(index=False))
        print(criteria(P).to_string(index=False))
        E = evaluate(T)
        print(E[E["cv"] == "LODO"].round(3).to_string(index=False))
    elif step == "sensitivity":
        T = pd.read_parquet(os.path.join(dc.DYN, "table.parquet"))
        print(sensitivity(T).round(4).to_string(index=False))
    elif step == "maps":
        T = pd.read_parquet(os.path.join(dc.DYN, "table.parquet"))
        final_maps(T, [tuple(a.split(":")) for a in sys.argv[2:]])
