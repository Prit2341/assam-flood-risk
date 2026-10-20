"""
Amendment 2 (exploratory, docs/dynamic_susceptibility_method.md): checks K, G, S and the slope sensitivity row.
The primary P1-P6 result (scripts/170, criteria.csv) is not touched; everything here goes to eval_amend2_*.csv.

  K  Karimganj main-stem discharge: fill q_main from GloFAS karimganj_barak (scripts/145), rerun H2.
  G  grouped leave-one-covariate-out ablation of H2 (Lei et al. 2018 JASA; Gregorutti et al. 2015 CSDA).
  S  spatially blocked CV, 0.1 deg blocks, 5 folds (Roberts et al. 2017 Ecography).
  slope  A2 with district-quintile slope ranks instead of the paper's degree breaks (sensitivity only).

Run:  python -u scripts/171_amendment2.py kg | s | slope | models
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location("s170", os.path.join(HERE, "170_dynamic_susceptibility.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
dc = m.dc

RAIN = ["r1", "r3", "r5", "r10", "r30"]
SM = ["sm1"]
Q = ["q_trib", "q_trib_max7", "q_main", "q_main_max7"]
H1 = m.STATIC_RAW
H2 = m.STATIC_RAW + m.DYN
ARMS_G = {"H2": H2, "H2-norain": [f for f in H2 if f not in RAIN],
          "H2-nosm": [f for f in H2 if f not in SM], "H2-noq": [f for f in H2 if f not in Q]}


def out(name):
    return os.path.join(dc.DYN, f"eval_amend2_{name}.csv")


def load_amended():
    """table.parquet with Karimganj q_main/q_main_max7 filled from the Barak main-stem point (check K)."""
    T = pd.read_parquet(os.path.join(dc.DYN, "table.parquet"))
    g = m.glofas_features("karimganj_barak")
    if g is None:
        sys.exit("karimganj_barak.csv missing: run scripts/145 first")
    k = (T["district"] == "Karimganj").values
    T.loc[k, "q_main"] = g["q"].reindex(T.loc[k, "date"]).values
    T.loc[k, "q_main_max7"] = g["qmax7"].reindex(T.loc[k, "date"]).values
    print("q_main NaN rate after fill:", T.groupby("district")["q_main"].apply(lambda s: s.isna().mean()).round(3).to_dict())
    return T


def barak_vs_kushiyara():
    rd = lambda t: pd.read_csv(os.path.join(dc.DATA, "glofas_dyn", f"{t}.csv"), parse_dates=["date"]).set_index("date")["discharge_m3s"]
    b, k = rd("karimganj_barak"), rd("karimganj_kushiyara")
    j = pd.concat([b, k], axis=1, keys=["barak", "kushiyara"]).dropna()
    pts = json.load(open(os.path.join(dc.DATA, "glofas_dyn", "points.json")))
    r = {"corr_daily": round(float(j.corr().iloc[0, 1]), 4), "mean_barak": round(float(j["barak"].mean()), 1),
         "mean_kushiyara": round(float(j["kushiyara"].mean()), 1), "n_days": len(j),
         "barak_cell": pts.get("karimganj_barak"), "kushiyara_cell": pts.get("karimganj_kushiyara")}
    print("K check, Barak main stem vs Kushiyara:", r, flush=True)
    return r


def fit_predict(cols, tr, te):
    mdl = XGBClassifier(**m.XGB).fit(tr[cols], tr["y"])
    return mdl.predict_proba(tr[cols])[:, 1], mdl.predict_proba(te[cols])[:, 1]


def pooled(T, arms, fold_of, label):
    """Pooled held-out predictions over folds (fold_of: array of fold ids per row), scored per district.
    Same thresholding as scripts/170 pooled_loyo: F1-optimal threshold from each fold's training rows."""
    preds = {a: np.full(len(T), np.nan) for a in arms}
    thr = {a: np.zeros(len(T)) for a in arms}
    for f in np.unique(fold_of):
        tem = fold_of == f
        for a, cols in arms.items():
            ptr, pte = fit_predict(cols, T[~tem], T[tem])
            preds[a][tem] = pte
            thr[a][tem] = m.best_f1_threshold(T[~tem]["y"].values, ptr)
        print(f"{label} fold {f} done", flush=True)
    rows = []
    for d in dc.DISTRICTS:
        dm = (T["district"] == d).values
        s, y = T[dm], T[dm]["y"].values
        for a in arms:
            r = m.score(s, preds[a][dm], 0)
            fl = preds[a][dm] >= thr[a][dm]
            tp = (fl & (y == 1)).sum(); fp = (fl & (y == 0)).sum(); fn = (~fl & (y == 1)).sum()
            r["csi"] = tp / (tp + fp + fn); r["f1"] = 2 * tp / (2 * tp + fp + fn)
            rows.append({"cv": label, "district": d, "arm": a, **r})
    return pd.DataFrame(rows)


def lodo(T, arms):
    rows = []
    for d in dc.DISTRICTS:
        tr, te = T[T["district"] != d], T[T["district"] == d]
        for a, cols in arms.items():
            ptr, pte = fit_predict(cols, tr, te)
            rows.append({"cv": "LODO", "district": d, "arm": a, **m.score(te, pte, m.best_f1_threshold(tr["y"].values, ptr))})
        print(f"LODO {d} done", flush=True)
    return pd.DataFrame(rows)


def run_kg():
    kchk = barak_vs_kushiyara()
    json.dump(kchk, open(os.path.join(dc.DYN, "eval_amend2_K_barak_check.json"), "w"), indent=1)
    T = load_amended()
    P = pooled(T, ARMS_G, T["year"].values, "LOYO")
    L = lodo(T, {"H1": H1, **ARMS_G})
    E = pd.concat([P, L], ignore_index=True)
    E.to_csv(out("kg"), index=False)
    v = lambda cv, a, col: E[(E.cv == cv) & (E.arm == a)].set_index("district")[col]
    c = [{"check": "K1 H2 LODO AUC Karimganj >= 0.790", "value": round(v("LODO", "H2", "auc")["Karimganj"], 4),
          "pass": bool(v("LODO", "H2", "auc")["Karimganj"] >= 0.790)},
         {"check": "K2 H2 LOYO AUC Karimganj >= 0.900", "value": round(v("LOYO", "H2", "auc")["Karimganj"], 4),
          "pass": bool(v("LOYO", "H2", "auc")["Karimganj"] >= 0.900)}]
    for g in ("norain", "nosm", "noq"):
        da = v("LOYO", "H2", "auc") - v("LOYO", f"H2-{g}", "auc")
        ds = v("LOYO", "H2", "spearman") - v("LOYO", f"H2-{g}", "spearman")
        ok = (da >= 0.01) | (ds >= 0.05)
        c.append({"check": f"G {g}: group contributes (AUC drop>=0.01 or Spearman drop>=0.05) in >=3/4",
                  "value": json.dumps({"auc_drop": da.round(4).to_dict(), "spearman_drop": ds.round(3).to_dict()}),
                  "pass": bool(ok.sum() >= 3)})
    C = pd.DataFrame(c)
    C.to_csv(out("kg_criteria"), index=False)
    print(E.round(3).to_string(index=False)); print(C.to_string(index=False))


def block_folds(T, deg=0.1, k=5):
    """Fold id per row from a 0.1 deg block of the cell centre; blocks randomly assigned to k folds (seed 0)."""
    rng = np.random.default_rng(0)
    fold = np.zeros(len(T), int)
    for d in dc.DISTRICTS:
        dm = (T["district"] == d).values
        z = np.load(os.path.join(dc.DYN, "labels", d, "_grid.npz"))
        tr = rasterio.Affine(*z["transform"])
        w = np.load(os.path.join(dc.DYN, "static", f"{d}_static.npz"))["usable"].shape[1]
        cell = T.loc[dm, "cell"].values
        r, cc = cell // w, cell % w
        lat, lon = tr.f + (r + 0.5) * tr.e, tr.c + (cc + 0.5) * tr.a
        b = np.floor(lat / deg).astype(int) * 100000 + np.floor(lon / deg).astype(int)
        ub = np.unique(b)
        assign = dict(zip(ub, rng.integers(0, k, len(ub))))
        fold[dm] = [assign[x] for x in b]
        print(f"{d}: {len(ub)} blocks", flush=True)
    return fold


def run_s():
    T = load_amended()
    fold = block_folds(T)
    S = pooled(T, {"H1": H1, "H2": H2}, fold, "BLOCK")
    for a in ("A1", "A2"):                               # AHP indices need no fitting; scored for reference
        for d in dc.DISTRICTS:
            s = T[T["district"] == d]
            S.loc[len(S)] = {"cv": "BLOCK", "district": d, "arm": a, **m.score(s, s[a].values, np.inf)}
    S.to_csv(out("s"), index=False)
    v = lambda a, col: S[S.arm == a].set_index("district")[col]
    c = pd.DataFrame([
        {"check": "S1 H2 pooled AUC > H1 (block CV)", "detail": (v("H2", "auc") - v("H1", "auc")).round(4).to_dict(),
         "districts_pass": int((v("H2", "auc") > v("H1", "auc")).sum())},
        {"check": "S2 H2 within-scene AUC >= H1-0.02 (block CV)", "detail": (v("H2", "within_auc") - v("H1", "within_auc")).round(4).to_dict(),
         "districts_pass": int((v("H2", "within_auc") >= v("H1", "within_auc") - 0.02).sum())}])
    c["pass"] = c["districts_pass"] >= 3
    c.to_csv(out("s_criteria"), index=False)
    print(S.round(3).to_string(index=False)); print(c.to_string(index=False))


def run_slope():
    T = pd.read_parquet(os.path.join(dc.DYN, "table.parquet"))
    w = json.load(open(os.path.join(dc.DYN, "ahp_weights.json")))["weights_10"]
    rows = []
    for d in dc.DISTRICTS:
        S = dict(np.load(os.path.join(dc.DYN, "static", f"{d}_static.npz")))
        valid = S["usable"] & np.all([~np.isnan(S[k]) for k in m.STATIC_RAW], axis=0)
        q = np.nanpercentile(S["slope"][valid], [20, 40, 60, 80])
        s = T[T["district"] == d]
        rk_q = 6 - (1 + np.digitize(s["slope"].values, q))       # steeper = lower susceptibility, as the paper
        a2q = s["A2"].values - w["slope"] * s["rk_slope"].values + w["slope"] * rk_q
        rows.append({"district": d, "slope_quintile_breaks_deg": np.round(q, 3).tolist(),
                     "A2_base_auc": round(roc_auc_score(s["y"], s["A2"]), 4),
                     "A2_slope_quintile_auc": round(roc_auc_score(s["y"], a2q), 4)})
    R = pd.DataFrame(rows)
    R.to_csv(out("slope"), index=False)
    print(R.to_string(index=False))


def run_st():
    """Beyond Amendment 2 (added after S, prompted by the shared-date confound in S): spatio-temporal blocking.
    Fold k tests rows in block-fold k AND year-group k; training excludes BOTH block-fold k and year-group k,
    so neither the cells' blocks nor the scene's date were seen in training (Roberts et al. 2017)."""
    T = load_amended()
    bf = block_folds(T)
    yrs = sorted(T["year"].unique())
    rng = np.random.default_rng(0)
    yg_of = dict(zip(yrs, rng.permutation(np.arange(len(yrs)) % 5)))
    yg = T["year"].map(yg_of).values
    print("year groups:", {int(k): int(v) for k, v in yg_of.items()}, flush=True)
    preds = {a: np.full(len(T), np.nan) for a in ("H1", "H2")}
    for k in range(5):
        te = (bf == k) & (yg == k)
        tr = (bf != k) & (yg != k)
        for a, cols in (("H1", H1), ("H2", H2)):
            preds[a][te] = fit_predict(cols, T[tr], T[te])[1]
        print(f"ST fold {k}: train {tr.sum()} test {te.sum()}", flush=True)
    rows = []
    for d in dc.DISTRICTS:
        dm = ((T["district"] == d).values) & ~np.isnan(preds["H2"])
        for a in preds:
            r = m.score(T[dm], preds[a][dm], np.inf)
            # test sets hold ~1/5 of each scene's 2000 cells, so 170's >=1000-cell rule would leave no scenes;
            # same within-scene AUC but >=200 cells, identical rule for H1 and H2
            g = T[dm].assign(p=preds[a][dm]).groupby("date")
            w = [roc_auc_score(x["y"], x["p"]) for _, x in g if 0.01 <= x["y"].mean() < 1 and len(x) >= 200]
            r["within_auc"], r["n_scenes_within"] = (np.mean(w) if w else np.nan), len(w)
            rows.append({"cv": "SPACE-TIME", "district": d, "arm": a, **r})
    R = pd.DataFrame(rows)
    R.to_csv(out("st"), index=False)
    v = lambda a, col: R[R.arm == a].set_index("district")[col]
    print(R.round(3).to_string(index=False))
    print("within-scene H2-H1:", (v("H2", "within_auc") - v("H1", "within_auc")).round(4).to_dict())
    print("pooled AUC H2-H1:", (v("H2", "auc") - v("H1", "auc")).round(4).to_dict())


def run_models():
    """One H2 model per held-out year (trained on all other years, K-amended table) for the dashboard date map,
    so a past date is always mapped by a model that never saw that year."""
    T = load_amended()
    od = os.path.join(dc.DYN, "models_h2_loyo")
    os.makedirs(od, exist_ok=True)
    for yr in sorted(T["year"].unique()):
        mdl = XGBClassifier(**m.XGB).fit(T.loc[T["year"] != yr, H2], T.loc[T["year"] != yr, "y"])
        mdl.save_model(os.path.join(od, f"h2_without_{yr}.json"))
        print(f"model without {yr} saved", flush=True)
    json.dump({"features": H2, "years": [int(y) for y in sorted(T["year"].unique())]},
              open(os.path.join(od, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    {"kg": run_kg, "s": run_s, "slope": run_slope, "models": run_models, "st": run_st}[sys.argv[1]]()
