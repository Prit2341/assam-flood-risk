"""
Amendment 5: leave-one-district-out (LODO) threshold transfer.
Step `diag` (descriptive, run before registration): why does H2's F1 threshold not transfer to a held-out district?
Fits H2 LODO on the K-amended table (Amendment 2), saves predictions, reports per held-out district:
prevalence, mean predicted probability, share flagged at the transferred threshold, CSI at that threshold,
and the oracle (test-optimal) CSI as an upper bound. Also the per-district range of the discharge features.

Run:  python -u scripts/174_lodo_threshold.py diag
Out:  data/dynamic_susceptibility/{amend5_lodo_h2_preds.npz, eval_amend5_diag.csv, eval_amend5_diag_q.csv}
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s172", os.path.join(HERE, "172_amendment3.py"))
a3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a3)
m, dc, H2 = a3.m, a3.dc, a3.H2
PREDS = os.path.join(dc.DYN, "amend5_lodo_h2_preds.npz")


def out(name):
    return os.path.join(dc.DYN, f"eval_amend5_{name}.csv")


def csi(y, f):
    tp = (f & (y == 1)).sum(); fp = (f & (y == 0)).sum(); fn = (~f & (y == 1)).sum()
    return tp / (tp + fp + fn) if tp + fp + fn else np.nan


def lodo_h2(T):
    """H2 trained on 3 districts, predicted on the 4th; also the training-set predictions for threshold work."""
    if os.path.exists(PREDS):
        z = np.load(PREDS)
        return z["p"], z["thr"], z["ptr_mean"]
    p = np.full(len(T), np.nan)
    thr = np.zeros(len(T))
    ptr_mean = np.zeros(len(T))
    for d in dc.DISTRICTS:
        te = (T["district"] == d).values
        mdl = a3.fit_xgb(H2, T[~te])
        ptr = mdl.predict_proba(T.loc[~te, H2])[:, 1]
        p[te] = mdl.predict_proba(T.loc[te, H2])[:, 1]
        thr[te] = m.best_f1_threshold(T.loc[~te, "y"].values, ptr)
        ptr_mean[te] = ptr.mean()
        print(f"LODO H2 {d} done", flush=True)
    np.savez_compressed(PREDS, p=p, thr=thr, ptr_mean=ptr_mean)
    return p, thr, ptr_mean


def run_diag():
    T = a3.a2.load_amended()
    p, thr, ptr_mean = lodo_h2(T)
    rows = []
    for d in dc.DISTRICTS:
        dm = (T["district"] == d).values
        y, pd_ = T.loc[dm, "y"].values, p[dm]
        t = thr[dm][0]
        ot = m.best_f1_threshold(y, pd_)
        rows.append({"district": d, "prev": y.mean(), "mean_p": pd_.mean(), "train_mean_p": ptr_mean[dm][0],
                     "thr_transfer": t, "flag_rate_transfer": (pd_ >= t).mean(), "csi_transfer": csi(y, pd_ >= t),
                     "thr_oracle": ot, "flag_rate_oracle": (pd_ >= ot).mean(), "csi_oracle": csi(y, pd_ >= ot),
                     "p_q50": np.quantile(pd_, 0.5), "p_q99": np.quantile(pd_, 0.99)})
    D = pd.DataFrame(rows)
    D.to_csv(out("diag"), index=False)
    print(D.round(4).to_string(index=False))
    Q = T.groupby("district")[["q_trib", "q_main", "r5", "sm1", "elevation", "hand"]].describe(percentiles=[.5, .9]).T
    Q.to_csv(out("diag_q"))
    print(Q.round(2).to_string())


# ---------------------------------------------------------------- Amendment 5 run (registered in the method doc)
H2INV = [f for f in H2 if f != "elevation"]
FEATS = {"H2": H2, "H2inv": H2INV}
RUN = os.path.join(dc.DYN, "amend5_lodo_preds.npz")


def saerens_em(p, prior_tr, iters=200, tol=1e-7):
    """Saerens, Latinne & Decaestecker 2002: re-estimate the target prior from unlabelled posteriors, return
    (adjusted posteriors, estimated prior)."""
    q = prior_tr
    for _ in range(iters):
        a = p * q / prior_tr
        b = (1 - p) * (1 - q) / (1 - prior_tr)
        pa = a / (a + b)
        qn = pa.mean()
        if abs(qn - q) < tol:
            break
        q = qn
    return pa, q


def lodo_nested(T):
    """Outer LODO per feature set: target predictions, in-sample threshold, nested (inner-LODO) threshold,
    inner held-out flag rate, training prevalence."""
    if os.path.exists(RUN):
        z = np.load(RUN, allow_pickle=True)
        return z["res"].item()
    res = {}
    for name, cols in FEATS.items():
        for d in dc.DISTRICTS:
            te = (T["district"] == d).values
            mdl = a3.fit_xgb(cols, T[~te])
            ptr = mdl.predict_proba(T.loc[~te, cols])[:, 1]
            inner_p, inner_y = [], []
            for d2 in [x for x in dc.DISTRICTS if x != d]:
                tr2 = ~te & (T["district"] != d2).values
                te2 = (T["district"] == d2).values
                mi = a3.fit_xgb(cols, T[tr2])
                inner_p.append(mi.predict_proba(T.loc[te2, cols])[:, 1]); inner_y.append(T.loc[te2, "y"].values)
                print(f"{name} outer {d} inner {d2} done", flush=True)
            ip, iy = np.concatenate(inner_p), np.concatenate(inner_y)
            tn = m.best_f1_threshold(iy, ip)
            res[(name, d)] = dict(p=mdl.predict_proba(T.loc[te, cols])[:, 1],
                                  thr_in=m.best_f1_threshold(T.loc[~te, "y"].values, ptr), thr_nested=tn,
                                  inner_rate=(ip >= tn).mean(), prior_tr=T.loc[~te, "y"].mean())
            print(f"{name} outer {d} done", flush=True)
    np.savez_compressed(RUN, res=np.array(res, dtype=object))
    return res


def loyo_h2inv(T):
    f = os.path.join(dc.DYN, "amend5_h2inv_loyo_preds.npz")
    if os.path.exists(f):
        return np.load(f)["p"]
    p = np.full(len(T), np.nan)
    for yr in sorted(T["year"].unique()):
        tem = (T["year"] == yr).values
        p[tem] = a3.fit_xgb(H2INV, T[~tem]).predict_proba(T.loc[tem, H2INV])[:, 1]
        print(f"H2inv LOYO {yr} done", flush=True)
    np.savez_compressed(f, p=p)
    return p


def run():
    T = a3.a2.load_amended()
    D = pd.read_csv(out("diag")).set_index("district")
    res = lodo_nested(T)
    rows = []
    for d in dc.DISTRICTS:
        te = (T["district"] == d).values
        s, y = T[te], T.loc[te, "y"].values
        for name in FEATS:
            r = res[(name, d)]
            p = r["p"]
            pem, qem = saerens_em(p, r["prior_tr"])
            arms = {f"{name}/in-sample": (p, p >= r["thr_in"]),
                    f"{name}/nested": (p, p >= r["thr_nested"]),
                    f"{name}/EM [report]": (pem, pem >= r["thr_nested"]),
                    f"{name}/rate [report]": (p, p >= np.quantile(p, 1 - r["inner_rate"]))}
            for arm, (pp, fl) in arms.items():
                sc = a3.scores(s, pp, fl)
                rows.append({"district": d, "arm": arm, "auc": sc["auc"], "within_auc": sc["within_auc"],
                             "csi": sc["csi"], "f1": sc["f1"], "flag_rate": fl.mean(), "prev": y.mean(),
                             "thr_in": r["thr_in"], "thr_nested": r["thr_nested"],
                             "em_prior": qem if "EM" in arm else np.nan})
    E = pd.DataFrame(rows)
    E.to_csv(out("lodo"), index=False)
    print(E.round(4).to_string(index=False))

    # LOYO AUC check for H2inv (M4) against Amendment 3 H2 LOYO predictions
    ph2, _ = a3.load_preds()
    pinv = loyo_h2inv(T)
    L = []
    for d in dc.DISTRICTS:
        dm = (T["district"] == d).values
        L.append({"district": d, "auc_h2": m.score(T[dm], ph2[dm], np.inf)["auc"],
                  "auc_h2inv": m.score(T[dm], pinv[dm], np.inf)["auc"]})
    L = pd.DataFrame(L)
    L.to_csv(out("loyo"), index=False)

    g = lambda arm, d, c: E[(E.arm == arm) & (E.district == d)][c].iloc[0]
    P, B = "H2inv/nested", "H2/in-sample"
    crit = [
        {"criterion": "M1 CSI >= 0.5 x H2 oracle CSI in Nagaon AND Sivasagar",
         "detail": {d: (round(g(P, d, "csi"), 4), round(0.5 * D.loc[d, "csi_oracle"], 4)) for d in ("Nagaon", "Sivasagar")},
         "pass": all(g(P, d, "csi") >= 0.5 * D.loc[d, "csi_oracle"] for d in ("Nagaon", "Sivasagar"))},
        {"criterion": "M2 CSI >= H2 in-sample CSI - 0.02 in Morigaon and Karimganj",
         "detail": {d: (round(g(P, d, "csi"), 4), round(g(B, d, "csi"), 4)) for d in ("Morigaon", "Karimganj")},
         "pass": all(g(P, d, "csi") >= g(B, d, "csi") - 0.02 for d in ("Morigaon", "Karimganj"))},
        {"criterion": "M3 LODO AUC >= H2 LODO AUC - 0.02 in all 4",
         "detail": {d: (round(g(P, d, "auc"), 4), round(g(B, d, "auc"), 4)) for d in dc.DISTRICTS},
         "pass": all(g(P, d, "auc") >= g(B, d, "auc") - 0.02 for d in dc.DISTRICTS)},
        {"criterion": "M4 H2inv LOYO AUC >= H2 LOYO AUC - 0.01 in >=3/4",
         "detail": L.round(4).set_index("district").to_dict("index"),
         "pass": int((L.auc_h2inv >= L.auc_h2 - 0.01).sum()) >= 3}]
    C = pd.DataFrame(crit)
    C.to_csv(out("criteria"), index=False)
    print(C.to_string(index=False))


if __name__ == "__main__":
    {"diag": run_diag, "run": run}[sys.argv[1]]()
