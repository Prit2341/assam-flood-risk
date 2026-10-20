"""
H5 for the dashboard (Amendment 4 working model for labelled districts).
1. `models`: refit the Amendment 4 H5 LOYO models (same features, leakage rule and XGB seed as scripts/173) and save
   them as models_h5_loyo/h5_without_{Y}.json. Sanity check: test-row predictions must match amend4_h5_loyo_preds.npz.
2. `history`: per district and held-out year Y, full-raster history features from scenes of years != Y (the same rule
   as the test rows in Amendment 4): ff (wet / observed scenes), B0 threshold, driver, n observed.
   Saved as h5_history/{district}_{Y}.npz.

Run:  python -u scripts/175_h5_dashboard.py models | history
"""
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s173", os.path.join(HERE, "173_amendment4.py"))
a4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a4)
a3, m, dc, H2, H5 = a4.a3, a4.m, a4.dc, a4.H2, a4.H5
MOD = os.path.join(dc.DYN, "models_h5_loyo")
HIST = os.path.join(dc.DYN, "h5_history")


def run_models():
    os.makedirs(MOD, exist_ok=True)
    T = a3.a2.load_amended()
    ff, bs = a4.history_features(T)
    ref = np.load(os.path.join(dc.DYN, "amend4_h5_loyo_preds.npz"))["p"]
    years = sorted(T["year"].unique())
    for Y in years:
        X = T[H2].assign(ff=ff[Y], b0s=bs[Y])
        tem = (T["year"] == Y).values
        mdl = a3.fit_xgb(H5, X[~tem].assign(y=T.loc[~tem, "y"].values))
        p = mdl.predict_proba(X.loc[tem, H5])[:, 1]
        print(f"H5 without {Y}: max |p - amend4| = {np.nanmax(np.abs(p - ref[tem])):.2e}", flush=True)
        mdl.save_model(os.path.join(MOD, f"h5_without_{Y}.json"))
    json.dump({"features": H5, "years": [int(y) for y in years],
               "rule": "Amendment 4 LOYO: test-year history from years != Y"}, open(os.path.join(MOD, "meta.json"), "w"), indent=1)


def run_history(chunk=200_000):
    os.makedirs(HIST, exist_ok=True)
    T = a3.a2.load_amended()
    for d in dc.DISTRICTS:
        sd = a3.scene_drivers(T, d)
        shape = np.load(os.path.join(dc.DYN, "labels", d, f"{sd.index[0]:%Y-%m-%d}.npz"))["label"].shape
        L = np.stack([np.load(os.path.join(dc.DYN, "labels", d, f"{t:%Y-%m-%d}.npz"))["label"].ravel() for t in sd.index])
        yr = sd["year"].values
        print(f"{d}: {L.shape[0]} scenes x {L.shape[1]} px stacked", flush=True)
        for Y in sorted(T["year"].unique()):
            tm = yr != Y
            w = (L[tm] == 1).sum(0); o = (L[tm] != 255).sum(0)
            ff = np.where(o > 0, w / np.maximum(o, 1), np.nan).astype(np.float32)
            thr = np.empty(L.shape[1]); drv = np.empty(L.shape[1], int); nobs = np.empty(L.shape[1], int)
            for a in range(0, L.shape[1], chunk):
                thr[a:a + chunk], drv[a:a + chunk], nobs[a:a + chunk] = a3.b0_fit(sd, L[:, a:a + chunk], tm)
            np.savez_compressed(os.path.join(HIST, f"{d}_{Y}.npz"), ff=ff.reshape(shape),
                                thr=thr.astype(np.float32).reshape(shape), drv=drv.astype(np.uint8).reshape(shape),
                                nobs=nobs.astype(np.int16).reshape(shape))
            print(f"{d} history without {Y} saved", flush=True)
        del L


if __name__ == "__main__":
    {"models": run_models, "history": run_history}[sys.argv[1]]()
