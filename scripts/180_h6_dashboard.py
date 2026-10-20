"""
H6 for the dashboard (Amendment 6 working model). Refits the Amendment 6 LOYO H6 models (same features, history rule
and XGB seed as scripts/179) and saves them as models_h6_loyo/h6_without_{Y}.json.
Sanity check: test-row predictions must equal amend6_loyo_preds.npz['H6'].
The per-pixel H5 history comes from h5_history/ (scripts/175); previous-scene features are computed on request.

Run:  python -u scripts/180_h6_dashboard.py
"""
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s179", os.path.join(HERE, "179_amendment6.py"))
a6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a6)
a3, dc, H2, H6 = a6.a3, a6.dc, a6.H2, a6.H6
MOD = os.path.join(dc.DYN, "models_h6_loyo")


def main():
    os.makedirs(MOD, exist_ok=True)
    T = a6.load(a3.a2.load_amended())
    z = np.load(a6.HIST)
    ref = np.load(a6.LOYO)["H6"]
    years = sorted(T["year"].unique())
    for Y in years:
        tem = (T["year"] == Y).values
        X = T[H2 + a6.PREV].assign(ff=z[f"ff_{Y}"], b0s=z[f"bs_{Y}"], y=T["y"].values)
        mdl = a3.fit_xgb(H6, X[~tem])
        p = mdl.predict_proba(X.loc[tem, H6])[:, 1]
        print(f"H6 without {Y}: max |p - amend6| = {np.nanmax(np.abs(p - ref[tem])):.2e}", flush=True)
        mdl.save_model(os.path.join(MOD, f"h6_without_{Y}.json"))
    json.dump({"features": H6, "years": [int(y) for y in years], "max_gap_days": a6.MAXGAP,
               "rule": "Amendment 6 LOYO; previous-scene features past-only"}, open(os.path.join(MOD, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
