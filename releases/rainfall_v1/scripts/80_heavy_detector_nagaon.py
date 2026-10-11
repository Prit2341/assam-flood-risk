"""
Heavy-hour DETECTOR comparison (Nagaon). Motivation (Chroma manual-0070, from the
assembler results): heavy-hour interval coverage = gate recall x within-gate coverage,
within-gate coverage is already ~85%, so the lever is how many truly heavy hours the
gate flags, at what gated share. The current gate flags 1.1% of hours at h1 but 8.0% at
h6 (precision 24% -> 2.7%, recall 91% -> 77%).

Candidates (all fixed a priori, defaults, NO tuning against the test set):
  D0_gate    : current gate (LGBM+XGB ensemble trained on RAINY rows only), score = gate_p
  D0_pxg     : same models, score = P(rain) * P(heavy | rain)   (proper decomposition;
               the current gate never learns to reject dry hours)
  D1_lgbm    : LightGBM classifier trained on ALL rows for the heavy label at t+h
  D2_lgbm_cnn: D1 + out-of-fold CNN forecast as a feature (needs 70_stack output)
Metrics on test (t >= 2024): average precision (PR-AUC; ROC-AUC is uninformative at a
0.27% base rate), and heavy recall at fixed gated shares, including each lead's CURRENT
gated share so it is like-for-like with today's gate.

Run: python -u scripts/80_heavy_detector_nagaon.py
"""
import importlib
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m = importlib.import_module("20_base_rainfall")
ROOT = m.ROOT
DISTRICT = "Nagaon"
LEADS = [1, 2, 3, 4, 5, 6]
TEST_START = pd.Timestamp(m.TEST_SPLIT_DATE)
FEATS = list(m.FEATURE_COLS)
CNN_PATH = os.path.join(ROOT, "data/rainfall_nowcast/cnn6_oof_features_nagaon.parquet")
SHARES = [0.01, 0.02, 0.04, 0.08]


def recall_at_share(score, hv, share):
    k = max(1, int(round(share * len(score))))
    top = np.argsort(-score)[:k]
    return float(hv[top].sum() / max(1, hv.sum()))


def main():
    raw = m.load_hourly(DISTRICT)
    df = m.build_features(raw, DISTRICT)
    heavy_thr = m.compute_heavy_threshold(df[df["datetime"] < TEST_START])
    have_cnn = os.path.exists(CNN_PATH)
    if have_cnn:
        cnn = pd.read_parquet(CNN_PATH)
        df = df.merge(cnn.reset_index().rename(columns={"index": "datetime"}), on="datetime", how="left")
    print(f"heavy thr {heavy_thr:.2f}; CNN OOF features available: {have_cnn}", flush=True)

    rows = []
    for h in LEADS:
        dfh = df.copy()
        dfh["tp_mm"] = df["tp_mm"].shift(-(h - 1))
        if have_cnn:
            dfh["cnn_pred"] = df[f"cnn_h{h}"]
        dfh = dfh.dropna(subset=["tp_mm"] + (["cnn_pred"] if have_cnn else [])).reset_index(drop=True)
        dfh["y_occurrence"] = (dfh["tp_mm"] > m.RAIN_THR_MM).astype(int)
        dfh["y_heavy"] = (dfh["tp_mm"] > heavy_thr).astype(int)
        tr = dfh[dfh["datetime"] < TEST_START].reset_index(drop=True)
        te = dfh[dfh["datetime"] >= TEST_START].reset_index(drop=True)
        hv = te["y_heavy"].values
        m.FEATURE_COLS = FEATS
        occ, _, gate, _ = m.fit_models(tr, heavy_thr)
        gate_p = gate.predict_proba(te[FEATS])[:, 1]
        occ_p = occ.predict_proba(te[FEATS])[:, 1]
        cur_share = float((gate_p > m.GATE_PROBABILITY_CUTOFF).mean())

        scores = {"D0_gate": gate_p, "D0_pxg": occ_p * gate_p}
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                 scale_pos_weight=20.0, random_state=42, verbose=-1)
        clf.fit(tr[FEATS], tr["y_heavy"])
        scores["D1_lgbm"] = clf.predict_proba(te[FEATS])[:, 1]
        if have_cnn:
            F2 = FEATS + ["cnn_pred"]
            clf2 = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                      scale_pos_weight=20.0, random_state=42, verbose=-1)
            clf2.fit(tr[F2], tr["y_heavy"])
            scores["D2_lgbm_cnn"] = clf2.predict_proba(te[F2])[:, 1]

        print(f"\nh={h}: train {len(tr):,} ({int(tr['y_heavy'].sum())} heavy)  test {len(te):,} ({int(hv.sum())} heavy)  "
              f"current gated share {cur_share:.3f}", flush=True)
        for name, sc in scores.items():
            r = {"lead": h, "detector": name, "AP": average_precision_score(hv, sc),
                 "recall@current_share": recall_at_share(sc, hv, cur_share), "current_share": cur_share,
                 "n_heavy": int(hv.sum())}
            for s in SHARES:
                r[f"recall@{int(s*100)}%"] = recall_at_share(sc, hv, s)
            rows.append(r)
            print(f"  {name:12s} AP={r['AP']:.3f}  recall@cur({cur_share:.3f})={r['recall@current_share']:.3f}  "
                  + "  ".join(f"@{int(s*100)}%={r[f'recall@{int(s*100)}%']:.2f}" for s in SHARES), flush=True)
    res = pd.DataFrame(rows)
    out = os.path.join(ROOT, "data/rainfall_nowcast",
                       "heavy_detector_nagaon_summary%s.csv" % ("_cnn" if have_cnn else ""))
    res.to_csv(out, index=False)
    print("Saved", out)


if __name__ == "__main__":
    main()
