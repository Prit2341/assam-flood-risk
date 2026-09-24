"""
Feature-level stacking: the pressure-grid CNN forecast as an extra input to the
base XGBoost/LightGBM rainfall model (Nagaon).

WHY: measured crossover (Chroma manual-0068) -- base wins at +1..2h (rainfall
history), CNN wins at +3..6h (pressure grid). Stacking lets one model use both.

Leakage-safe design (keeps the base model's FULL 2000-2023 training window):
  * Blocked K-fold over 2000-2023: 8 blocks of 3 years. For each block a CNN is
    trained on the OTHER blocks and predicts the held-out block, so every base
    TRAINING row carries a CNN value that CNN never trained on (out-of-fold).
  * Test rows (2024+): CNN trained on ALL rows < 2024 (no CNN ever saw 2024+).
  * One MULTI-LEAD CNN (6 outputs, shared trunk, masked MSE) so a fold costs one
    training, not six. Its skill is compared with the single-lead CNNs (manual-0068).
  * CONTROL arm: the identical base model WITHOUT the CNN feature, retrained in
    this same script, so any difference is the CNN feature and nothing else.
  * Same gate/threshold/CSG+GPD interval construction as 20_base_rainfall.py.
The CNN has physics weight 0 (the moisture penalty was inert, manual-0068): it is a
plain CNN and is called one.

Run: python -u scripts/70_stack_cnn_base_nagaon.py [Sivasagar|Marigaon|Nagaon|Karimganj] [seed]
"""
import importlib
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m = importlib.import_module("20_base_rainfall")
pn = importlib.import_module("50_pinn_assam")
ROOT = m.ROOT
DEV = pn.DEV
DISTRICT = sys.argv[1] if len(sys.argv) > 1 else "Nagaon"
LEADS = [1, 2, 3, 4, 5, 6]
TEST_START = pd.Timestamp(m.TEST_SPLIT_DATE)
BLOCKS = [(2000 + 3 * i, 2003 + 3 * i) for i in range(8)]     # [start_year, end_year)
EPOCHS, BATCH = 10, 128
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0          # CNN seed only; base models keep random_state=42
SFX = "" if SEED == 0 else f"_s{SEED}"                       # seed 0 keeps the original file names
BASE_FEATS = list(m.FEATURE_COLS)


class CNN6(nn.Module):
    def __init__(self, in_ch, n_out=6):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_out), nn.Softplus())

    def forward(self, x):
        return self.fc(self.cnn(x))


def train_cnn6(Xtr, Ytr, tag):
    """Xtr (N,C,5,5) normalised on-device; Ytr (N,6) with NaN for missing targets."""
    torch.manual_seed(SEED)
    net = CNN6(Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = Xtr.shape[0]
    mask = ~torch.isnan(Ytr)
    Yz = torch.nan_to_num(Ytr, nan=0.0)
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(n, device=DEV)
        tot, nb = 0.0, 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            pred = net(Xtr[idx])
            mk = mask[idx]
            loss = (((pred - Yz[idx]) ** 2) * mk).sum() / mk.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep in (0, EPOCHS - 1):
            print(f"    {tag} ep{ep+1}/{EPOCHS} loss={tot/nb:.4f}", flush=True)
    net.eval()
    return net


def predict6(net, X):
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 4096):
            out.append(net(X[i:i + 4096]).cpu().numpy())
    return np.concatenate(out)


def build_cnn_features(dts, Xall, tp):
    """Returns DataFrame indexed by base-row time t=f+1 with cnn_h1..cnn_h6 (OOF for <2024)."""
    Y = np.column_stack([tp.reindex(dts + pd.Timedelta(hours=h)).values for h in LEADS]).astype(np.float32)
    t_row = dts + pd.Timedelta(hours=1)
    yr = t_row.year.values
    is_pre = t_row < TEST_START
    preds = np.full((len(dts), 6), np.nan, dtype=np.float32)

    def fit_predict(train_mask, pred_mask, tag):
        mu = Xall[train_mask].mean(axis=(0, 2, 3), keepdims=True)
        sd = Xall[train_mask].std(axis=(0, 2, 3), keepdims=True) + 1e-12
        Xn = torch.tensor((Xall - mu) / sd, dtype=torch.float32)
        net = train_cnn6(Xn[train_mask].to(DEV), torch.tensor(Y[train_mask]).to(DEV), tag)
        preds[pred_mask] = predict6(net, Xn[pred_mask].to(DEV))

    for (y0, y1) in BLOCKS:
        held = is_pre & (yr >= y0) & (yr < y1)
        train = is_pre & ~held
        t0 = time.time()
        print(f"  fold {y0}-{y1 - 1}: train {train.sum():,} rows, predict {held.sum():,}", flush=True)
        fit_predict(train, held, f"fold{y0}")
        print(f"    ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    print(f"  final CNN: train all <2024 ({is_pre.sum():,}), predict test", flush=True)
    fit_predict(is_pre, ~is_pre, "final")
    print(f"    ({time.time()-t0:.0f}s)", flush=True)

    # OOF / test skill of the multi-lead CNN vs the single-lead CNN numbers in manual-0068
    for name, sel in (("OOF 2000-2023", is_pre), ("test 2024+", ~is_pre)):
        cs = []
        for j, h in enumerate(LEADS):
            ok = sel & ~np.isnan(Y[:, j]) & ~np.isnan(preds[:, j])
            cs.append(np.corrcoef(preds[ok, j], Y[ok, j])[0, 1])
        print(f"  CNN6 corr_all {name}: " + "  ".join(f"h{h}={c:.3f}" for h, c in zip(LEADS, cs)), flush=True)
    return pd.DataFrame(preds, index=t_row, columns=[f"cnn_h{h}" for h in LEADS])


def interval_score(y, lo, hi, a=0.2):
    return (hi - lo) + (2 / a) * np.maximum(lo - y, 0) + (2 / a) * np.maximum(y - hi, 0)


def metrics(out, gate_p, heavy_thr, arm, h):
    y, pr = out["tp_mm"].values, out["q50_final"].values
    rn = y > m.RAIN_THR_MM
    hv = y > heavy_thr
    lo, hi = out["q10_ens"].values, out["q90_ens"].values
    return {"lead": h, "arm": arm,
            "corr_all": np.corrcoef(y, pr)[0, 1],
            "rmse_all": float(np.sqrt(mean_squared_error(y, pr))),
            "corr_rainy": np.corrcoef(y[rn], pr[rn])[0, 1],
            "bias_rainy": float((pr[rn].mean() - y[rn].mean()) / y[rn].mean()),
            "auc_gate_heavy": roc_auc_score(hv.astype(int), gate_p),
            "heavy_cov": float(np.mean((y[hv] >= lo[hv]) & (y[hv] <= hi[hv]))),
            "heavy_width": float(np.mean(hi[hv] - lo[hv])),
            "heavy_IS": float(np.mean(interval_score(y[hv], lo[hv], hi[hv]))),
            "n_heavy": int(hv.sum()), "n_test": len(y)}


def main():
    raw = m.load_hourly(DISTRICT)
    df = m.build_features(raw, DISTRICT)
    heavy_thr = m.compute_heavy_threshold(df[df["datetime"] < TEST_START])
    print(f"heavy threshold (train<2024 p99.5): {heavy_thr:.2f} mm/h", flush=True)

    dts, Xall = pn.load_grid(DISTRICT)
    surf = m.load_surface(DISTRICT).set_index("datetime")["tp_mm"]
    tp = surf.reindex(pd.date_range(surf.index.min(), surf.index.max(), freq="h"))
    cnn_df = build_cnn_features(dts, Xall, tp)
    cnn_df.to_parquet(os.path.join(ROOT, f"data/rainfall_nowcast/cnn6_oof_features_{DISTRICT.lower()}{SFX}.parquet"))
    df = df.merge(cnn_df.reset_index().rename(columns={"index": "datetime"}), on="datetime", how="left")

    rows = []
    for h in LEADS:
        dfh = df.copy()
        dfh["tp_mm"] = df["tp_mm"].shift(-(h - 1))
        dfh["cnn_pred"] = df[f"cnn_h{h}"]
        dfh = dfh.dropna(subset=["tp_mm", "cnn_pred"]).reset_index(drop=True)
        dfh["y_occurrence"] = (dfh["tp_mm"] > m.RAIN_THR_MM).astype(int)
        tr = dfh[dfh["datetime"] < TEST_START].reset_index(drop=True)
        te = dfh[dfh["datetime"] >= TEST_START].reset_index(drop=True)
        print(f"\nh={h}: train {len(tr):,}  test {len(te):,}", flush=True)
        for arm, feats in (("control", BASE_FEATS), ("stacked", BASE_FEATS + ["cnn_pred"])):
            m.FEATURE_COLS = feats
            out, models = m.run_split(tr, te, heavy_thr)
            gp = models[2].predict_proba(te[feats])[:, 1]
            r = metrics(out.reset_index(drop=True), gp, heavy_thr, arm, h)
            rows.append(r)
            print(f"  {arm:8s} corr_all={r['corr_all']:.3f} rmse={r['rmse_all']:.3f} corr_rainy={r['corr_rainy']:.3f} "
                  f"bias={r['bias_rainy']:+.2f} gateAUC={r['auc_gate_heavy']:.3f} heavy_cov={r['heavy_cov']:.3f} "
                  f"w={r['heavy_width']:.1f} IS={r['heavy_IS']:.1f}", flush=True)
        m.FEATURE_COLS = BASE_FEATS

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(ROOT, f"data/rainfall_nowcast/stack_cnn_base_{DISTRICT.lower()}_summary{SFX}.csv"), index=False)
    pd.set_option("display.width", 250)
    print("\n", res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
