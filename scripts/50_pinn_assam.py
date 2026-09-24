"""
CNN leg on a 5x5 pressure-level ERA5 grid, with an attempted moisture physics
penalty, ported from Seychelles' 3D_approach/10_pinn_26yr.py, as a FORECAST
model. NAMING NOTE (Chroma manual-0067/0068, confirmed across all leads/seeds
run so far): the physics penalty never activates (raw phys term <=1e-6, "pinn"
vs "cnn_nophys" identical to <=0.002 corr at every lead) -- this is a plain
CNN, not a working PINN. File/variable names below keep the original "pinn"
label only for continuity with Seychelles' naming and this project's own
Chroma/TASK history; do not describe it as a PINN in any thesis writeup or new
KB entry -- call it "the pressure-grid CNN" and state the physics term was
tried and found inert.

Deliberate changes from the Seychelles script (each has a reason):
  * FORECAST alignment. Seychelles predicts rain at hour t from the state at
    the same hour t (diagnostic). Here features are the grid at hour f and
    the target is ERA5 tp at f+h, matching 20_base_rainfall.py (its row t uses
    lag-1 features, target at t+(h-1) = (t-1)+h, so f = t-1).
  * Split matches the base model: train rows with t < 2024-01-01, test t >=.
    Seychelles used a chronological 80/20 split.
  * Channel normalisation from TRAIN rows only, epsilon 1e-12 (Seychelles
    used all data and std+1e-6, which distorts pv500 whose std is ~4e-7).
  * Latitude is sorted ASCENDING (row 0 = south) so v>0 (northward) agrees
    with the finite-difference d/dy in the physics term. The parquet is
    stored latitude-DESCENDING; not sorting would flip the sign of dq/dy.
  * Physics-penalty rain threshold = the district's adaptive heavy threshold
    (train p99.5 of rainy hours, as in the base model) instead of Seychelles'
    fixed 5 mm.
  * ABLATION: the same CNN is also trained with the physics weight = 0, and
    both loss terms are logged, because the physics term (u*dq per grid
    index, ~1e-3..1e-2) may be numerically negligible next to the MSE.
    NOTE the physics term is -(u dq/dx + v dq/dy) = moisture ADVECTION sign,
    per grid index, not full moisture-flux convergence -(div(qV)); it is
    kept as in the source, not "corrected", so results stay comparable.

Usage (project root):
    python -u scripts/50_pinn_assam.py --district Nagaon --leads 1 3 6 --seeds 0
"""
import argparse
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
ROOT = m.ROOT
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHANNELS = ["u200", "v200", "z200", "w500", "z500", "t500", "pv500", "t700", "r700",
            "u850", "v850", "t850", "q850", "r850", "z850"]
TEST_START = pd.Timestamp(m.TEST_SPLIT_DATE)
EPOCHS = 10
BATCH = 128


class PhysicsInformedNN(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus())

    def forward(self, x):
        return self.fc(self.cnn(x))


def mfc_proxy(qvu):
    """qvu: (B,3,5,5) raw [q850,u850,v850]; grid row0=south, col0=west."""
    q, u, v = qvu[:, 0], qvu[:, 1], qvu[:, 2]
    dq_dx = torch.zeros_like(q)
    dq_dy = torch.zeros_like(q)
    dq_dx[:, :, 1:-1] = (q[:, :, 2:] - q[:, :, :-2]) / 2.0
    dq_dy[:, 1:-1, :] = (q[:, 2:, :] - q[:, :-2, :]) / 2.0
    return (-(u * dq_dx + v * dq_dy))[:, 2, 2]


def load_grid(district):
    pp = m.PATH_SPELLING[district]["pressure"]
    p = pd.read_parquet(os.path.join(ROOT, "data", "icechunk_pressure", f"{pp}_pressure_5x5.parquet"))
    p["datetime"] = pd.to_datetime(p["datetime"])
    p = p.sort_values(["datetime", "latitude", "longitude"])       # lat ascending
    cnt = p.groupby("datetime").size()
    keep = cnt.index[cnt == 25]
    p = p[p["datetime"].isin(keep)]
    dts = p["datetime"].drop_duplicates().values
    X = p[CHANNELS].values.astype(np.float32).reshape(len(dts), 5, 5, len(CHANNELS))
    return pd.DatetimeIndex(dts), np.transpose(X, (0, 3, 1, 2))     # (N,C,5,5)


def train_one(Xtr, Ytr, QVUtr, heavy_thr, lam_phys, seed, log_prefix):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = PhysicsInformedNN(Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = Xtr.shape[0]
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(n, device=DEV)
        tl = td = tp_ = 0.0
        nb = 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            pred = net(Xtr[idx])
            data = nn.functional.mse_loss(pred, Ytr[idx])
            viol = torch.relu(pred - heavy_thr) * torch.relu(-mfc_proxy(QVUtr[idx])).unsqueeze(1)
            phys = viol.mean()
            w = min(1.0, ep / 5.0) * lam_phys
            loss = data + w * phys
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item(); td += data.item(); tp_ += phys.item(); nb += 1
        print(f"  {log_prefix} ep{ep+1:>2}/{EPOCHS} loss={tl/nb:.4f} data={td/nb:.4f} "
              f"phys(raw)={tp_/nb:.6f}  phys*lam(={w:.1f})={w*tp_/nb:.6f}", flush=True)
    net.eval()
    return net


def predict(net, X):
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 4096):
            out.append(net(X[i:i + 4096]).cpu().numpy().ravel())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--district", default="Nagaon")
    ap.add_argument("--leads", type=int, nargs="+", default=[1, 3, 6])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--tag", default="", help="suffix for output files so runs do not overwrite each other")
    a = ap.parse_args()
    d = a.district
    print(f"Device: {DEV}  district={d}  leads={a.leads}  seeds={a.seeds}", flush=True)

    dts, Xall = load_grid(d)
    surf = m.load_surface(d).set_index("datetime")["tp_mm"]
    tp = surf.reindex(pd.date_range(surf.index.min(), surf.index.max(), freq="h"))
    print(f"Grid: {Xall.shape} {dts.min()} -> {dts.max()}", flush=True)

    # heavy threshold from base-model training rows (same definition as base model)
    train_rain = tp[(tp.index < TEST_START) & (tp > m.RAIN_THR_MM)]
    heavy_thr = float(train_rain.quantile(m.HEAVY_PERCENTILE / 100.0))
    print(f"Heavy threshold (train p{m.HEAVY_PERCENTILE}): {heavy_thr:.2f} mm/h", flush=True)

    base = pd.read_csv(os.path.join(ROOT, "data/rainfall_nowcast", f"forecast_horizons_{d.lower()}.csv"),
                       parse_dates=["datetime"])
    q_idx = [CHANNELS.index(c) for c in ("q850", "u850", "v850")]

    rows, preds_out = [], []
    for h in a.leads:
        tgt = tp.reindex(dts + pd.Timedelta(hours=h)).values
        t_row = dts + pd.Timedelta(hours=1)                    # base-model row time t = f+1
        ok = ~np.isnan(tgt)
        is_tr = ok & (t_row < TEST_START)
        is_te = ok & (t_row >= TEST_START)
        mu = Xall[is_tr].mean(axis=(0, 2, 3), keepdims=True)
        sd = Xall[is_tr].std(axis=(0, 2, 3), keepdims=True) + 1e-12
        Xn = torch.tensor((Xall - mu) / sd, dtype=torch.float32)
        Xtr = Xn[is_tr].to(DEV)
        Ytr = torch.tensor(tgt[is_tr], dtype=torch.float32).unsqueeze(1).to(DEV)
        QVUtr = torch.tensor(Xall[is_tr][:, q_idx], dtype=torch.float32).to(DEV)
        Xte = Xn[is_te].to(DEV)
        y_te = tgt[is_te]
        t_te = t_row[is_te]
        print(f"\n=== lead h={h}: train {is_tr.sum():,} rows, test {is_te.sum():,} rows ===", flush=True)

        b = base[base["lead_hour"] == h].set_index("datetime")
        b_q50 = b["q50_final"].reindex(t_te).values
        b_act = b["tp_mm"].reindex(t_te).values
        # sanity: my target must equal the base model's target on the same rows
        both = ~np.isnan(b_act)
        assert np.allclose(b_act[both], y_te[both], atol=1e-4), "target misalignment vs base model"
        print(f"  target alignment vs base model OK on {both.sum():,} rows", flush=True)

        for seed in a.seeds:
            for tag, lam in (("pinn", 10.0), ("cnn_nophys", 0.0)):
                t0 = time.time()
                net = train_one(Xtr, Ytr, QVUtr, heavy_thr, lam, seed, f"h{h} s{seed} {tag}")
                pr = predict(net, Xte)
                rainy = y_te > m.RAIN_THR_MM
                hv = (y_te > heavy_thr).astype(int)
                res_b = (b_act - b_q50)
                row = {"district": d, "lead": h, "seed": seed, "model": tag,
                       "corr_all": np.corrcoef(y_te, pr)[0, 1],
                       "rmse_all": float(np.sqrt(mean_squared_error(y_te, pr))),
                       "corr_rainy": np.corrcoef(y_te[rainy], pr[rainy])[0, 1],
                       "bias_rainy": float((pr[rainy].mean() - y_te[rainy].mean()) / y_te[rainy].mean()),
                       "auc_heavy": roc_auc_score(hv, pr),
                       "corr_with_base_residual": float(np.corrcoef(pr[both], res_b[both])[0, 1]),
                       "n_test": len(y_te), "n_heavy": int(hv.sum()),
                       "train_seconds": time.time() - t0}
                rows.append(row)
                preds_out.append(pd.DataFrame({"datetime": t_te, "lead_hour": h, "seed": seed,
                                               "model": tag, "tp_mm": y_te, "pred": pr}))
                print(f"  -> h{h} s{seed} {tag}: corr_all={row['corr_all']:.3f} rmse={row['rmse_all']:.3f} "
                      f"corr_rainy={row['corr_rainy']:.3f} bias={row['bias_rainy']:+.2f} "
                      f"auc_heavy={row['auc_heavy']:.3f} corr_w_base_resid={row['corr_with_base_residual']:+.3f} "
                      f"({row['train_seconds']:.0f}s)", flush=True)
        # base-model reference on the same rows
        rb = b_q50[both]; ya = y_te[both]; rn = ya > m.RAIN_THR_MM
        print(f"  base (2000-2023 model) same rows: corr_all={np.corrcoef(ya, rb)[0,1]:.3f} "
              f"rmse={np.sqrt(mean_squared_error(ya, rb)):.3f} "
              f"corr_rainy={np.corrcoef(ya[rn], rb[rn])[0,1]:.3f}", flush=True)
        rows.append({"district": d, "lead": h, "seed": -1, "model": "base",
                     "corr_all": np.corrcoef(ya, rb)[0, 1],
                     "rmse_all": float(np.sqrt(mean_squared_error(ya, rb))),
                     "corr_rainy": np.corrcoef(ya[rn], rb[rn])[0, 1],
                     "bias_rainy": float((rb[rn].mean() - ya[rn].mean()) / ya[rn].mean()),
                     "auc_heavy": roc_auc_score((ya > heavy_thr).astype(int), rb),
                     "n_test": int(both.sum()), "n_heavy": int((ya > heavy_thr).sum())})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(ROOT, "data/rainfall_nowcast", f"pinn_{d.lower()}_summary{a.tag}.csv"), index=False)
    pd.concat(preds_out).to_csv(os.path.join(ROOT, "data/rainfall_nowcast", f"pinn_{d.lower()}_predictions{a.tag}.csv"),
                                index=False)
    pd.set_option("display.width", 250)
    print("\n", res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
