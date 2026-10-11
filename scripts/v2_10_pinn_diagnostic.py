"""
v2 step 1 -- the Seychelles "PINN" AS WRITTEN, ported to Assam.

Source: Seychelles/3D_approach/10_pinn_26yr.py (read in full 2026-10-09). Copied as written,
including its known weaknesses (documented in Chroma manual-0102/0111):
  * DIAGNOSTIC alignment: the grid at hour t predicts rain at the SAME hour t.
  * chronological 80/20 split; the saved predictions cover ALL rows, including the rows the
    network trained on.
  * normalisation mean/std over ALL data, std + 1e-6.
  * loss = MSE + lambda * mean(relu(pred - 5.0) * relu(-MFC_centre)), lambda ramped
    min(1, epoch/5) * 10; MFC = -(u dq/dx + v dq/dy) per GRID INDEX at the centre cell.
  * 2 conv layers (32, 64) + BatchNorm + global average pool + MLP + Softplus; Adam 1e-3;
    10 epochs; batch 128.

Adaptations that are not choices: Assam's grid has 15 channels (adds pv500); incomplete hours
(not all 25 grid points) are dropped; torch seed 0 for reproducibility; latitude ascending
(same sort as Seychelles: datetime, latitude, longitude).

ADDED INSTRUMENTATION (does not change the model): the raw physics penalty and its share of
the loss are logged every epoch, because Seychelles saved no log and never showed whether the
term was active. Output: data/rainfall_nowcast/v2_pinn_diag_<district>.parquet and a log CSV.

Run: python -u scripts/v2_10_pinn_diagnostic.py Nagaon
"""
import importlib
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m = importlib.import_module("20_base_rainfall")
ROOT = m.ROOT
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join(ROOT, "data", "rainfall_nowcast")
EPOCHS, BATCH = 10, 128


class PhysicsInformedNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus())

    def forward(self, x):
        return self.fc(self.cnn(x))


def run(district):
    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    pp = m.PATH_SPELLING[district]["pressure"]
    p = pd.read_parquet(os.path.join(ROOT, "data", "icechunk_pressure", f"{pp}_pressure_5x5.parquet"))
    p["datetime"] = pd.to_datetime(p["datetime"])
    p = p.sort_values(["datetime", "latitude", "longitude"])
    cnt = p.groupby("datetime").size()
    p = p[p["datetime"].isin(cnt.index[cnt == 25])]
    surf = m.load_surface(district)[["datetime", "tp_mm"]].drop_duplicates("datetime").sort_values("datetime")
    common = np.intersect1d(surf["datetime"].values, p["datetime"].unique())
    surf = surf[surf["datetime"].isin(common)]
    p = p[p["datetime"].isin(common)]

    feature_cols = [c for c in p.columns if c not in ["datetime", "latitude", "longitude"]]
    n_samples, n_features = len(common), len(feature_cols)
    X_raw = p[feature_cols].values.reshape(n_samples, 5, 5, n_features)
    X_raw = np.transpose(X_raw, (0, 3, 1, 2)).astype(np.float32)
    q_i, u_i, v_i = (feature_cols.index(c) for c in ("q850", "u850", "v850"))
    Y_raw = surf["tp_mm"].values.astype(np.float32)

    X_norm = (X_raw - X_raw.mean(axis=(0, 2, 3), keepdims=True)) / (X_raw.std(axis=(0, 2, 3), keepdims=True) + 1e-6)
    X_t = torch.tensor(X_norm, dtype=torch.float32)
    Y_t = torch.tensor(Y_raw, dtype=torch.float32).unsqueeze(1)

    def mfc_centre(xr):
        q, u, v = xr[:, q_i], xr[:, u_i], xr[:, v_i]
        dq_dx, dq_dy = torch.zeros_like(q), torch.zeros_like(q)
        dq_dx[:, :, 1:-1] = (q[:, :, 2:] - q[:, :, :-2]) / 2.0
        dq_dy[:, 1:-1, :] = (q[:, 2:, :] - q[:, :-2, :]) / 2.0
        return (-(u * dq_dx + v * dq_dy))[:, 2, 2]

    split = int(n_samples * 0.8)
    loader = DataLoader(TensorDataset(X_t[:split], torch.tensor(X_raw[:split]), Y_t[:split]),
                        batch_size=BATCH, shuffle=True)
    model = PhysicsInformedNN(n_features).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    log = []
    print(f"{district}: {n_samples:,} hours, {n_features} channels, train {split:,}, "
          f"train ends {pd.Timestamp(common[split-1])}", flush=True)
    model.train()
    for ep in range(EPOCHS):
        tl = td = tph = 0.0
        npos = nb = 0
        for xn, xr, y in loader:
            xn, xr, y = xn.to(DEV), xr.to(DEV), y.to(DEV)
            opt.zero_grad()
            pred = model(xn)
            data = nn.MSELoss()(pred, y)
            mfc = mfc_centre(xr).unsqueeze(1)
            viol = torch.relu(pred - 5.0) * torch.relu(-mfc)
            phys = viol.mean()
            lam = min(1.0, ep / 5.0) * 10.0
            loss = data + lam * phys
            loss.backward()
            opt.step()
            tl += loss.item(); td += data.item(); tph += phys.item(); nb += 1
            npos += int((viol > 0).sum().item())
        row = {"district": district, "epoch": ep + 1, "loss": tl / nb, "data": td / nb,
               "phys_raw": tph / nb, "lambda": lam, "phys_share_of_loss": (lam * tph / nb) / (tl / nb),
               "frac_samples_with_nonzero_violation": npos / (nb * BATCH)}
        log.append(row)
        print(f"  ep{ep+1:>2}/{EPOCHS} loss={row['loss']:.4f} data={row['data']:.4f} phys_raw={row['phys_raw']:.3e} "
              f"phys_share={row['phys_share_of_loss']:.2e} nonzero_viol={row['frac_samples_with_nonzero_violation']:.2e}",
              flush=True)

    model.eval()
    with torch.no_grad():
        preds = np.concatenate([model(X_t[i:i + 8192].to(DEV)).cpu().numpy().ravel()
                                for i in range(0, n_samples, 8192)])
    pd.DataFrame({"datetime": common, "pinn_tp_mm": preds}).to_parquet(
        os.path.join(OUT_DIR, f"v2_pinn_diag_{district.lower()}.parquet"), index=False)
    pd.DataFrame(log).to_csv(os.path.join(OUT_DIR, f"v2_pinn_diag_{district.lower()}_log.csv"), index=False)
    te = slice(split, None)
    c = np.corrcoef(Y_raw[te], preds[te])[0, 1]
    print(f"{district}: held-out (last 20pct) same-hour corr = {c:.3f}  ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "Nagaon"
    run(d)
