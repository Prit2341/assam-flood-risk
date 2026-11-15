"""DIRECT (non-autoregressive) GNN student v3: the teacher is a cold-start, uniform-rain model, so depth at hour t is a function of the static mesh and the rain history to t.
The net maps (static node/edge features + scalar rain-history summary) -> depth of every cell at that hour, with no hour-to-hour feedback, hence no rollout drift.
Optional PHYS = weight of a domain water-volume loss (HEC-RAS cell volume tables): predicted total volume must match the teacher and must not exceed the cumulative rain volume.
  python -u scripts/hecras_sivasagar/26_train_direct.py epochs max_events PHYS steps_per_event
(original docstring of the autoregressive script follows) GNN student v1 (PHYS=0) / v2 (PHYS>0, adds domain mass-balance loss from the HEC-RAS cell volume-elevation tables): learns the HEC-RAS (frozen teacher) one-hour update of cell depth on the 600 m mesh graph, then rolls out 96 h from a dry start.
Depth = water surface - cell minimum elevation (>= 0). Graph = faces joining two real cells (both directions). Message passing in plain torch (no torch_geometric).
Node inputs: depth, hourly rain (mm), cumulative rain, time fraction, static (zmin, area, Manning n). Edge inputs: dx, dy, distance, dz of cell minima, face length, face-min-elevation offsets.
Train on split=train events (one-step targets, input noise), pick epoch on split=val rollouts, report split=test. Baselines: persistence (depth stays 0) and teacher.
  python -u scripts/hecras_sivasagar/24_train_gnn.py [epochs] [max_events]"""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
ROOT = Path(__file__).resolve().parents[2]; D = ROOT / "digital_twin/sivasagar/surrogate"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 20; maxev = int(sys.argv[2]) if len(sys.argv) > 2 else 999
PHYS = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0   # weight of the domain mass-balance loss (0 = purely data-driven v1)
CONT = len(sys.argv) > 5 and sys.argv[5] == "cont"   # continue from the saved checkpoint with extra weight on shallow wet cells
import os  # optional overrides (2026-11-06): GNN_RUNS (surrogate/ subfolder), GNN_STATIC (static graph file), GNN_SUFFIX (appended to TAG)
RUNSDIR = os.environ.get("GNN_RUNS", "runs"); STATIC = os.environ.get("GNN_STATIC", "static_graph.npz")
import torch as _t  # seed + loss-weight overrides (2026-11-06): GNN_SEED, GNN_W="wet,shallow" (default "4,6")
if os.environ.get("GNN_SEED"):
    np.random.seed(int(os.environ["GNN_SEED"])); _t.manual_seed(int(os.environ["GNN_SEED"]))
WW = [float(v) for v in os.environ.get("GNN_W", "4,6").split(",")]
TAG = "direct" + ("_phys" if PHYS > 0 else "") + ("_cont" if CONT else "") + os.environ.get("GNN_SUFFIX", "")
S = np.load(D / STATIC); N = int(S["n_real"]); xy = S["xy"][:N]; zmin = S["zmin"][:N].astype("float64")
fc = S["face_cells"]; fl = S["face_nx_ny_len"]; fz = S["face_zmin"]
inner = (fc[:, 0] >= 0) & (fc[:, 1] >= 0) & (fc[:, 0] < N) & (fc[:, 1] < N)
a, b = fc[inner, 0], fc[inner, 1]; src = np.r_[a, b]; dst = np.r_[b, a]
dxy = (xy[dst] - xy[src]) / 1000.0; dist = np.hypot(*dxy.T)[:, None]
flen = np.r_[fl[inner, 2], fl[inner, 2]][:, None] / 600.0; fzm = np.r_[fz[inner], fz[inner]]
ef = np.c_[dxy, dist, (zmin[dst] - zmin[src])[:, None] / 10.0, flen, ((fzm - zmin[src]) / 10.0)[:, None], ((fzm - zmin[dst]) / 10.0)[:, None]].astype("float32")
sf = np.c_[(zmin - zmin.mean()) / zmin.std(), (S["area"][:N] - S["area"][:N].mean()) / S["area"][:N].std(), (S["mann"][:N] - 0.04) * 10].astype("float32")
src_t, dst_t, ef_t, sf_t = [torch.tensor(x).to(dev) for x in (src, dst, ef, sf)]
zmin_t = torch.tensor(zmin, dtype=torch.float32, device=dev)
import h5py
with h5py.File(ROOT / "digital_twin/sivasagar/results" / os.environ.get("GNN_TABLES_HDF", "c600_d4_r30_0705_432h_cal.p01.hdf")) as f:
    _A = "Geometry/2D Flow Areas/Sivasagar_Domain"; vinfo = f[_A + "/Cells Volume Elevation Info"][:N]; vval = f[_A + "/Cells Volume Elevation Values"][:]
K = int(vinfo[:, 1].max()); Ep = np.full((N, K), 1e9, "float64"); Vp = np.zeros((N, K), "float64")
for i in range(N):
    o, c = vinfo[i]
    if c > 0: Ep[i, :c] = vval[o:o + c, 0]; Vp[i, :c] = vval[o:o + c, 1]; Vp[i, c:] = vval[o + c - 1, 1]
Ep_t = torch.tensor(Ep, dtype=torch.float64, device=dev); Vp_t = torch.tensor(Vp, dtype=torch.float64, device=dev); zmin64 = torch.tensor(zmin, dtype=torch.float64, device=dev)
AREA_TOT = float(S["area"][:N].sum())
if os.environ.get("GNN_FEATS"):   # 2026-11-06: per-cell wet-area fraction at depth 0.1/0.3/1/3 m from the HEC-RAS volume-elevation table (hypsometric slope)
    def _vol_at(w):
        idx = torch.searchsorted(Ep_t, w[:, None].contiguous()).clamp(1, K - 1)[:, 0]; ar = torch.arange(N)
        e0 = Ep_t[ar, idx - 1]; e1 = Ep_t[ar, idx]; v0 = Vp_t[ar, idx - 1]; v1 = Vp_t[ar, idx]
        return v0 + ((w - e0) / (e1 - e0).clamp_min(1e-9)).clamp(0, 1) * (v1 - v0)
    _area = torch.tensor(S["area"][:N], dtype=torch.float64, device=dev); _z = zmin64.clone()
    wf = torch.stack([(_vol_at(_z + d + 0.05) - _vol_at(_z + d - 0.05)) / 0.1 / _area for d in (0.1, 0.3, 1.0, 3.0)], 1).clamp(0, 1.5).float().cpu().numpy()
    print("wet-area-fraction features", wf.shape, "mean", wf.mean(0).round(3), flush=True)
    sf_t = torch.tensor(np.c_[sf, wf]).to(dev)
def total_volume(depth):   # m3, piecewise-linear in the HEC-RAS cell volume-elevation tables (differentiable in depth)
    w = zmin64 + depth.double(); idx = torch.searchsorted(Ep_t, w[:, None].contiguous()).clamp(1, K - 1)[:, 0]
    e0 = Ep_t[torch.arange(N), idx - 1]; e1 = Ep_t[torch.arange(N), idx]; v0 = Vp_t[torch.arange(N), idx - 1]; v1 = Vp_t[torch.arange(N), idx]
    fr = ((w - e0) / (e1 - e0).clamp_min(1e-9)).clamp(0, 1); return (v0 + fr * (v1 - v0)).sum()
ev = pd.read_csv(D / "events.csv"); have = {p.stem for p in (D / RUNSDIR).glob("*.npz")}
ev = ev[ev.id.isin(have)].head(maxev); print(ev.split.value_counts().to_dict(), "events with outputs", len(ev), "device", dev, flush=True)

def load(i):
    d = np.load(D / RUNSDIR / f"{i}.npz"); depth = np.maximum(d["wse"].astype("float64") - zmin, 0).astype("float32"); rain = d["rain_mm"].astype("float32")
    return torch.tensor(depth), torch.tensor(rain)
DATA = {r.id: load(r.id) for r in ev.itertuples()}
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 20   # gradient steps per event per epoch
VT = {}   # teacher domain water volume (m3) per hour, precomputed once (no gradient needed)
if PHYS > 0:
    with torch.no_grad():
        for i, (dd, _) in DATA.items(): VT[i] = torch.stack([total_volume(dd[k].to(dev)) for k in range(dd.shape[0])]).float()
DS, RS = 2.0, 5.0   # depth scale (m), rain scale (mm/h)


def feats(rain, cum, t, T):
    c = float(cum[t]); r = lambda k: float(rain[max(0, t - k + 1):t + 1].sum()) if t > 0 else 0.0
    return torch.tensor([c / 100.0, r(1) / 10.0, r(3) / 20.0, r(6) / 40.0, r(12) / 60.0, r(24) / 100.0, t / (T - 1), float(rain[:t + 1].max()) / 10.0], dtype=torch.float32, device=dev)

class Net(nn.Module):
    def __init__(s, h=64, K=4):
        super().__init__(); mlp = lambda i, o: nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, o))
        s.enc = mlp(sf_t.shape[1] + 8, h); s.eenc = mlp(ef.shape[1], h); s.K = K
        s.msg = nn.ModuleList([mlp(3 * h, h) for _ in range(K)]); s.upd = nn.ModuleList([mlp(2 * h, h) for _ in range(K)]); s.dec = mlp(h, 1)
    def forward(s, f):
        x = torch.cat([sf_t, f[None, :].expand(N, -1)], 1); h = s.enc(x); e = s.eenc(ef_t)
        for k in range(s.K):
            m = s.msg[k](torch.cat([h[src_t], h[dst_t], e], 1)); agg = torch.zeros_like(h).index_add_(0, dst_t, m) / 6.0
            h = h + s.upd[k](torch.cat([h, agg], 1))
        return torch.relu(s.dec(h)[:, 0]) * DS   # depth in m

net = Net().to(dev)
if os.environ.get('GNN_DECBIAS'):   # 2026-11-06: positive final bias so the output ReLU is active at init (two seeded runs collapsed to all-zero output)
    net.dec[-1].bias.data.fill_(float(os.environ['GNN_DECBIAS']))
opt = torch.optim.Adam(net.parameters(), 5e-4 if CONT else 1e-3)
if CONT:
    ck = D / f"gnn_{TAG}.pt"; ck = ck if ck.exists() else D / f"gnn_{TAG.replace('_cont','')}.pt"   # resume own checkpoint if present, else the base model
    net.load_state_dict(torch.load(ck, map_location=dev)); print('resumed from', ck.name, flush=True)
CUM = {i: np.cumsum(DATA[i][1].numpy()) - DATA[i][1].numpy()[0] for i in DATA}   # rain[0] is the unused first value
@torch.no_grad()
def rollout(i):
    depth, rain = DATA[i]; T = depth.shape[0]; out = [net(feats(rain.numpy(), CUM[i], t, T)).cpu() for t in range(T)]; return torch.stack(out), depth

def metrics(split):
    rows = []
    for i in ev[ev.split == split].id:
        t_ = time.time(); p, y = rollout(i); sec = time.time() - t_; w = y > 0.1; q = p > 0.1
        rows.append(dict(id=i, sec=sec, rmse=float(((p - y) ** 2).mean().sqrt()), csi=float((w & q).sum() / max((w | q).sum(), 1)), vol_err=float((p.sum() - y.sum()) / max(y.sum(), 1))))
    df = pd.DataFrame(rows); (D / f"per_event_{TAG}_{split}.csv").write_text(df.to_csv(index=False)) if rows else None; return df.drop(columns='id').mean().to_dict() if rows else {}
train_ids = list(ev[ev.split == "train"].id); t0 = time.time(); best = (1e9, None)
for epoch in range(epochs):
    net.train(); tot = 0; n = 0
    for i in np.random.permutation(train_ids):
        depth, rain = DATA[i]; depth = depth.to(dev); T = depth.shape[0]; rn = rain.numpy()
        for t in np.random.permutation(T)[:STEPS]:
            pred = net(feats(rn, CUM[i], t, T)); w = 1.0 + WW[0] * (depth[t] > 0.1).float() + (WW[1] * ((depth[t] > 0) & (depth[t] <= 0.5)).float() if CONT else 0.0)
            loss = (w * ((pred - depth[t]) / DS) ** 2).mean()
            if PHYS > 0:
                Vp = total_volume(pred); Vt = VT[i][t]; rvol = float(CUM[i][t]) / 1000.0 * AREA_TOT * 0.0 + float(rain[1:t + 1].sum()) / 1000.0 * AREA_TOT
                loss = loss + PHYS * (((Vp - Vt) / 1e6) ** 2 + torch.relu((Vp - rvol) / 1e6) ** 2).float()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); tot += float(loss); n += 1
    net.eval(); mv = metrics("val")
    print(f"epoch {epoch+1} loss {tot/max(n,1):.5f} val {mv} t={time.time()-t0:.0f}s", flush=True)
    if mv and mv["rmse"] < best[0]: best = (mv["rmse"], {k: v.clone() for k, v in net.state_dict().items()}); torch.save(best[1], D / f"gnn_{TAG}.pt")
if best[1] is not None: net.load_state_dict(best[1])
res = dict(val=metrics("val"), test=metrics("test"), baseline_zero_rmse={s_: float(np.mean([float(DATA[i][0].pow(2).mean().sqrt()) for i in ev[ev.split == s_].id])) for s_ in ("val", "test")})
print(json.dumps(res, indent=1)); (D / f"gnn_{TAG}_results.json").write_text(json.dumps(res, indent=1))
