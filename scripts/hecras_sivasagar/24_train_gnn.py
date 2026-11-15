"""GNN student v1 (PHYS=0) / v2 (PHYS>0, adds domain mass-balance loss from the HEC-RAS cell volume-elevation tables): learns the HEC-RAS (frozen teacher) one-hour update of cell depth on the 600 m mesh graph, then rolls out 96 h from a dry start.
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
PF = int(sys.argv[4]) if len(sys.argv) > 4 else 0   # max pushforward steps
TAG = ("v2" if PHYS > 0 else "v1") + (f"pf{PF}" if PF else "")
S = np.load(D / "static_graph.npz"); N = int(S["n_real"]); xy = S["xy"][:N]; zmin = S["zmin"][:N].astype("float64")
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
with h5py.File(ROOT / "digital_twin/sivasagar/results/c600_d4_r30_0705_432h_cal.p01.hdf") as f:
    _A = "Geometry/2D Flow Areas/Sivasagar_Domain"; vinfo = f[_A + "/Cells Volume Elevation Info"][:N]; vval = f[_A + "/Cells Volume Elevation Values"][:]
K = int(vinfo[:, 1].max()); Ep = np.full((N, K), 1e9, "float64"); Vp = np.zeros((N, K), "float64")
for i in range(N):
    o, c = vinfo[i]
    if c > 0: Ep[i, :c] = vval[o:o + c, 0]; Vp[i, :c] = vval[o:o + c, 1]; Vp[i, c:] = vval[o + c - 1, 1]
Ep_t = torch.tensor(Ep, dtype=torch.float64, device=dev); Vp_t = torch.tensor(Vp, dtype=torch.float64, device=dev); zmin64 = torch.tensor(zmin, dtype=torch.float64, device=dev)
AREA_TOT = float(S["area"][:N].sum())
def total_volume(depth):   # m3, piecewise-linear in the HEC-RAS cell volume-elevation tables (differentiable in depth)
    w = zmin64 + depth.double(); idx = torch.searchsorted(Ep_t, w[:, None].contiguous()).clamp(1, K - 1)[:, 0]
    e0 = Ep_t[torch.arange(N), idx - 1]; e1 = Ep_t[torch.arange(N), idx]; v0 = Vp_t[torch.arange(N), idx - 1]; v1 = Vp_t[torch.arange(N), idx]
    fr = ((w - e0) / (e1 - e0).clamp_min(1e-9)).clamp(0, 1); return (v0 + fr * (v1 - v0)).sum()
ev = pd.read_csv(D / "events.csv"); have = {p.stem for p in (D / "runs").glob("*.npz")}
ev = ev[ev.id.isin(have)].head(maxev); print(ev.split.value_counts().to_dict(), "events with outputs", len(ev), "device", dev, flush=True)

def load(i):
    d = np.load(D / "runs" / f"{i}.npz"); depth = np.maximum(d["wse"].astype("float64") - zmin, 0).astype("float32"); rain = d["rain_mm"].astype("float32")
    return torch.tensor(depth), torch.tensor(rain)
DATA = {r.id: load(r.id) for r in ev.itertuples()}
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 8   # gradient steps per event per epoch
VT = {}   # teacher domain water volume (m3) per hour, precomputed once (no gradient needed)
if PHYS > 0:
    with torch.no_grad():
        for i, (dd, _) in DATA.items(): VT[i] = torch.stack([total_volume(dd[k].to(dev)) for k in range(dd.shape[0])]).float()
DS, RS = 2.0, 5.0   # depth scale (m), rain scale (mm/h)

class Net(nn.Module):
    def __init__(s, h=64, K=4):
        super().__init__(); mlp = lambda i, o: nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, o))
        s.enc = mlp(3 + 4, h); s.eenc = mlp(ef.shape[1], h); s.K = K
        s.msg = nn.ModuleList([mlp(3 * h, h) for _ in range(K)]); s.upd = nn.ModuleList([mlp(2 * h, h) for _ in range(K)]); s.dec = mlp(h, 1)
    def forward(s, depth, rain, cum, tf):
        x = torch.cat([sf_t, (depth / DS)[:, None], torch.full_like(depth, rain / RS)[:, None], torch.full_like(depth, cum / 100.0)[:, None], torch.full_like(depth, tf)[:, None]], 1)
        h = s.enc(x); e = s.eenc(ef_t)
        for k in range(s.K):
            m = s.msg[k](torch.cat([h[src_t], h[dst_t], e], 1)); agg = torch.zeros_like(h).index_add_(0, dst_t, m) / 6.0
            h = h + s.upd[k](torch.cat([h, agg], 1))
        return s.dec(h)[:, 0] * 0.25   # predicted change of depth in units of DS

net = Net().to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
def step(depth, rain, cum, tf): return torch.relu(depth + net(depth, rain, cum, tf) * DS)
@torch.no_grad()
def rollout(i):
    depth, rain = DATA[i]; T = depth.shape[0]; cur = torch.zeros(N, device=dev); out = [cur.cpu()]; cum = 0.0
    for t in range(T - 1):
        r = float(rain[t + 1]); cum += r; cur = step(cur, r, cum, t / (T - 1)); out.append(cur.cpu())
    return torch.stack(out), depth
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
        depth, rain = DATA[i]; depth = depth.to(dev); T = depth.shape[0]; cum = np.cumsum(rain.numpy())
        for t in np.random.permutation(T - 1)[:STEPS]:
            # pushforward trick (Brandstetter et al. 2022): start from the teacher state at t, roll the net forward n_pf hours WITHOUT gradient, then train one step from that
            # self-generated state toward the teacher's next state, so the net learns to correct its own drift (plain one-step training gave unstable 96 h rollouts)
            n_pf = int(np.random.randint(0, min(PF, T - 2 - t) + 1)) if PF > 0 else 0; x = depth[t]; cum_t = cum[t]
            with torch.no_grad():
                for j in range(n_pf):
                    cum_t = cum[t + j + 1]; x = step(x, float(rain[t + j + 1]), float(cum_t), (t + j) / (T - 1))
            tt = t + n_pf; x_in = torch.relu(x + torch.randn_like(x) * 0.01 * (x > 0))
            pred = net(x_in, float(rain[tt + 1]), float(cum[tt + 1]), tt / (T - 1)); tgt = (depth[tt + 1] - x_in) / DS
            loss = ((pred - tgt) ** 2).mean()
            if PHYS > 0:
                nxt = torch.relu(x_in + pred * DS); dV_pred = (total_volume(nxt) - total_volume(x_in.detach())).float(); dV_true = VT[i][tt + 1] - VT[i][tt]
                rvol = float(rain[tt + 1]) / 1000.0 * AREA_TOT
                loss = loss + PHYS * (((dV_pred - dV_true) / 1e6) ** 2 + torch.relu((dV_pred - rvol) / 1e6) ** 2)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); tot += float(loss); n += 1
    net.eval(); mv = metrics("val") if (ev.split == "val").any() else {}
    print(f"epoch {epoch+1} loss {tot/max(n,1):.5f} val {mv} t={time.time()-t0:.0f}s", flush=True)
    if mv and mv["rmse"] < best[0]: best = (mv["rmse"], {k: v.clone() for k, v in net.state_dict().items()}); torch.save(best[1], D / f"gnn_{TAG}.pt")
if best[1] is not None: net.load_state_dict(best[1])
elif (D / f"gnn_{TAG}.pt").exists(): net.load_state_dict(torch.load(D / f"gnn_{TAG}.pt", map_location=dev)); print("loaded checkpoint", TAG)
res = dict(val=metrics("val"), test=metrics("test"), baseline_zero_rmse={s: float(np.mean([float(DATA[i][0].pow(2).mean().sqrt()) for i in ev[ev.split == s].id] or [np.nan])) for s in ("val", "test")})
print(json.dumps(res, indent=1)); (D / f"gnn_{TAG}_results.json").write_text(json.dumps(res, indent=1))
