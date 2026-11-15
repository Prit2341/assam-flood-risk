"""Fine depth-cutoff calibration for the student (pre-registered in Chroma, 2026-11-06). Chooses tau on VAL (closest mean area ratio to 1), reports TEST.
  python -u scripts/hecras_sivasagar/49_tau_fine.py direct_cont [direct_cont_s1b ...]"""
import sys
tags = sys.argv[1:]
sys.argv = ["x", "0", "999", "0", "20"]
from pathlib import Path
src = open(Path(__file__).with_name("26_train_direct.py")).read(); src = src[:src.index("train_ids = list")]
exec(compile(src, "26_train_direct", "exec"))
pm = np.load(ROOT / "digital_twin/sivasagar/data/pixmap_Sivasagar_2026_near_peak_a_2026-07-17_flood_26814.npz"); cid, z, dom = pm["cid"], pm["z"], pm["dom"]
okc = (cid < N) & dom & np.isfinite(z); cidc = np.clip(cid, 0, N - 1); zc = torch.tensor(zmin, dtype=torch.float32)
TAUS = [0, 0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20, 0.22, 0.25]
def stats(split, tau, R):
    out = []
    for i in ev[ev.split == split].id:
        p, y = R[i]; p = p * (p > tau)
        for h in (24, 48, 72, 96):
            mp = ((p[h] + zc).numpy()[cidc] - z) > 0.1; mt = ((y[h] + zc).numpy()[cidc] - z) > 0.1; v = okc
            hit = (mp & mt & v).sum(); T = (mt & v).sum(); Sx = (mp & v).sum()
            out.append((hit / max(T, 1), hit / max(Sx, 1), hit / max(T + Sx - hit, 1), Sx / max(T, 1)))
    return np.mean(out, 0)       # POD, precision, CSI, area ratio
rows = []
for tag in tags:
    net.load_state_dict(torch.load(D / f"gnn_{tag}.pt", map_location=dev)); net.eval()
    R = {i: rollout(i) for i in ev[ev.split.isin(["val", "test"])].id}
    val = {t: stats("val", t, R) for t in TAUS}; test = {t: stats("test", t, R) for t in TAUS}
    best = min(TAUS, key=lambda t: abs(np.log(max(val[t][3], 1e-6))))
    print(f"\n{tag}: chosen tau (val ratio closest to 1) = {best}")
    print("  tau   VAL POD/prec/CSI/ratio            TEST POD/prec/CSI/ratio")
    for t in TAUS:
        mark = " <== chosen" if t == best else ""
        print(f"  {t:4.2f}  " + " ".join(f"{x:.3f}" for x in val[t]) + "     " + " ".join(f"{x:.3f}" for x in test[t]) + mark, flush=True)
    rows.append(dict(tag=tag, tau=best, POD=test[best][0], precision=test[best][1], CSI=test[best][2], ratio=test[best][3]))
df = pd.DataFrame(rows).round(3); print("\nTEST at chosen tau:\n", df.to_string(index=False)); df.to_csv(D / "tau_fine_test.csv", index=False)
