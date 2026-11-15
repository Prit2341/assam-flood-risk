"""Flood-EXTENT agreement of the GNN student with its HEC-RAS teacher on the held-out events, at the 20 m GFM pixel grid used for G1 scoring.
Pixel flooded = (cell water surface - unburned 30 m terrain at the pixel) > 0.1 m, cell = nearest cell centre (cached map from 05_score.py). Student and teacher are
scored against each other (teacher = reference), at hours 24, 48, 72, 96 of each test event, over the in-domain pixels with valid terrain.
  python -u scripts/hecras_sivasagar/28_extent_eval.py direct|direct_phys"""
import sys
tag = sys.argv[1]; phys = "0.00002" if "phys" in tag else "0"
sys.argv = ["x", "0", "999", phys, "20"] + (["cont"] if tag.endswith("_cont") else [])
src = open(__import__("pathlib").Path(__file__).with_name("26_train_direct.py")).read(); src = src[:src.index("train_ids = list")]
exec(compile(src, "26_train_direct", "exec"))
net.load_state_dict(torch.load(D / f"gnn_{tag}.pt", map_location=dev)); net.eval()
pm = np.load(ROOT / "digital_twin/sivasagar/data/pixmap_Sivasagar_2026_near_peak_a_2026-07-17_flood_26814.npz"); cid, z, dom = pm["cid"], pm["z"], pm["dom"]
okc = (cid < N) & dom & np.isfinite(z); cidc = np.clip(cid, 0, N - 1)
zc = torch.tensor(zmin, dtype=torch.float32)
rows = []
for i in ev[ev.split == "test"].id:
    p, y = rollout(i)
    for h in (24, 48, 72, 96):
        wp = (p[h] + zc).numpy(); wt = (y[h] + zc).numpy()
        mp = (wp[cidc] - z) > 0.1; mt = (wt[cidc] - z) > 0.1; v = okc
        hit = (mp & mt & v).sum(); fa = (mp & ~mt & v).sum(); mi = (~mp & mt & v).sum()
        rows.append(dict(id=i, hour=h, teacher_km2=mt[v].sum() * 4e-4, student_km2=mp[v].sum() * 4e-4, csi=hit / max(hit + fa + mi, 1), area_ratio=mp[v].sum() / max(mt[v].sum(), 1)))
df = pd.DataFrame(rows); df.to_csv(D / f"extent_{tag}_test.csv", index=False)
print(df.round(3).to_string()); print(df.groupby("hour")[["csi", "area_ratio"]].mean().round(3)); print("mean CSI", round(df.csi.mean(), 3), "median", round(df.csi.median(), 3))

# ---- post-processing threshold: predicted cell depths below tau are set to 0 (the teacher has exact zeros for dry cells); tau chosen on VAL, reported on TEST
def extent_scores(split, tau):
    out = []
    for i in ev[ev.split == split].id:
        p, y = rollout(i); p = p * (p > tau)
        for h in (48, 72, 96):
            mp = ((p[h] + zc).numpy()[cidc] - z) > 0.1; mt = ((y[h] + zc).numpy()[cidc] - z) > 0.1; v = okc
            hit = (mp & mt & v).sum(); fa = (mp & ~mt & v).sum(); mi = (~mp & mt & v).sum(); out.append((hit / max(hit + fa + mi, 1), mp[v].sum() / max(mt[v].sum(), 1)))
    return np.mean(out, 0)
print("\nthreshold tau (m)   VAL csi / area ratio   TEST csi / area ratio   (hours 48,72,96)")
for tau in (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
    a, b = extent_scores("val", tau), extent_scores("test", tau); print(f"{tau:5.2f}   {a[0]:.3f} / {a[1]:.2f}      {b[0]:.3f} / {b[1]:.2f}", flush=True)
