"""Training event list for the GNN student (decision 2026-11-04): 96 h windows of domain-mean ERA5-Land hourly rain from the monsoon seasons pulled by
20_pull_rain_long.sh plus 2026 Jul-Aug, picked to cover the range of window totals (8 quantile bins, 6 non-overlapping windows each), plus the 6 wettest windows scaled x1.5 and x2.
Split by year: 2025 = test, 2024 = validation, others = train (events never share a window). Out: digital_twin/sivasagar/surrogate/events.csv
Precedent for event-wise datasets of this size: HydroGraphNet / dual_flood_gnn use 56 HEC-RAS events (see spec s11)."""
import glob
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; D = ROOT / "digital_twin/sivasagar"
r = pd.concat([pd.read_csv(f, parse_dates=["timestamp"]) for f in sorted(glob.glob(str(D / "data/era5land_domain_hourly_*.csv")))]).drop_duplicates("timestamp").set_index("timestamp").tp_mm.sort_index()
r = r[~r.index.duplicated()]; H = 96
full = r.reindex(pd.date_range(r.index.min(), r.index.max(), freq="h"))
cand = []
for t0 in full.index[::24]:
    seg = full[t0:t0 + pd.Timedelta(hours=H)]
    if len(seg) == H + 1 and seg.notna().all():
        cand.append(dict(start=t0, total=float(seg.values[1:].sum()), peak6=float(seg.rolling(6).sum().max())))
c = pd.DataFrame(cand); print("candidate windows", len(c), "years", sorted(c.start.dt.year.unique()))
c["bin"] = pd.qcut(c.total.rank(method="first"), 8, labels=False)
rng = np.random.default_rng(42); chosen = []
def free(t): return all(abs((t - u).total_seconds()) >= H * 3600 for u in chosen)
for b in range(8):
    pool = c[c.bin == b].sample(frac=1, random_state=int(rng.integers(1e6)))
    n = 0
    for _, row in pool.iterrows():
        if n == 6: break
        if free(row.start): chosen.append(row.start); n += 1
ev = c[c.start.isin(chosen)].sort_values("start").reset_index(drop=True); ev["scale"] = 1.0; ev["kind"] = "real"
top = ev.sort_values("total", ascending=False).head(6)
sc_rows = pd.concat([top.assign(scale=s, kind="scaled") for s in (1.5, 2.0)])
ev = pd.concat([ev, sc_rows]).reset_index(drop=True)
ev["split"] = np.where(ev.start.dt.year == 2025, "test", np.where(ev.start.dt.year == 2024, "val", "train"))
ev.insert(0, "id", [f"e{i:03d}" for i in range(len(ev))]); ev["hours"] = H
ev.to_csv(D / "surrogate/events.csv", index=False)
print(ev.groupby(["kind", "split"]).size()); print(ev[["id", "start", "total", "peak6", "scale", "split"]].to_string())
