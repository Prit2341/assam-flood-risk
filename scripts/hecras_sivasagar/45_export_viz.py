"""(v2 2026-11-06: map layer = share of each 600 m cell under water, from the 20 m pixel terrain used for G1 scoring; cell-minimum depth > 0.1 m is NOT shown because it marks ~80% of the basin wet.)
Export teacher (HEC-RAS) vs student (GNN direct_cont) depth frames for held-out 2025 events -> docs/site/ts_data.json (viewer: docs/site/teacher_student.html).
Same rollout code as 28_extent_eval.py (exec of the top of 26_train_direct.py). Depth is stored as uint8 with sqrt scaling: q = round(255*sqrt(min(d,4)/4)), every 2nd hour.
Also stores per-hour domain water volume, wet area (depth > 0.1 m) and rain for both models, cell coordinates and a normalised terrain layer.
  python -u scripts/hecras_sivasagar/45_export_viz.py"""
import sys, json, base64
sys.argv = ["x", "0", "999", "0", "20", "cont"]
from pathlib import Path
src = open(Path(__file__).with_name("26_train_direct.py")).read(); src = src[:src.index("train_ids = list")]
exec(compile(src, "26_train_direct", "exec"))
import os
VIZ_TAG = os.environ.get("VIZ_TAG", "direct_cont")   # student checkpoint tag; for f1 also set GNN_FEATS=1
net.load_state_dict(torch.load(D / f"gnn_{VIZ_TAG}.pt", map_location=dev)); net.eval()
IDS = ["e037", "e036", "e041", "e039"]
area = S["area"][:N].astype("float64")
def q(a): return np.round(255 * np.sqrt(np.clip(a, 0, 4) / 4)).astype("uint8")
def b64(a): return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
x, y_ = xy[:, 0], xy[:, 1]
zt = (zmin - np.percentile(zmin, 1)) / (np.percentile(zmin, 99) - np.percentile(zmin, 1)); zt = np.clip(zt, 0, 1)
out = dict(n=int(N), x=b64(np.round((x - x.min()) / 10).astype("uint16")), y=b64(np.round((y_.max() - y_) / 10).astype("uint16")),
           w=int((x.max() - x.min()) / 10) + 1, h=int((y_.max() - y_.min()) / 10) + 1, cell_m=600, terrain=b64((zt * 255).astype("uint8")),
           zmin_range=[float(np.percentile(zmin, 1)), float(np.percentile(zmin, 99))], events={}, student_tag=os.environ.get('VIZ_TAG', 'direct_cont'))
evd = ev.set_index("id")
pm = np.load(ROOT / "digital_twin/sivasagar/data/pixmap_Sivasagar_2026_near_peak_a_2026-07-17_flood_26814.npz"); cid, z, dom = pm["cid"], pm["z"], pm["dom"]
okp = (cid < N) & dom & np.isfinite(z); cid_ = np.clip(cid, 0, N - 1)[okp]; z_ = z[okp].astype("float32")
npix = np.bincount(cid_, minlength=N).astype("float64"); evalmask = (npix > 0).astype("uint8"); npix[npix == 0] = 1
out['evalmask'] = b64(evalmask); out['eval_km2'] = float(okp.sum() * 4e-4)
def wetmask(d): return ((d + zmin)[cid_].astype("float32") - z_) > 0.1          # depth is wse - zmin, so wse = d + zmin
def share(m): return np.bincount(cid_, weights=m, minlength=N) / npix
for i in IDS:
    p, yt = rollout(i); p = p.numpy().astype("float64"); yt = yt.numpy().astype("float64")
    d = np.load(D / "runs" / f"{i}.npz"); rain = d["rain_mm"].astype("float64")
    hrs = list(range(0, p.shape[0], 2)); wt_km, ws_km, csi_h, ov_h, fa_h, mi_h = [], [], [], [], [], []; Ft, Fs = {}, {}
    for h in range(p.shape[0]):
        mt, mp = wetmask(yt[h]), wetmask(p[h]); hit, fa_, mi_ = (mt & mp).sum(), (~mt & mp).sum(), (mt & ~mp).sum()
        wt_km.append(float(mt.sum() * 4e-4)); ws_km.append(float(mp.sum() * 4e-4)); csi_h.append(float(hit / max(hit + fa_ + mi_, 1)))
        fa_h.append(float(fa_ * 4e-4)); mi_h.append(float(mi_ * 4e-4))
        if h in hrs: Ft[h] = np.round(255 * share(mt)).astype("uint8"); Fs[h] = np.round(255 * share(mp)).astype("uint8")
    out["events"][i] = dict(start=str(evd.loc[i, "start"])[:10], total_mm=float(evd.loc[i, "total"]), scale=float(evd.loc[i, "scale"]), hours=hrs,
        rain=[float(v) for v in rain], teacher=[b64(Ft[h]) for h in hrs], student=[b64(Fs[h]) for h in hrs],
        vol_t=[float((yt[h] * area).sum() / 1e6) for h in range(p.shape[0])], vol_s=[float((p[h] * area).sum() / 1e6) for h in range(p.shape[0])],
        wet_t=wt_km, wet_s=ws_km, csi=csi_h, fa=fa_h, miss=mi_h,
        rmse=float(np.sqrt(((p - yt) ** 2).mean())))
    print(i, "frames", len(hrs), "rmse", round(out["events"][i]["rmse"], 3), flush=True)
fn = ROOT / "docs/site" / os.environ.get("VIZ_OUT", "ts_data.json"); fn.write_text(json.dumps(out)); print("wrote", fn, round(fn.stat().st_size / 1e6, 1), "MB")
