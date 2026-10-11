"""
Seychelles physics-based grid distillation as extra inputs to the rainfall base model.

Source method: Seychelles/approach_b_nwp/127_prototype_physics_distillation.py
(read in full 2026-10-09; Chroma manual-0102/0103). From u, v, q, z at 850 hPa on the
grid it builds full moisture flux convergence, divergence, vorticity and geopotential
gradients at the centre point plus 3x3 means. Seychelles' own evidence for it is thin
(one Dec-2023 test month, no control, R2 0.035 -> 0.126), so it is re-tested here under
this project's matched-control protocol, not assumed to help.

WHY IT CAN ADD INFORMATION: 20_base_rainfall.py load_pressure() keeps ONLY the centre
grid point, so the base model has never seen a spatial gradient.

Deliberate adaptations (each has a reason):
  * dx uses cos(latitude) per grid row. Seychelles used a constant 27.5 km (equator);
    Assam sits at 24.5-27 N where cos(lat) is ~0.89-0.91.
  * The 5x5 grid gives a 3x3 block of interior points with full central differences;
    "centre" is that block's middle and the "3x3 mean" averages the block.
  * Latitude is ascending (row 0 = south) via 50_pinn_assam.load_grid, so d/dy is
    northward-positive.
  * Features are lag-1 aligned like every other base feature: base row t uses the grid
    at hour t-1 (same convention as 70_stack_cnn_base_nagaon.py, t_row = f + 1h).

Four arms per (district, lead), identical split/threshold/gate/interval code
(20_base_rainfall.run_split): control, distilled, stacked (existing out-of-fold CNN
feature, cached parquet), stacked_distilled. All arms are fitted on the SAME rows.

SUCCESS CRITERIA, fixed before running (TASK.md 2026-10-09, Chroma manual-0103):
  PRIMARY: corr_all gain of distilled over control positive in >=18 of 24
  (district x lead) cells AND no cell with rmse_all >5% worse than control.
  SECONDARY (report only): corr_rainy, bias_rainy, gate AUC, heavy_cov.
  Same rule for stacked_distilled vs stacked.

Run:  python -u scripts/90_distill_features.py Nagaon
      python -u scripts/90_distill_features.py summary
"""
import importlib
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m = importlib.import_module("20_base_rainfall")
pn = importlib.import_module("50_pinn_assam")
ROOT = m.ROOT
LEADS = [1, 2, 3, 4, 5, 6]
TEST_START = pd.Timestamp(m.TEST_SPLIT_DATE)
BASE_FEATS = list(m.FEATURE_COLS)
DIST_FEATS = ["dist_mfc_c", "dist_div_c", "dist_vort_c", "dist_dzdx_c", "dist_dzdy_c",
              "dist_mfc_m", "dist_div_m", "dist_vort_m"]
M_PER_DEG = 111_320.0
STEP_DEG = 0.25
OUT_DIR = os.path.join(ROOT, "data", "rainfall_nowcast")


def distilled_features(district):
    """DataFrame indexed by base-row time t = f + 1h (lag-1 aligned) with DIST_FEATS."""
    dts, X = pn.load_grid(district)                       # X: (N, C, 5, 5), lat ascending
    ch = {c: i for i, c in enumerate(pn.CHANNELS)}
    u, v, q, z = (X[:, ch[c]].astype(np.float64) for c in ("u850", "v850", "q850", "z850"))
    lat0 = m.DISTRICTS[district][0]
    lats = lat0 + STEP_DEG * np.array([-2, -1, 0, 1, 2])
    dy = STEP_DEG * M_PER_DEG
    dx = (STEP_DEG * M_PER_DEG * np.cos(np.radians(lats)))[1:4].reshape(1, 3, 1)   # interior rows

    def ddx(a):
        return (a[:, 1:4, 2:5] - a[:, 1:4, 0:3]) / (2 * dx)

    def ddy(a):
        return (a[:, 2:5, 1:4] - a[:, 0:3, 1:4]) / (2 * dy)

    ui, vi, qi = u[:, 1:4, 1:4], v[:, 1:4, 1:4], q[:, 1:4, 1:4]
    div = ddx(u) + ddy(v)
    vort = ddx(v) - ddy(u)
    mfc = -(ui * ddx(q) + vi * ddy(q) + qi * div)          # full flux convergence, 1/s * kg/kg
    feats = pd.DataFrame({
        "dist_mfc_c": mfc[:, 1, 1], "dist_div_c": div[:, 1, 1], "dist_vort_c": vort[:, 1, 1],
        "dist_dzdx_c": ddx(z)[:, 1, 1], "dist_dzdy_c": ddy(z)[:, 1, 1],
        "dist_mfc_m": mfc.mean(axis=(1, 2)), "dist_div_m": div.mean(axis=(1, 2)),
        "dist_vort_m": vort.mean(axis=(1, 2)),
    })
    feats.index = pd.DatetimeIndex(dts) + pd.Timedelta(hours=1)
    feats.index.name = "datetime"
    return feats


def load_cnn_feats(district):
    path = os.path.join(OUT_DIR, f"cnn6_oof_features_{district.lower()}.parquet")
    c = pd.read_parquet(path)
    c.index.name = "datetime"
    return c.reset_index()


def interval_score(y, lo, hi, a=0.2):
    return (hi - lo) + (2 / a) * np.maximum(lo - y, 0) + (2 / a) * np.maximum(y - hi, 0)


def metrics(out, gate_p, heavy_thr, arm, h, district):
    y, pr = out["tp_mm"].values, out["q50_final"].values
    rn = y > m.RAIN_THR_MM
    hv = y > heavy_thr
    lo, hi = out["q10_ens"].values, out["q90_ens"].values
    return {"district": district, "lead": h, "arm": arm,
            "corr_all": np.corrcoef(y, pr)[0, 1],
            "rmse_all": float(np.sqrt(mean_squared_error(y, pr))),
            "corr_rainy": np.corrcoef(y[rn], pr[rn])[0, 1],
            "bias_rainy": float((pr[rn].mean() - y[rn].mean()) / y[rn].mean()),
            "auc_gate_heavy": roc_auc_score(hv.astype(int), gate_p),
            "heavy_cov": float(np.mean((y[hv] >= lo[hv]) & (y[hv] <= hi[hv]))),
            "heavy_width": float(np.mean(hi[hv] - lo[hv])),
            "heavy_IS": float(np.mean(interval_score(y[hv], lo[hv], hi[hv]))),
            "n_heavy": int(hv.sum()), "n_test": len(y)}


def run_district(district):
    t_all = time.time()
    raw = m.load_hourly(district)
    df = m.build_features(raw, district)
    heavy_thr = m.compute_heavy_threshold(df[df["datetime"] < TEST_START])
    print(f"{district}: heavy threshold (train<2024 p99.5) {heavy_thr:.2f} mm/h", flush=True)

    dist = distilled_features(district).reset_index()
    df = df.merge(dist, on="datetime", how="left").merge(load_cnn_feats(district), on="datetime", how="left")
    print("distilled feature summary:\n" + df[DIST_FEATS].describe().loc[["mean", "std", "min", "max"]].T.to_string(),
          flush=True)

    arms = (("control", BASE_FEATS),
            ("distilled", BASE_FEATS + DIST_FEATS),
            ("stacked", BASE_FEATS + ["cnn_pred"]),
            ("stacked_distilled", BASE_FEATS + DIST_FEATS + ["cnn_pred"]))
    rows = []
    for h in LEADS:
        dfh = df.copy()
        dfh["tp_mm"] = df["tp_mm"].shift(-(h - 1))
        dfh["cnn_pred"] = df[f"cnn_h{h}"]
        dfh = dfh.dropna(subset=["tp_mm", "cnn_pred"] + DIST_FEATS).reset_index(drop=True)
        dfh["y_occurrence"] = (dfh["tp_mm"] > m.RAIN_THR_MM).astype(int)
        tr = dfh[dfh["datetime"] < TEST_START].reset_index(drop=True)
        te = dfh[dfh["datetime"] >= TEST_START].reset_index(drop=True)
        print(f"\nh={h}: train {len(tr):,}  test {len(te):,}", flush=True)
        for arm, feats in arms:
            m.FEATURE_COLS = feats
            t0 = time.time()
            out, models = m.run_split(tr, te, heavy_thr)
            gp = models[2].predict_proba(te[feats])[:, 1]
            r = metrics(out.reset_index(drop=True), gp, heavy_thr, arm, h, district)
            rows.append(r)
            print(f"  {arm:18s} corr_all={r['corr_all']:.3f} rmse={r['rmse_all']:.3f} corr_rainy={r['corr_rainy']:.3f} "
                  f"bias={r['bias_rainy']:+.2f} gateAUC={r['auc_gate_heavy']:.3f} heavy_cov={r['heavy_cov']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        m.FEATURE_COLS = BASE_FEATS

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, f"distill_{district.lower()}_summary.csv"), index=False)
    print(f"\n{district} done in {(time.time()-t_all)/60:.1f} min", flush=True)


def verdict(res, a, b, label):
    """Apply the pre-registered rule to arm a vs arm b over all available cells."""
    wide = res.pivot_table(index=["district", "lead"], columns="arm", values=["corr_all", "rmse_all"])
    gain = wide["corr_all"][a] - wide["corr_all"][b]
    rmse_pct = (wide["rmse_all"][a] / wide["rmse_all"][b] - 1) * 100
    n = len(gain)
    pos = int((gain > 0).sum())
    worst = float(rmse_pct.max())
    need = int(np.ceil(0.75 * n))
    ok = pos >= need and worst <= 5.0
    print(f"\n{label}: {a} vs {b}: corr_all gain >0 in {pos}/{n} cells (need >={need}); "
          f"mean gain {gain.mean():+.4f}; worst rmse change {worst:+.1f}% (limit +5%)  ->  "
          f"{'PASS' if ok else 'FAIL'} (primary criterion)")
    return gain, rmse_pct


def summary():
    files = [os.path.join(OUT_DIR, f"distill_{d.lower()}_summary.csv") for d in m.DISTRICTS]
    have = [f for f in files if os.path.exists(f)]
    res = pd.concat([pd.read_csv(f) for f in have], ignore_index=True)
    print(f"districts present: {sorted(res['district'].unique())}  ({len(res)//4} cells)")
    if len(res["district"].unique()) < 4:
        print("NOTE: fewer than 4 districts, so the 18/24 threshold is scaled to 75% of available cells; "
              "the pre-registered verdict needs all 4.")
    verdict(res, "distilled", "control", "DISTILLATION")
    verdict(res, "stacked_distilled", "stacked", "DISTILLATION ON TOP OF PRODUCTION STACK")
    verdict(res, "stacked", "control", "(reference) CNN STACK")
    pd.set_option("display.width", 250)
    for col in ("corr_rainy", "bias_rainy", "auc_gate_heavy", "heavy_cov"):
        w = res.pivot_table(index=["district", "lead"], columns="arm", values=col)
        print(f"\nSECONDARY {col}: mean over cells\n{w.mean().round(4).to_string()}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "Nagaon"
    if arg == "summary":
        summary()
    elif arg in m.DISTRICTS:
        run_district(arg)
    else:
        print(f"Unknown '{arg}'. Choices: {list(m.DISTRICTS)} or summary")
