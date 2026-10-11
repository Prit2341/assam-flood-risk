"""
4-member adaptive assembler for the heavy-rain interval (Nagaon), ported from
Seychelles' approach_b_nwp/111_adaptive_ensemble_26yr.py with the problems found
when reading it fixed (Chroma manual-0069). The assembler reshapes ONLY the
q10/q90 interval on gated (heavy-predicted) rows; the point forecast is not
changed here.

Members (all give q10/q90 for a gated row):
  csg_b  : CSG-Gamma conditioned on the base model's point forecast (q50_mm)
  lgpd   : Local GPD tail of the district series
  rgpd   : Regional GPD (index-flood RFA, Hosking & Wallis) over the 5x5 ERA5 grid
           + the district series. SIGN FIXED: Seychelles' fit returns Hosking's
           kappa (heavy tail <=> kappa<0) but passes it to scipy genpareto(c=),
           where heavy tail <=> c>0 -> here c = -kappa. Verified on synthetic GPD
           data (manual-0069).
  csg_c  : CSG-Gamma conditioned on the CNN point forecast (data-driven; replaces
           Seychelles' unsupported pinn*0.5 / pinn*1.5 band)

Out-of-sample discipline (the original fit weights on in-sample rows):
  base model + CNN trained  : 2000-01 .. 2015-12
  weights / CSG params fit  : 2016-01 .. 2023-12  (both models never saw it)
                              CSG params leave-one-YEAR-out inside this period
  scored                    : 2024-01 ..           (same rows as earlier results)
  weights: SLSQP, sum=1, >=0, minimise pinball(0.1)+pinball(0.9) on GATED rows
  (the rows the interval is applied to), NOT on rows already known to be heavy.
  Tails: weight-fitting stage uses data <2016; final scoring refits on <2024.

Arms (same gate, same rows; only the composition of the heavy interval differs):
  meanof2   : 0.5*(csg_b + lgpd)            = the current production construction
  3member   : csg_b, lgpd, rgpd  (SLSQP)    = Seychelles without the PINN
  4member   : + csg_c            (SLSQP)
  equal4    : 4 members, equal weights
  sey4      : literal Seychelles recipe: PINN member = cnn*0.5 / cnn*1.5, weights by
              MSE of q90 vs actual on heavy fit rows, applied to q10 and q90
  3m_seysign: 3member but with Seychelles' UNFIXED regional sign (shows what the bug does)

Metrics on TEST rows that are truly heavy (actual > thr): coverage, mean width and
the 80% interval score (Gneiting & Raftery 2007) so width cannot buy coverage.
Also on all gated test rows (where the interval is actually applied).

Run: python -u scripts/60_assembler_nagaon.py [--leads 1 2 3 4 5 6]
"""
import argparse
import importlib
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m = importlib.import_module("20_base_rainfall")
pn = importlib.import_module("50_pinn_assam")
ROOT = m.ROOT

DISTRICT = "Nagaon"
TRAIN_END = pd.Timestamp("2016-01-01")
FIT_END = pd.Timestamp("2024-01-01")
ALPHA = 0.2
MIN_SITE_EXCEED, MIN_EXC, DISC_CUT = 5, 15, 3.0


# ----------------------------------------------------------------- RFA (Hosking & Wallis)
def sample_lmoments(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n < 3:
        return None
    i = np.arange(1, n + 1)
    b0 = x.mean()
    b1 = np.sum(((i - 1) / (n - 1)) * x) / n
    b2 = np.sum(((i - 1) * (i - 2) / ((n - 1) * (n - 2))) * x) / n
    L1, L2, L3 = b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0
    if L2 <= 0:
        return None
    return L1, L2, L3, L2 / L1, L3 / L2


def hw_discordancy(mom):
    N = len(mom)
    if N < 4:
        return np.zeros(N)
    U = np.array(mom)
    dev = U - U.mean(axis=0)
    try:
        Sinv = np.linalg.inv(np.cov(U, rowvar=False))
    except np.linalg.LinAlgError:
        return np.zeros(N)
    return np.array([(N / 2.0) * (d @ Sinv @ d) for d in dev])


def regional_gpd(site_vals, own_vals, thr):
    """Returns dict(sey=(kappa_as_passed_by_seychelles, scale), fixed=(c_scipy, scale)) or None."""
    sites = {}
    for k, v in site_vals.items():
        ex = v[v > thr] - thr
        if len(ex) >= MIN_SITE_EXCEED:
            sites[k] = ex
    own = own_vals[own_vals > thr] - thr
    if len(own) >= MIN_SITE_EXCEED:
        sites["own"] = own
    if len(sites) < 4:
        return None
    keys, idx, mom = [], [], []
    for k, ex in sites.items():
        lm = sample_lmoments(ex)
        if not lm or lm[0] <= 0:
            continue
        keys.append(k); idx.append(lm[0]); mom.append((lm[3], lm[4]))
    if len(keys) < 4:
        return None
    keep = hw_discordancy(mom) <= DISC_CUT
    z = np.concatenate([sites[k] / i for k, i, kp in zip(keys, idx, keep) if kp])
    if len(z) < MIN_EXC:
        return None
    lp = sample_lmoments(z)
    if not lp:
        return None
    kappa = lp[0] / lp[1] - 2.0
    sigma = lp[0] * (1.0 + kappa)
    index_own = float(own.mean()) if len(own) >= 3 else float(np.mean(idx))
    return {"sey": (kappa, sigma * index_own), "fixed": (-kappa, sigma * index_own)}


def local_gpd(vals, thr):
    ex = vals[vals > thr] - thr
    if len(ex) < MIN_EXC:
        return None
    c, _, s = stats.genpareto.fit(ex, floc=0)
    return float(c), float(s)


def tails(tp, reg_wide, t_lo, t_hi, thr):
    d = tp[(tp.index >= t_lo) & (tp.index < t_hi)].dropna().values
    r = reg_wide[(reg_wide.index >= t_lo) & (reg_wide.index < t_hi)]
    return local_gpd(d, thr), regional_gpd({c: r[c].values for c in r.columns}, d, thr)


# ----------------------------------------------------------------- interval helpers
def csg_fit(pt, actual, thr):
    hv = actual > thr
    if hv.sum() >= 10:
        return m.fit_csg_gamma(np.clip(pt[hv], 0, None), actual[hv])
    rn = actual > m.RAIN_THR_MM
    return m.fit_csg_gamma(np.clip(pt[rn], 0, None), actual[rn])


def csg_q(p, shape, lam, pt):
    return m.csg_quantiles_gamma(0.10, 0.90, p, shape, lam, np.clip(pt, 0, None))


def gpd_q(p, cs, thr):
    return m.gpd_quantiles(0.10, 0.90, p, cs[0], cs[1], thr)


def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def interval_score(y, lo, hi):
    return (hi - lo) + (2 / ALPHA) * np.maximum(lo - y, 0) + (2 / ALPHA) * np.maximum(y - hi, 0)


def slsqp(fun, n):
    r = minimize(fun, np.ones(n) / n, method="SLSQP", bounds=[(0, 1)] * n,
                 constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},))
    w = np.clip(r.x, 0, None)
    return w / w.sum() if np.isfinite(w).all() and w.sum() > 0 else np.ones(n) / n


def fit_weights_pinball(y, Q10, Q90):
    def f(w):
        return pinball(y, Q10 @ w, 0.1).mean() + pinball(y, Q90 @ w, 0.9).mean()
    return slsqp(f, Q10.shape[1])


def fit_weights_sey(y, Q90):
    return slsqp(lambda w: np.mean((y - Q90 @ w) ** 2), Q90.shape[1])


# ----------------------------------------------------------------- CNN preds out-of-sample
def cnn_predictions(Xall, dts, tp, h, heavy_thr):
    tgt = tp.reindex(dts + pd.Timedelta(hours=h)).values
    t_row = dts + pd.Timedelta(hours=1)
    ok = ~np.isnan(tgt)
    is_tr = ok & (t_row < TRAIN_END)
    mu = Xall[is_tr].mean(axis=(0, 2, 3), keepdims=True)
    sd = Xall[is_tr].std(axis=(0, 2, 3), keepdims=True) + 1e-12
    Xn = torch.tensor((Xall - mu) / sd, dtype=torch.float32)
    q_idx = [pn.CHANNELS.index(c) for c in ("q850", "u850", "v850")]
    Xtr = Xn[is_tr].to(pn.DEV)
    Ytr = torch.tensor(tgt[is_tr], dtype=torch.float32).unsqueeze(1).to(pn.DEV)
    QVU = torch.tensor(Xall[is_tr][:, q_idx], dtype=torch.float32).to(pn.DEV)
    net = pn.train_one(Xtr, Ytr, QVU, heavy_thr, 0.0, 0, f"cnn h{h}")   # physics off: it was inert
    pred_rows = t_row >= TRAIN_END
    pr = pn.predict(net, Xn[pred_rows].to(pn.DEV))
    return pd.Series(pr, index=t_row[pred_rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--tag", default="", help="suffix for the output CSV so runs do not overwrite each other")
    a = ap.parse_args()

    raw = m.load_hourly(DISTRICT)
    df = m.build_features(raw, DISTRICT)
    heavy_thr = m.compute_heavy_threshold(df[df["datetime"] < TRAIN_END])
    print(f"heavy threshold (train<2016 p99.5): {heavy_thr:.2f} mm/h", flush=True)

    surf = m.load_surface(DISTRICT).set_index("datetime")["tp_mm"]
    tp = surf.reindex(pd.date_range(surf.index.min(), surf.index.max(), freq="h"))
    reg = pd.read_parquet(os.path.join(ROOT, "data/icechunk_regional_tp/nagaon_regional_tp.parquet"))
    reg["site"] = reg["latitude"].round(2).astype(str) + "_" + reg["longitude"].round(2).astype(str)
    reg_wide = reg.pivot(index="datetime", columns="site", values="tp_mm")
    dts, Xall = pn.load_grid(DISTRICT)

    tail_early = tails(tp, reg_wide, tp.index.min(), TRAIN_END, heavy_thr)
    tail_final = tails(tp, reg_wide, tp.index.min(), FIT_END, heavy_thr)
    for name, (lg, rg) in (("early(<2016)", tail_early), ("final(<2024)", tail_final)):
        print(f"tails {name}: local c={lg[0]:+.3f} scale={lg[1]:.2f} | regional fixed c={rg['fixed'][0]:+.3f} "
              f"scale={rg['fixed'][1]:.2f} (Seychelles-sign c={rg['sey'][0]:+.3f})", flush=True)

    rows = []
    for h in a.leads:
        dfh = df.copy()
        dfh["tp_mm"] = df["tp_mm"].shift(-(h - 1))
        dfh = dfh.dropna(subset=["tp_mm"]).reset_index(drop=True)
        dfh["y_occurrence"] = (dfh["tp_mm"] > m.RAIN_THR_MM).astype(int)
        tr = dfh[dfh["datetime"] < TRAIN_END].reset_index(drop=True)
        rest = dfh[dfh["datetime"] >= TRAIN_END].reset_index(drop=True)
        out, _ = m.run_split(tr, rest, heavy_thr)
        out = out.reset_index(drop=True)
        cnn = cnn_predictions(Xall, dts, tp, h, heavy_thr)
        out["cnn"] = cnn.reindex(out["datetime"]).values
        out = out.dropna(subset=["cnn"]).reset_index(drop=True)
        out["year"] = out["datetime"].dt.year
        fit = out[out["datetime"] < FIT_END].reset_index(drop=True)
        te = out[out["datetime"] >= FIT_END].reset_index(drop=True)
        print(f"\nh={h}: base+CNN trained on {len(tr):,} rows; fit {len(fit):,} (gated {int(fit['gated'].sum())}, "
              f"heavy {int((fit['tp_mm']>heavy_thr).sum())}); test {len(te):,} (gated {int(te['gated'].sum())}, "
              f"heavy {int((te['tp_mm']>heavy_thr).sum())})", flush=True)

        def members(d, csg_params, tail, gated):
            """member q10/q90 arrays (n_gated,) for rows d[gated]"""
            g = d[gated]
            p = np.clip(g["p_rain"].values, 1e-4, 1 - 1e-4)
            (sb, lb), (sc, lc) = csg_params(g)
            lg, rg = tail
            M = {"csg_b": csg_q(p, sb, lb, g["q50_mm"].values),
                 "lgpd": gpd_q(p, lg, heavy_thr) if lg else csg_q(p, sb, lb, g["q50_mm"].values),
                 "rgpd": gpd_q(p, rg["fixed"], heavy_thr) if rg else None,
                 "rgpd_sey": gpd_q(p, rg["sey"], heavy_thr) if rg else None,
                 "csg_c": csg_q(p, sc, lc, g["cnn"].values)}
            return M

        # ---- fit stage: leave-one-year-out CSG params on fit-period rows, early tails
        fit_g = fit["gated"].values
        Mfit = {k: [np.zeros(fit_g.sum()), np.zeros(fit_g.sum())] for k in ("csg_b", "lgpd", "rgpd", "rgpd_sey", "csg_c")}
        gi = np.where(fit_g)[0]
        for y in sorted(fit["year"].unique()):
            other = fit[fit["year"] != y]
            pb = csg_fit(other["q50_mm"].values, other["tp_mm"].values, heavy_thr)
            pc = csg_fit(other["cnn"].values, other["tp_mm"].values, heavy_thr)
            sel = fit_g & (fit["year"].values == y)
            if not sel.any():
                continue
            Mm = members(fit, lambda g, pb=pb, pc=pc: (pb, pc), tail_early, sel)
            pos = np.searchsorted(gi, np.where(sel)[0])
            for k in Mfit:
                Mfit[k][0][pos], Mfit[k][1][pos] = Mm[k][0], Mm[k][1]
        yg = fit["tp_mm"].values[fit_g]
        heavy_fit_g = yg > heavy_thr
        names4 = ["csg_b", "lgpd", "rgpd", "csg_c"]
        Q10 = np.column_stack([Mfit[k][0] for k in names4])
        Q90 = np.column_stack([Mfit[k][1] for k in names4])
        w4 = fit_weights_pinball(yg, Q10, Q90)
        w3 = fit_weights_pinball(yg, Q10[:, :3], Q90[:, :3])
        # Seychelles literal: PINN band 0.5x/1.5x, MSE on q90 vs actual, heavy fit rows only
        cnn_fit_g = fit["cnn"].values[fit_g]
        Q10s = np.column_stack([Q10[:, :3], 0.5 * cnn_fit_g])
        Q90s = np.column_stack([Q90[:, :3], 1.5 * cnn_fit_g])
        ws = fit_weights_sey(yg[heavy_fit_g], Q90s[heavy_fit_g]) if heavy_fit_g.sum() >= 5 else np.ones(4) / 4
        print(f"  weights 4member (csg_b,lgpd,rgpd,csg_c): {np.round(w4,2)}  3member: {np.round(w3,2)}  "
              f"sey4(MSE): {np.round(ws,2)}  [{len(yg)} gated fit rows, {int(heavy_fit_g.sum())} heavy]", flush=True)

        # ---- test stage: CSG params on all fit rows, final tails
        pb = csg_fit(fit["q50_mm"].values, fit["tp_mm"].values, heavy_thr)
        pc = csg_fit(fit["cnn"].values, fit["tp_mm"].values, heavy_thr)
        te_g = te["gated"].values
        Mt = members(te, lambda g: (pb, pc), tail_final, te_g)
        yt = te["tp_mm"].values
        base_lo = te["q10_mm"].values.copy()
        base_hi = np.maximum(te["q90_mm"].values, te["q50_final"].values)

        def build(lo_g, hi_g):
            lo, hi = base_lo.copy(), base_hi.copy()
            lo[te_g], hi[te_g] = lo_g, np.maximum(hi_g, te["q50_final"].values[te_g])
            return lo, hi

        def comb(names, w, key=None):
            lo = sum(wi * Mt[n][0] for wi, n in zip(w, names))
            hi = sum(wi * Mt[n][1] for wi, n in zip(w, names))
            return lo, hi

        arms = {}
        arms["meanof2"] = build(*comb(["csg_b", "lgpd"], [0.5, 0.5]))
        arms["3member"] = build(*comb(["csg_b", "lgpd", "rgpd"], w3))
        arms["4member"] = build(*comb(names4, w4))
        arms["equal4"] = build(*comb(names4, np.ones(4) / 4))
        c_te = te["cnn"].values[te_g]
        lo_s = sum(wi * Mt[n][0] for wi, n in zip(ws[:3], ["csg_b", "lgpd", "rgpd"])) + ws[3] * 0.5 * c_te
        hi_s = sum(wi * Mt[n][1] for wi, n in zip(ws[:3], ["csg_b", "lgpd", "rgpd"])) + ws[3] * 1.5 * c_te
        arms["sey4"] = build(lo_s, hi_s)
        arms["3m_seysign"] = build(*comb(["csg_b", "lgpd", "rgpd_sey"], w3))
        for n in names4:                      # single-member arms: evidence for why weights collapse
            arms[f"only_{n}"] = build(*comb([n], [1.0]))

        hv = yt > heavy_thr
        for arm, (lo, hi) in arms.items():
            cov = float(np.mean((yt[hv] >= lo[hv]) & (yt[hv] <= hi[hv])))
            gcov = float(np.mean((yt[te_g] >= lo[te_g]) & (yt[te_g] <= hi[te_g])))
            rows.append({"lead": h, "arm": arm,
                         "heavy_cov": cov, "heavy_width": float(np.mean(hi[hv] - lo[hv])),
                         "heavy_IS": float(np.mean(interval_score(yt[hv], lo[hv], hi[hv]))),
                         "gated_cov": gcov, "gated_width": float(np.mean(hi[te_g] - lo[te_g])),
                         "gated_IS": float(np.mean(interval_score(yt[te_g], lo[te_g], hi[te_g]))),
                         "n_heavy": int(hv.sum()), "n_gated": int(te_g.sum()),
                         "w_csg_b": {"4member": w4[0], "3member": w3[0], "sey4": ws[0]}.get(arm, np.nan),
                         "w_lgpd": {"4member": w4[1], "3member": w3[1], "sey4": ws[1]}.get(arm, np.nan),
                         "w_rgpd": {"4member": w4[2], "3member": w3[2], "sey4": ws[2]}.get(arm, np.nan),
                         "w_cnn": {"4member": w4[3], "sey4": ws[3]}.get(arm, np.nan)})
        r = pd.DataFrame([x for x in rows if x["lead"] == h])
        print(r[["arm", "heavy_cov", "heavy_width", "heavy_IS", "gated_cov", "gated_width", "gated_IS"]]
              .round(3).to_string(index=False), flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(ROOT, f"data/rainfall_nowcast/assembler_nagaon_summary{a.tag}.csv"), index=False)
    print("\nSaved data/rainfall_nowcast/assembler_nagaon_summary.csv")


if __name__ == "__main__":
    main()
