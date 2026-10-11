"""
Regional GPD (index-flood Regional Frequency Analysis) as a third heavy-interval leg.

Ported from Seychelles approach_b_nwp/101_production_ensemble.py (fit_regional_gpd,
sample_lmoments, hosking_wallis_discordancy) and 98_regional_frequency_analysis.py
(read in full 2026-10-09). Method: Hosking & Wallis (1997), pool exceedances over a
common threshold from nearby sites, normalise each by its own mean exceedance
(index-flood), fit one L-moment GPD to the pooled dimensionless sample, rescale to the
target site's own index.

It ONLY reshapes the q10/q90 interval on gated (heavy) rows. q50_final is untouched,
exactly as in the existing CSG-Gamma + Local-GPD construction in 20_base_rainfall.py.

SOURCE BUG, FIXED HERE (verified numerically 2026-10-09, Chroma manual-0106):
Seychelles passes Hosking's kappa = L1/L2 - 2 straight into scipy.stats.genpareto as
`c`. SciPy's c is the extreme-value index xi = -kappa. This port uses c = -kappa. The
arm "mean3_seychelles_sign" reproduces the as-written Seychelles behaviour only to show
what the bug does; it is a diagnostic, not a candidate.

Other deliberate points:
  * One absolute threshold per district (its adaptive heavy threshold, train p99.5 of
    rainy hours) applied at all 25 grid sites, as Seychelles applied 10 mm at all sites.
  * Seychelles adds the target's own record as an extra site on top of the 25-cell grid.
    Here the local record and the grid-centre cell are the same ERA5 cell, so if they
    correlate above 0.99 the grid-centre cell is dropped from the pool to avoid counting
    one record twice (printed per district).
  * The 25 neighbouring 0.25 deg cells are strongly spatially correlated, which violates
    RFA's intersite-independence assumption (Hosking & Wallis 1997). Kept, not hidden.
  * Regional fit uses TRAIN period only (< 2024-01-01).

Arms (all on the same gated rows, same split, same base-model code):
  mean2                  current construction: (CSG-Gamma + Local GPD) / 2
  mean3                  (CSG-Gamma + Local GPD + Regional GPD) / 3, correct sign
  regional_only          Regional GPD alone
  mean3_seychelles_sign  as mean3 but with Seychelles' sign (diagnostic)

SUCCESS CRITERION, fixed before running (TASK.md 2026-10-09, Chroma manual-0106):
  PRIMARY: heavy-event interval score (alpha 0.2, lower is better) of mean3 is lower
  than mean2 in >= 18 of 24 (district x lead) cells.
  SECONDARY (report only): heavy coverage vs 80% nominal, width. n_heavy in test is
  small (tens of rows), so coverage has a standard error of roughly 5%.

Run:  python -u scripts/100_regional_gpd.py Nagaon
      python -u scripts/100_regional_gpd.py summary
"""
import importlib
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m = importlib.import_module("20_base_rainfall")
ROOT = m.ROOT
LEADS = [1, 2, 3, 4, 5, 6]
TEST_START = pd.Timestamp(m.TEST_SPLIT_DATE)
BASE_FEATS = list(m.FEATURE_COLS)
MIN_SITE_EXCEED = 5
DISCORDANCY_CUTOFF = 3.0
OUT_DIR = os.path.join(ROOT, "data", "rainfall_nowcast")
REG_DIR = os.path.join(ROOT, "data", "icechunk_regional_tp")
ARMS = ("mean2", "mean3", "regional_only", "mean3_seychelles_sign", "mean3_hw", "regional_only_hw")


def sample_lmoments(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 3:
        return None
    i = np.arange(1, n + 1)
    b0 = np.mean(x)
    b1 = np.sum(((i - 1) / (n - 1)) * x) / n
    b2 = np.sum(((i - 1) * (i - 2) / ((n - 1) * (n - 2))) * x) / n
    L1, L2, L3 = b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0
    if L2 <= 0:
        return None
    return L1, L2, L3, L2 / L1, L3 / L2


def gpd_from_lmoments(L1, L2):
    """Hosking (1990): L1 = sigma/(1+k), L2 = sigma/((1+k)(2+k)). Returns (k, sigma)."""
    k = L1 / L2 - 2.0
    return k, L1 * (1.0 + k)


def discordancy(site_moments, mode="sey"):
    """mode "sey": Seychelles as written, (N/2) * Mahalanobis^2 with np.cov (divisor N-1).
    That is ~40x larger than Hosking & Wallis' D and flags ~21 of 25 sites at the cutoff of 3
    (verified 2026-10-09). mode "hw": D_i = (N/3) (u_i-ubar)' A^-1 (u_i-ubar), A = sum of
    outer products (Hosking & Wallis 1993/1997); its mean over sites is exactly 2/3."""
    N = len(site_moments)
    if N < 4:
        return np.zeros(N)
    U = np.array(site_moments)
    dev = U - U.mean(axis=0)
    try:
        if mode == "hw":
            Ainv = np.linalg.inv(dev.T @ dev)
            return (N / 3.0) * np.einsum("ij,jk,ik->i", dev, Ainv, dev)
        Sinv = np.linalg.inv(np.cov(U, rowvar=False))
    except np.linalg.LinAlgError:
        return np.zeros(N)
    return np.array([(N / 2.0) * (d @ Sinv @ d) for d in dev])


def fit_regional_gpd(reg_train, local_actual, threshold, drop_centre_latlon=None, disc_mode="sey"):
    """Returns (kappa_R, scale_local, diag) or None. kappa_R is HOSKING's kappa (xi = -kappa)."""
    site_data = {}
    for (lat, lon), g in reg_train.groupby(["latitude", "longitude"]):
        if drop_centre_latlon is not None and (lat, lon) == drop_centre_latlon:
            continue
        v = g["tp_mm"].values
        ex = v[v > threshold] - threshold
        if len(ex) >= MIN_SITE_EXCEED:
            site_data[(lat, lon)] = ex
    local_ex = local_actual[local_actual > threshold] - threshold
    if len(local_ex) >= MIN_SITE_EXCEED:
        site_data[("local", "local")] = local_ex
    if len(site_data) < 4:
        return None
    keys, idx, lm = [], [], []
    for k, ex in site_data.items():
        s = sample_lmoments(ex)
        if not s or s[0] <= 0:
            continue
        keys.append(k); idx.append(s[0]); lm.append((s[3], s[4]))
    if len(keys) < 4:
        return None
    Di = discordancy(lm, disc_mode)
    keep = Di <= DISCORDANCY_CUTOFF
    pooled = []
    for k, index_s, ok in zip(keys, idx, keep):
        if ok:
            pooled.extend((site_data[k] / index_s).tolist())
    pooled = np.array(pooled)
    if len(pooled) < m.MIN_EXCEEDANCES:
        return None
    sp = sample_lmoments(pooled)
    if not sp:
        return None
    kappa, sigma = gpd_from_lmoments(sp[0], sp[1])
    index_local = float(np.mean(local_ex)) if len(local_ex) >= 3 else float(np.mean(idx))
    return kappa, sigma * index_local, {"n_sites": len(keys), "n_dropped": int((~keep).sum()),
                                        "n_pooled": len(pooled), "index_local": index_local,
                                        "sigma_R": sigma}


def interval_score(y, lo, hi, a=0.2):
    return (hi - lo) + (2 / a) * np.maximum(lo - y, 0) + (2 / a) * np.maximum(y - hi, 0)


def run_district(district):
    t_all = time.time()
    raw = m.load_hourly(district)
    df = m.build_features(raw, district)
    thr = m.compute_heavy_threshold(df[df["datetime"] < TEST_START])
    print(f"{district}: heavy threshold {thr:.2f} mm/h", flush=True)

    sp = m.PATH_SPELLING[district]["pressure"]
    reg = pd.read_parquet(os.path.join(REG_DIR, f"{sp}_regional_tp.parquet"))
    reg["datetime"] = pd.to_datetime(reg["datetime"])
    reg_train = reg[reg["datetime"] < TEST_START]
    lat0, lon0 = m.DISTRICTS[district]
    centre = reg_train[(reg_train.latitude == lat0) & (reg_train.longitude == lon0)].set_index("datetime")["tp_mm"]
    loc = df[df["datetime"] < TEST_START].set_index("datetime")["tp_mm"]
    j = centre.to_frame("g").join(loc.rename("l"), how="inner").dropna()
    corr = float(np.corrcoef(j["g"], j["l"])[0, 1])
    drop = (lat0, lon0) if corr > 0.99 else None
    print(f"local record vs grid-centre cell corr = {corr:.4f}; "
          f"{'dropping grid centre from pool (same cell)' if drop else 'keeping grid centre'}", flush=True)

    rows = []
    for h in LEADS:
        dfh = df.copy()
        dfh["tp_mm"] = df["tp_mm"].shift(-(h - 1))
        dfh = dfh.dropna(subset=["tp_mm"]).reset_index(drop=True)
        dfh["y_occurrence"] = (dfh["tp_mm"] > m.RAIN_THR_MM).astype(int)
        tr = dfh[dfh["datetime"] < TEST_START].reset_index(drop=True)
        te = dfh[dfh["datetime"] >= TEST_START].reset_index(drop=True)
        m.FEATURE_COLS = BASE_FEATS
        out, models = m.run_split(tr, te, thr)
        q_models = models[1]

        train_rain = tr[tr["tp_mm"] > m.RAIN_THR_MM]
        tpp = np.clip(q_models[0.5].predict(train_rain[BASE_FEATS]), 0, None)
        ta = train_rain["tp_mm"].values
        hv_tr = ta > thr
        shape_l, lam_l = m.fit_csg_gamma(tpp[~hv_tr], ta[~hv_tr])
        shape_h, lam_h = m.fit_csg_gamma(tpp[hv_tr], ta[hv_tr]) if hv_tr.sum() >= 10 else (shape_l, lam_l)
        lgpd = m.fit_gpd_tail(ta, thr)
        rfit = fit_regional_gpd(reg_train, tr["tp_mm"].values, thr, drop, "sey")
        rfit_hw = fit_regional_gpd(reg_train, tr["tp_mm"].values, thr, drop, "hw")

        p = np.clip(out["p_rain"].values, 1e-4, 1 - 1e-4)
        gated = out["gated"].values.astype(bool)
        point = out["q50_mm"].values
        q50f = out["q50_final"].values
        y = out["tp_mm"].values
        base10, base90 = out["q10_mm"].values.copy(), out["q90_mm"].values.copy()

        q10g, q90g = m.csg_quantiles_gamma(0.10, 0.90, p[gated], shape_h, lam_h, point[gated])
        q10l, q90l = (m.gpd_quantiles(0.10, 0.90, p[gated], lgpd[0], lgpd[1], thr) if lgpd else (q10g, q90g))

        def assemble(q10_g_rows, q90_g_rows):
            a10, a90 = base10.copy(), base90.copy()
            a10[gated], a90[gated] = q10_g_rows, q90_g_rows
            return a10, np.maximum(a90, q50f)

        arms = {"mean2": assemble((q10g + q10l) / 2, (q90g + q90l) / 2)}
        # regression check: mean2 must reproduce the shipped interval exactly
        assert np.allclose(arms["mean2"][0], out["q10_ens"].values, atol=1e-6), "mean2 q10 != shipped q10_ens"
        assert np.allclose(arms["mean2"][1], out["q90_ens"].values, atol=1e-6), "mean2 q90 != shipped q90_ens"
        if rfit and gated.sum() > 0:
            kap, scale, dg = rfit
            q10r, q90r = m.gpd_quantiles(0.10, 0.90, p[gated], -kap, scale, thr)         # correct: c = -kappa
            q10w, q90w = m.gpd_quantiles(0.10, 0.90, p[gated], kap, scale, thr)          # Seychelles as written
            arms["mean3"] = assemble((q10g + q10l + q10r) / 3, (q90g + q90l + q90r) / 3)
            arms["regional_only"] = assemble(q10r, q90r)
            arms["mean3_seychelles_sign"] = assemble((q10g + q10l + q10w) / 3, (q90g + q90l + q90w) / 3)
            diag = {"kappa_R": kap, "xi_R": -kap, "scale_local": scale, **dg}
            if rfit_hw:
                kh, sch, dgh = rfit_hw
                q10h, q90h = m.gpd_quantiles(0.10, 0.90, p[gated], -kh, sch, thr)
                arms["mean3_hw"] = assemble((q10g + q10l + q10h) / 3, (q90g + q90l + q90h) / 3)
                arms["regional_only_hw"] = assemble(q10h, q90h)
                diag.update({"xi_R_hw": -kh, "n_sites_hw": dgh["n_sites"], "n_dropped_hw": dgh["n_dropped"],
                             "n_pooled_hw": dgh["n_pooled"]})
        else:
            diag = {}
        hv = y > thr
        for name, (a10, a90) in arms.items():
            rows.append({"district": district, "lead": h, "arm": name,
                         "heavy_cov": float(np.mean((y[hv] >= a10[hv]) & (y[hv] <= a90[hv]))),
                         "heavy_width": float(np.mean(a90[hv] - a10[hv])),
                         "heavy_IS": float(np.mean(interval_score(y[hv], a10[hv], a90[hv]))),
                         "n_heavy": int(hv.sum()), "n_gated": int(gated.sum()), **diag})
        r2 = [r for r in rows if r["lead"] == h and r["district"] == district]
        print(f"h={h}: n_heavy={hv.sum()} gated={gated.sum()} regional={'ok xi=%.3f' % diag['xi_R'] if diag else 'NONE'} | " +
              "  ".join(f"{r['arm']}: cov={r['heavy_cov']:.3f} IS={r['heavy_IS']:.1f}" for r in r2), flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"regional_gpd_{district.lower()}_summary.csv"), index=False)
    print(f"{district} done in {(time.time()-t_all)/60:.1f} min", flush=True)


def summary():
    files = [os.path.join(OUT_DIR, f"regional_gpd_{d.lower()}_summary.csv") for d in m.DISTRICTS]
    res = pd.concat([pd.read_csv(f) for f in files if os.path.exists(f)], ignore_index=True)
    print(f"districts present: {sorted(res['district'].unique())}")
    w = res.pivot_table(index=["district", "lead"], columns="arm", values="heavy_IS")
    n = len(w.dropna(subset=["mean2", "mean3"]))
    better = int((w["mean3"] < w["mean2"]).sum())
    need = int(np.ceil(0.75 * n))
    ok = better >= need
    print(f"\nPRIMARY: mean3 interval score < mean2 in {better}/{n} cells (need >={need}) -> "
          f"{'PASS' if ok else 'FAIL'}")
    print(f"  mean IS by arm:\n{w.mean().round(2).to_string()}")
    for col in ("heavy_cov", "heavy_width"):
        ww = res.pivot_table(index=["district", "lead"], columns="arm", values=col)
        print(f"\nSECONDARY mean {col} by arm (nominal coverage 0.80):\n{ww.mean().round(3).to_string()}")
    d = res[res["arm"] == "regional_only"].groupby("district")[
        ["xi_R", "n_sites", "n_dropped", "n_pooled", "xi_R_hw", "n_dropped_hw", "n_pooled_hw"]].mean()
    print(f"\nRegional fit diagnostics (xi_R = scipy shape, >0 heavy tail; _hw = corrected Hosking-Wallis "
          f"discordancy):\n{d.round(3).to_string()}")
    if "mean3_hw" in w.columns:
        n2 = len(w.dropna(subset=["mean2", "mean3_hw"]))
        b2 = int((w["mean3_hw"] < w["mean2"]).sum())
        need2 = int(np.ceil(0.75 * n2))
        print(f"\nCORRECTED-DISCORDANCY REPLICATION (added AFTER seeing the as-ported result, so NOT part of the "
              f"pre-registered rule): mean3_hw IS < mean2 in {b2}/{n2} cells (bar >={need2}) -> "
              f"{'PASS' if b2 >= need2 else 'FAIL'}; mean IS {w['mean3_hw'].mean():.2f}")
    wb = (w["mean3_seychelles_sign"] < w["mean3"]).sum()
    print(f"\nDIAGNOSTIC: Seychelles sign gives a lower IS than the corrected sign in {wb}/{n} cells "
          f"(mean IS {w['mean3_seychelles_sign'].mean():.2f} vs {w['mean3'].mean():.2f})")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "Nagaon"
    if arg == "summary":
        summary()
    elif arg in m.DISTRICTS:
        run_district(arg)
    else:
        print(f"Unknown '{arg}'. Choices: {list(m.DISTRICTS)} or summary")
