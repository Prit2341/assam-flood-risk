"""
Flood susceptibility + rainfall-evaluation endpoints for the dashboard.

Everything served here is read from files produced by scripts 120-170 (no synthetic values):
  data/rainfall_nowcast/v1_rolling_daily_summary.csv, v1_1_heavy_threshold_summary.csv
  data/dynamic_susceptibility/{ahp_weights.json, static/*.npz, labels/*, met/*, criteria.csv, eval_*.csv, maps/*}
The dynamic AHP map (A2) needs no training, so it is computed on request for any date with met data.
ML maps (H1/H2/H4) appear only after scripts/170 'maps' has been run.
"""
import importlib.util
import io
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

ROOT = Path(__file__).resolve().parents[2]
DYN = ROOT / "data" / "dynamic_susceptibility"
RAIN = ROOT / "data" / "rainfall_nowcast"
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("dyn170", ROOT / "scripts" / "170_dynamic_susceptibility.py")
dyn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dyn)

router = APIRouter(prefix="/api/susc")
rain_router = APIRouter(prefix="/api/rain")
ID2D = {"nagaon": "Nagaon", "marigaon": "Morigaon", "morigaon": "Morigaon", "karimganj": "Karimganj", "sivasagar": "Sivasagar"}
STATIC_LAYERS = {"hand": "Height above nearest drainage (m)", "elevation": "Elevation (m)", "slope": "Slope (°)",
                 "twi": "Topographic wetness index", "dist_major": "Distance to major river (km)",
                 "drain_dens": "Drainage density (km/km²)", "ndvi": "NDVI (dry season)", "lulc": "Land cover (Dynamic World)"}


def _d(district: str) -> str:
    if district.lower() not in ID2D:
        raise HTTPException(404, f"unknown district {district}")
    return ID2D[district.lower()]


def _clean(df: pd.DataFrame):
    return json.loads(df.replace({np.nan: None}).to_json(orient="records"))


# ------------------------------------------------------------------ rainfall evaluation (honest baselines)
@rain_router.get("/evaluation")
def rain_evaluation():
    p = RAIN / "v1_rolling_daily_summary.csv"
    if not p.exists():
        raise HTTPException(404, "run scripts/120_v1_rolling_daily.py")
    s = pd.read_csv(p)
    out = {"summary": _clean(s[["district", "scale", "forecast", "corr", "rmse", "bias_pct", "SS_vs_persistence",
                                "heavy_POD", "heavy_FAR", "heavy_CSI", "modplus_CSI", "imdheavy_CSI"]])}
    t = RAIN / "v1_1_heavy_threshold_summary.csv"
    if t.exists():
        out["threshold_test"] = _clean(pd.read_csv(t))
    return out


@rain_router.get("/day/{district}")
def rain_day(district: str, date: str):
    """24 hourly values for one test-period day: observed, v1 1-h-ahead prediction, persistence."""
    d = {"Morigaon": "marigaon"}.get(_d(district), _d(district).lower())
    p = RAIN / f"v1_rolling1h_{d}_hourly.csv"
    if not p.exists():
        raise HTTPException(404, "run scripts/120_v1_rolling_daily.py")
    h = _hourly(str(p))
    day = h[h["datetime"].dt.strftime("%Y-%m-%d") == date]
    if day.empty:
        return {"date": date, "available": False,
                "reason": "No out-of-sample prediction for this date (test period is 2024-01-01 to 2026-03-31; "
                          "earlier dates were used for training)."}
    return {"date": date, "available": True, "hours": day["datetime"].dt.strftime("%H:%M").tolist(),
            "observed": day["tp_mm"].round(3).tolist(), "v1": day["q50_final"].round(3).tolist(),
            "v1_q90": day["q90_ens"].round(3).tolist(), "persistence": day["persistence"].round(3).tolist(),
            "total_obs": round(float(day["tp_mm"].sum()), 1), "total_v1": round(float(day["q50_final"].sum()), 1)}


@lru_cache(maxsize=8)
def _hourly(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["datetime"])


# ------------------------------------------------------------------ susceptibility status + results
@router.get("/status")
def status():
    labels = {}
    for d in dyn.dc.DISTRICTS:
        f = DYN / "labels" / d / "labels_index.csv"
        if f.exists():
            i = pd.read_csv(f)
            labels[d] = {"scenes_checked": int(len(i)), "usable_scenes": int((i["obs_frac"] >= dyn.MIN_OBS_FRAC).sum()),
                         "latest_date": str(i["date"].max())}
    w = json.load(open(DYN / "ahp_weights.json")) if (DYN / "ahp_weights.json").exists() else None
    crit = _clean(pd.read_csv(DYN / "criteria.csv")) if (DYN / "criteria.csv").exists() else None
    pooled = DYN / "eval_loyo_pooled.csv"
    res = _clean(pd.read_csv(pooled)) if pooled.exists() else None
    final = crit is not None and len(labels) == 4
    return {"labels": labels, "ahp": w, "criteria": crit, "results": res, "results_final": final,
            "static_layers": STATIC_LAYERS}


@router.get("/dates/{district}")
def dates(district: str):
    d = _d(district)
    f = DYN / "labels" / d / "labels_index.csv"
    if not f.exists():
        return []
    i = pd.read_csv(f)
    i = i[i["obs_frac"] >= dyn.MIN_OBS_FRAC]
    return _clean(i[["date", "obs_frac", "flood_frac_obs"]].sort_values("date"))


# ------------------------------------------------------------------ rasters -> PNG overlays
@lru_cache(maxsize=4)
def _static(d: str):
    z = dict(np.load(DYN / "static" / f"{d}_static.npz"))
    usable = z.pop("usable")
    g = np.load(DYN / "labels" / d / "_grid.npz") if (DYN / "labels" / d / "_grid.npz").exists() else None
    tr = rasterio.Affine(*g["transform"]) if g is not None else None
    return z, usable, tr


@lru_cache(maxsize=4)
def _met(d: str):
    return dyn.met_features(d)


def _bounds(tr, shape):
    h, w = shape
    return [[tr.f + h * tr.e, tr.c], [tr.f, tr.c + w * tr.a]]


def _png(arr, cmap, vmin, vmax, categorical=None):
    from PIL import Image
    import matplotlib
    rgba = np.zeros(arr.shape + (4,), np.uint8)
    ok = ~np.isnan(arr)
    if categorical is not None:
        for k, col in categorical.items():
            m = ok & (np.round(arr) == k)
            rgba[m] = col
    else:
        cm = SUSC_CMAP if cmap == "susc" else matplotlib.colormaps[cmap]
        x = np.clip((arr - vmin) / (vmax - vmin + 1e-12), 0, 1)
        rgba[ok] = (np.array(cm(x[ok])) * 255).astype(np.uint8)
        rgba[ok, 3] = 210
    b = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(b, "PNG")
    return b.getvalue()


def _susc_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    # light blue (low) -> very pale blue -> pale yellow -> orange -> dark red (high)
    return LinearSegmentedColormap.from_list("susc", [(0, "#9ecae1"), (0.27, "#deebf7"), (0.54, "#ffffbf"),
                                                      (0.81, "#fdae61"), (0.9, "#f46d43"), (1, "#a50026")])


SUSC_CMAP = _susc_cmap()

DW_COLORS = {0: (65, 155, 223, 220), 1: (57, 125, 73, 220), 2: (136, 176, 83, 220), 3: (122, 135, 198, 220),
             4: (228, 150, 53, 220), 5: (223, 195, 90, 220), 6: (196, 40, 27, 220), 7: (165, 155, 143, 220)}


@router.get("/layer/{district}/{name}.png")
def layer_png(district: str, name: str, date: str = Query(default=None)):
    d = _d(district)
    S, usable, tr = _static(d)
    if name in S:
        a = np.where(usable, S[name], np.nan)
        if name == "lulc":
            return Response(_png(a, None, 0, 1, categorical=DW_COLORS), media_type="image/png")
        lo, hi = np.nanpercentile(a, [2, 98])
        return Response(_png(a, "viridis" if name != "hand" else "viridis_r", lo, hi), media_type="image/png")
    if name == "observed":
        f = DYN / "labels" / d / f"{date}.npz"
        if not f.exists():
            raise HTTPException(404, "no GFM scene for this date")
        lab = np.load(f)["label"].astype(float)
        lab[lab == 255] = np.nan
        return Response(_png(lab, None, 0, 1, categorical={0: (148, 163, 184, 90), 1: (220, 38, 38, 230)}),
                        media_type="image/png")
    if name in ("ml_h2_rel", "ml_h5_rel", "ml_h6_rel"):
        # percentile rank of each cell within this date: shows WHERE is most prone on any date, not how much floods
        a = {"ml_h2_rel": _h2_prob, "ml_h5_rel": _h5_map, "ml_h6_rel": _h6_map}[name](d, date)
        v = ~np.isnan(a)
        r = np.full(a.shape, np.nan)
        r[v] = (np.argsort(np.argsort(a[v])) + 0.5) / v.sum()
        return Response(_png(r, "susc", 0, 1), media_type="image/png")
    if name in ("ml_h2", "ml_h5", "ml_h6"):
        a = {"ml_h2": _h2_prob, "ml_h5": _h5_map, "ml_h6": _h6_map}[name](d, date)
        # one fixed LOG scale for every date, 0.01% .. 50%: probabilities span ~4 orders of magnitude (dry-day median
        # ~1e-4, flood-day median ~0.12), so a linear scale painted every dry-day cell the same pale colour
        lg = np.log10(np.clip(a, 1e-4, 0.5))
        return Response(_png(lg, "susc", -4, np.log10(0.5)), media_type="image/png")
    if name == "ahp_dynamic":
        a, _ = _ahp_map(d, date)
        return Response(_png(a, "susc", 1, 5), media_type="image/png")
    m = DYN / "maps" / f"{d}_{date}_{name}.tif"
    if m.exists():
        with rasterio.open(m) as s:
            a = s.read(1).astype(float)
            a[a == s.nodata] = np.nan
        return Response(_png(a, "susc", 0, np.nanpercentile(a, 99)), media_type="image/png")
    raise HTTPException(404, f"layer {name} not available")


@lru_cache(maxsize=4)
def _near(d):
    """Nearest 0.1-degree met point for every 90 m cell (same mapping as scripts/170)."""
    S, usable, tr = _static(d)
    pts = np.array(list(_met(d).keys()))
    h, wd = usable.shape
    rr, cc = np.mgrid[0:h, 0:wd]
    from scipy.spatial import cKDTree
    return pts, cKDTree(pts).query(np.c_[(tr.f + (rr + .5) * tr.e).ravel(), (tr.c + (cc + .5) * tr.a).ravel()])[1]


@lru_cache(maxsize=4)
def _daily_rain(d):
    """District-mean IMERG daily rain (mm), each 0.1-degree point weighted by how many usable cells map to it."""
    S, usable, tr = _static(d)
    pts, near = _near(d)
    wts = np.bincount(near[usable.ravel()], minlength=len(pts)).astype(float)
    met = pd.read_parquet(DYN / "met" / f"{d}_daily_met.parquet")
    met["date"] = pd.to_datetime(met["date"])
    idx = {tuple(p): i for i, p in enumerate(pts)}
    met["w"] = [wts[idx[(a, b)]] for a, b in zip(met["lat"], met["lon"])]
    met = met[met["w"] > 0]
    g = met.assign(rw=met["rain_mm"] * met["w"]).groupby("date")[["rw", "w"]].sum()
    return (g["rw"] / g["w"]).asfreq("D")


@router.get("/rain_window/{district}")
def rain_window(district: str, date: str):
    """Daily rain for the 10 days up to the chosen date; the AHP rain factor uses the 5 days before it."""
    d = _d(district)
    s = _daily_rain(d)
    t = pd.Timestamp(date)
    w = s.reindex(pd.date_range(t - pd.Timedelta(days=9), t))
    return {"date": date, "source": "GPM IMERG V07 daily, district mean",
            "days": [x.strftime("%Y-%m-%d") for x in w.index],
            "rain_mm": [None if np.isnan(v) else round(float(v), 1) for v in w.values],
            "in_r5": [bool(t - pd.Timedelta(days=5) <= x < t) for x in w.index]}


def _ahp_map(d, date):
    S, usable, tr = _static(d)
    t = pd.Timestamp(date)
    valid = usable & np.all([~np.isnan(S[k]) for k in dyn.STATIC_RAW], axis=0)
    R = dyn.ahp_static_ranks(S, valid)
    w = json.load(open(DYN / "ahp_weights.json"))["weights_10"]
    met = _met(d)
    pts, near = _near(d)
    h, wd = valid.shape
    r5 = np.array([met[tuple(p)]["r5"].get(t, np.nan) for p in pts])[near].reshape(h, wd)
    if np.all(np.isnan(r5)):
        raise HTTPException(404, "no rainfall data for this date")
    score = sum(w[k] * R[k] for k in dyn.AHP_F if k != "rainfall") + w["rainfall"] * dyn.amc_rank(r5)
    score = np.where(valid, score, np.nan)
    return score, {"r5_mean_mm": float(np.nanmean(np.where(valid, r5, np.nan))),
                   "amc_class": {1: "I (dry)", 3: "II (normal)", 5: "III (wet)"}[int(np.nanmedian(dyn.amc_rank(r5[valid])))]}


H2_DIR = DYN / "models_h2_loyo"
GLOFAS_H2 = {**dyn.GLOFAS, "Karimganj": ("karimganj_kushiyara", "karimganj_barak")}   # Amendment 2, check K


@lru_cache(maxsize=4)
def _glofas(tag):
    return dyn.glofas_features(tag)


@lru_cache(maxsize=16)
def _h2_model(year: int):
    from xgboost import XGBClassifier
    f = H2_DIR / f"h2_without_{year}.json"
    if not f.exists():
        return None
    m = XGBClassifier()
    m.load_model(str(f))
    return m


@lru_cache(maxsize=8)
def _h2_map(d, date):
    """Dynamic ML (H2) flood probability for every cell on this date, from the model trained WITHOUT this
    date's year (scripts/171 models), so past dates are mapped out-of-sample."""
    t = pd.Timestamp(date)
    mdl = _h2_model(t.year)
    if mdl is None:
        raise HTTPException(404, f"no held-out-year H2 model for {t.year}")
    S, usable, tr = _static(d)
    valid = usable & np.all([~np.isnan(S[k]) for k in dyn.STATIC_RAW], axis=0)
    met = _met(d)
    pts, near = _near(d)
    cols = list(next(iter(met.values())).columns)
    mf = np.stack([met[tuple(p)].reindex([t]).iloc[0].values for p in pts])
    ii = np.flatnonzero(valid.ravel())
    X = pd.DataFrame({k: S[k].ravel()[ii] for k in dyn.STATIC_RAW})
    for c_i, c in enumerate(cols):
        X[c] = mf[near[ii], c_i]
    if X["r5"].isna().all():
        raise HTTPException(404, "no rainfall data for this date")
    for name, tag in zip(("q_trib", "q_main"), GLOFAS_H2[d]):
        g = _glofas(tag) if tag else None
        ok = g is not None and t in g.index
        X[name] = g.loc[t, "q"] if ok else np.nan
        X[f"{name}_max7"] = g.loc[t, "qmax7"] if ok else np.nan
    p = np.full(valid.size, np.nan)
    p[ii] = mdl.predict_proba(X[json.load(open(H2_DIR / "meta.json"))["features"]])[:, 1]
    return p.reshape(valid.shape), X, ii, valid


def _h2_prob(d, date):
    return _h2_map(d, date)[0]


H5_DIR = DYN / "models_h5_loyo"
H5_HIST = DYN / "h5_history"
B0_MIN_HIST = 10      # scripts/172 MIN_HIST: pixels with fewer observed training scenes get b0s = NaN


@lru_cache(maxsize=16)
def _h5_model(year: int):
    from xgboost import XGBClassifier
    f = H5_DIR / f"h5_without_{year}.json"
    if not f.exists():
        return None
    m = XGBClassifier()
    m.load_model(str(f))
    return m


def _h5_available(d, year):
    return (H5_DIR / f"h5_without_{year}.json").exists() and (H5_HIST / f"{d}_{year}.npz").exists()


@lru_cache(maxsize=8)
def _h5_map(d, date):
    """H5 (Amendment 4 working model): H2 inputs + per-pixel flood history (ff, b0s), model and history both
    built without this date's year (scripts/175), so past dates are held-out predictions."""
    t = pd.Timestamp(date)
    mdl = _h5_model(t.year)
    if mdl is None or not _h5_available(d, t.year):
        raise HTTPException(404, f"no held-out-year H5 model/history for {t.year}")
    _, X, ii, valid = _h2_map(d, date)
    z = np.load(H5_HIST / f"{d}_{t.year}.npz")
    thr, drv, nobs = z["thr"].ravel()[ii], z["drv"].ravel()[ii], z["nobs"].ravel()[ii]
    q = np.where(drv == 1, X["q_main"].values, X["q_trib"].values)
    inf = np.isinf(thr)
    b0s = np.where(inf, 0.0, q / np.where(inf, 1, thr))
    b0s[(nobs < B0_MIN_HIST) | np.isnan(q)] = np.nan
    X5 = X.assign(ff=z["ff"].ravel()[ii], b0s=b0s)
    p = np.full(valid.size, np.nan)
    p[ii] = mdl.predict_proba(X5[json.load(open(H5_DIR / "meta.json"))["features"]])[:, 1]
    return p.reshape(valid.shape)


H6_DIR = DYN / "models_h6_loyo"
H6_MAXGAP = 30        # scripts/179 MAXGAP


@lru_cache(maxsize=16)
def _h6_model(year: int):
    from xgboost import XGBClassifier
    f = H6_DIR / f"h6_without_{year}.json"
    if not f.exists():
        return None
    m = XGBClassifier()
    m.load_model(str(f))
    return m


def _h6_available(d, year):
    return (H6_DIR / f"h6_without_{year}.json").exists() and _h5_available(d, year)


def _prev_scene(d, date):
    """Past-only previous-scene features (scripts/179): per pixel the GFM state at the most recent usable scene
    1-30 days before `date` where it was observed, its age, and the previous scene's district flooded fraction."""
    t = pd.Timestamp(date)
    idx = pd.read_csv(DYN / "labels" / d / "labels_index.csv", parse_dates=["date"])
    idx = idx[(idx["obs_frac"] >= dyn.MIN_OBS_FRAC) & (idx["date"] < t)].sort_values("date")
    S, usable, tr = _static(d)
    state = np.full(usable.shape, np.nan, np.float32)
    gap = np.full(usable.shape, np.nan, np.float32)
    for _, r in idx[idx["date"] >= t - pd.Timedelta(days=H6_MAXGAP)].iterrows():
        lab = np.load(DYN / "labels" / d / f"{r['date']:%Y-%m-%d}.npz")["label"]
        seen = lab != 255
        state[seen] = lab[seen] == 1
        gap[seen] = (t - r["date"]).days
    pf = pfg = np.nan
    if len(idx):
        g = (t - idx.iloc[-1]["date"]).days
        if 1 <= g <= H6_MAXGAP:
            pf, pfg = float(idx.iloc[-1]["flood_frac_obs"]), float(g)
    return state, gap, pf, pfg


@lru_cache(maxsize=8)
def _h6_map(d, date):
    """H6 (Amendment 6 working model): H5 inputs + previous-scene state, held-out-year model (scripts/180)."""
    t = pd.Timestamp(date)
    mdl = _h6_model(t.year)
    if mdl is None or not _h6_available(d, t.year):
        raise HTTPException(404, f"no held-out-year H6 model/history for {t.year}")
    _, X, ii, valid = _h2_map(d, date)
    z = np.load(H5_HIST / f"{d}_{t.year}.npz")
    thr, drv, nobs = z["thr"].ravel()[ii], z["drv"].ravel()[ii], z["nobs"].ravel()[ii]
    q = np.where(drv == 1, X["q_main"].values, X["q_trib"].values)
    inf = np.isinf(thr)
    b0s = np.where(inf, 0.0, q / np.where(inf, 1, thr))
    b0s[(nobs < B0_MIN_HIST) | np.isnan(q)] = np.nan
    state, gap, pf, pfg = _prev_scene(d, date)
    X6 = X.assign(ff=z["ff"].ravel()[ii], b0s=b0s, prev_wet=state.ravel()[ii], prev_gap=gap.ravel()[ii],
                  prev_frac=pf, prev_frac_gap=pfg)
    p = np.full(valid.size, np.nan)
    p[ii] = mdl.predict_proba(X6[json.load(open(H6_DIR / "meta.json"))["features"]])[:, 1]
    return p.reshape(valid.shape)


@router.get("/map/{district}")
def map_meta(district: str, date: str):
    """Bounds + which overlays exist for this district/date, plus the AHP summary for that date."""
    d = _d(district)
    S, usable, tr = _static(d)
    score, info = _ahp_map(d, date)
    v = score[~np.isnan(score)]
    cls = np.digitize(v, [2, 2.75, 3.5, 4.25])          # 5 equal-width classes on the 1-5 index scale
    out = {"district": d, "date": date, "bounds": _bounds(tr, usable.shape), **info,
           "ahp_class_pct": {n: round(float((cls == i).mean() * 100), 1)
                             for i, n in enumerate(["very low", "low", "moderate", "high", "very high"])},
           "observed_available": (DYN / "labels" / d / f"{date}.npz").exists(),
           "h2_available": (H2_DIR / f"h2_without_{pd.Timestamp(date).year}.json").exists(),
           "h5_available": _h5_available(d, pd.Timestamp(date).year),
           "h6_available": _h6_available(d, pd.Timestamp(date).year),
           "ml_maps": sorted(p.stem.split("_")[-1] for p in (DYN / "maps").glob(f"{d}_{date}_*.tif")) if (DYN / "maps").exists() else []}
    if out["observed_available"]:
        lab = np.load(DYN / "labels" / d / f"{date}.npz")["label"]
        o = lab != 255
        out["observed_flood_pct"] = round(float((lab[o] == 1).mean() * 100), 2)
        sc = score[o & ~np.isnan(score)]
        y = lab[o & ~np.isnan(score)]
        if 0 < y.mean() < 1:
            from sklearn.metrics import roc_auc_score
            out["ahp_auc_this_scene"] = round(float(roc_auc_score(y, sc)), 3)
            try:
                h2 = _h2_prob(d, date)
                ok = o & ~np.isnan(h2)
                out["h2_auc_this_scene"] = round(float(roc_auc_score(lab[ok], h2[ok])), 3)
                out["h2_mean_prob_pct"] = round(float(np.nanmean(h2[ok]) * 100), 2)
            except HTTPException:
                pass
            try:
                h6 = _h6_map(d, date)
                ok = o & ~np.isnan(h6)
                out["h6_auc_this_scene"] = round(float(roc_auc_score(lab[ok], h6[ok])), 3)
                out["h6_mean_prob_pct"] = round(float(np.nanmean(h6[ok]) * 100), 2)
            except HTTPException:
                pass
            try:
                h5 = _h5_map(d, date)
                ok = o & ~np.isnan(h5)
                out["h5_auc_this_scene"] = round(float(roc_auc_score(lab[ok], h5[ok])), 3)
                out["h5_mean_prob_pct"] = round(float(np.nanmean(h5[ok]) * 100), 2)
            except HTTPException:
                pass
    return out
