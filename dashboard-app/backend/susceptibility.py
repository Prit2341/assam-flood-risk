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
        cm = matplotlib.colormaps[cmap]
        x = np.clip((arr - vmin) / (vmax - vmin + 1e-12), 0, 1)
        rgba[ok] = (np.array(cm(x[ok])) * 255).astype(np.uint8)
        rgba[ok, 3] = 210
    b = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(b, "PNG")
    return b.getvalue()


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
    if name == "ahp_dynamic":
        a, _ = _ahp_map(d, date)
        return Response(_png(a, "YlOrRd", 1, 5), media_type="image/png")
    m = DYN / "maps" / f"{d}_{date}_{name}.tif"
    if m.exists():
        with rasterio.open(m) as s:
            a = s.read(1).astype(float)
            a[a == s.nodata] = np.nan
        return Response(_png(a, "YlOrRd", 0, np.nanpercentile(a, 99)), media_type="image/png")
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
    return out
