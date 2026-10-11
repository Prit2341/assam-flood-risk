"""
Base rainfall model (step 1 of 4) -- Seychelles-architecture port, all districts.

Renamed/generalized from 20_base_rainfall_nagaon.py (Nagaon-only first pass)
after Prit flagged that all 4 districts need a proper model, not just one.
Same architecture and same deliberate adaptations as that first version
(mm/metres unit fix, real ERA5 soil moisture instead of Seychelles' API
proxy, Magnus-formula kx_real) -- see that version's docstring / Chroma
manual-0051 for the full reasoning. This version additionally:

- Takes DISTRICT as sys.argv[1] instead of hardcoding Nagaon -- one script,
  not four diverging copies (the same lesson already learned building
  fetch_icechunk_pressure_assam.py).
- Fixes a real naming inconsistency across data sources: era5_longrecord/,
  dem_fabdem/, and lulc_dynamicworld_latest/summary.csv all spell it
  "Morigaon", but icechunk_pressure/ (and DISTRICTS in
  fetch_icechunk_pressure_assam.py) spell it "Marigaon"/"marigaon". Handled
  via an explicit per-district path-spelling map (PATH_SPELLING), not a
  single district-name string reused across all four sources.
- Fixes a real bug caught by the advisor before this rerun: the cin-NaN
  fillna was gated on `if "cin" in df.columns`, which was False at that
  point (still named convective_inhibition, renamed one line later) --
  the fillna never ran, so cin flowed unfilled into FEATURE_COLS and every
  stable-atmosphere (near-zero CAPE) hour got silently dropped by the
  final dropna(). Moved the fillna to after the rename.
"""
import os
import sys
import glob
import json
import joblib
import numpy as np
import pandas as pd
import rasterio
import lightgbm as lgb
import xgboost as xgb
from scipy import stats, optimize
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DISTRICTS = {
    "Sivasagar": (27.00, 94.75),
    "Marigaon":  (26.25, 92.25),
    "Nagaon":    (26.25, 92.75),
    "Karimganj": (24.50, 92.50),
}
# canonical key -> spelling used by each data source (era5_longrecord,
# dem_fabdem, lulc summary.csv all use "Morigaon"; icechunk_pressure and
# DISTRICTS above use "Marigaon"/"marigaon" -- confirmed by directly
# listing each directory, not assumed consistent).
PATH_SPELLING = {
    "Sivasagar": {"surface": "Sivasagar", "pressure": "sivasagar", "dem": "Sivasagar", "lulc": "Sivasagar"},
    "Marigaon":  {"surface": "Morigaon",  "pressure": "marigaon",  "dem": "Morigaon",  "lulc": "Morigaon"},
    "Nagaon":    {"surface": "Nagaon",    "pressure": "nagaon",    "dem": "Nagaon",    "lulc": "Nagaon"},
    "Karimganj": {"surface": "Karimganj", "pressure": "karimganj", "dem": "Karimganj", "lulc": "Karimganj"},
}

RAIN_THR_MM = 0.1
# HEAVY_THR_MM is no longer a fixed constant -- Seychelles' 10.0mm was
# calibrated for an equatorial archipelago and does not transfer: it gave
# Sivasagar only 10 training exceedances in 24 years (below the GPD leg's
# own MIN_EXCEEDANCES=15 floor), not because heavy rain is rare there but
# because 10mm/hr sits too deep in ERA5's smoothed-hourly tail for that
# cell. Standard EVT practice (Coles 2001) picks the threshold from the
# data -- using a fixed high percentile of rainy-hour tp_mm here, computed
# per-district from the TRAINING split only (avoids test leakage). See
# compute_heavy_threshold().
HEAVY_PERCENTILE = 99.5
GATE_PROBABILITY_CUTOFF = 0.3
TEST_SPLIT_DATE = "2024-01-01"


def compute_heavy_threshold(train_df):
    rainy = train_df.loc[train_df["tp_mm"] > RAIN_THR_MM, "tp_mm"]
    return float(rainy.quantile(HEAVY_PERCENTILE / 100.0))

FEATURE_COLS = [
    "tp_lag1h", "tp_sum3h", "tp_sum6h", "tp_sum12h", "tp_sum24h",
    "cape_lag1", "cin_lag1", "tcwv_lag1", "kx_real_lag1",
    "sp_lag1", "t2m_lag1",
    "q850_lag1", "w500_lag1", "r850_lag1",
    "blh_lag1", "soil_moisture_lag1", "v10_lag1", "u10_lag1", "ssr_lag1",
    "wind_speed_10m",
    "pressure_tend_2h", "cape_tcwv_interaction",
    "vpd_lag1", "wetbulb_lag1",
    "tcwv_trend_6h", "tcwv_max_6h", "kx_max_6h", "cape_trend_6h", "cape_x_kx_lag1",
    "api_7d_lag1h", "api_30d_lag1h",
    "pct_built_up", "pct_tree_cover", "pct_bare_sparse_vegetation",
    "elevation_m", "slope_deg",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month",
]


def _sat_vapor_pressure_kpa(temp_c):
    return 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))


def _dewpoint_from_rh(temp_c, rh_pct):
    a, b = 17.27, 237.3
    rh_pct = np.clip(rh_pct, 1e-3, 100)
    alpha = np.log(rh_pct / 100.0) + (a * temp_c) / (b + temp_c)
    return (b * alpha) / (a - alpha)


def load_surface(district):
    sp = PATH_SPELLING[district]["surface"]
    surface_dir = os.path.join(ROOT, "data", "era5_longrecord", sp)
    files = sorted(glob.glob(os.path.join(surface_dir, f"{sp}_*.csv")))
    if not files:
        raise FileNotFoundError(f"No surface CSVs found in {surface_dir}")
    chunks = [pd.read_csv(f) for f in files]
    df = pd.concat(chunks, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["timestamp"])
    df = df.drop(columns=["timestamp"]).sort_values("datetime").reset_index(drop=True)
    df["tp_mm"] = (df["total_precipitation"].clip(lower=0) * 1000.0)
    full_range = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    missing = full_range.difference(df["datetime"])
    print(f"Surface: {len(df):,} rows, {df['datetime'].min()} -> {df['datetime'].max()}, "
          f"{len(missing)} missing hours out of {len(full_range):,} expected")
    return df


def load_pressure(district):
    pp = PATH_SPELLING[district]["pressure"]
    path = os.path.join(ROOT, "data", "icechunk_pressure", f"{pp}_pressure_5x5.parquet")
    p = pd.read_parquet(path)
    center_lat, center_lon = DISTRICTS[district]
    p = p[(p["latitude"] == center_lat) & (p["longitude"] == center_lon)].copy()
    if len(p) == 0:
        raise ValueError(f"No pressure rows matched centre ({center_lat},{center_lon}) for {district} "
                          f"-- check float match against actual grid values, don't assume.")
    p = p.drop(columns=["latitude", "longitude"]).reset_index(drop=True)
    print(f"Pressure (centre point {center_lat},{center_lon}): {len(p):,} rows, "
          f"{p['datetime'].min()} -> {p['datetime'].max()}")
    return p


def load_terrain(district):
    dp = PATH_SPELLING[district]["dem"]
    dem_path = os.path.join(ROOT, "data", "dem_fabdem", f"{dp}_fabdem_30m.tif")
    with rasterio.open(dem_path) as src:
        arr = src.read(1).astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        land = arr > 0
        elevation_m = float(np.nanmean(arr[land]))
        dy, dx = np.gradient(np.nan_to_num(arr, nan=0.0))
        slope_deg_arr = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2) / 30.0))
        slope_deg = float(np.nanmean(slope_deg_arr[land]))
    return elevation_m, slope_deg


def load_static_landcover(district):
    lp = PATH_SPELLING[district]["lulc"]
    lulc_path = os.path.join(ROOT, "data", "lulc_dynamicworld_latest", "summary.csv")
    lulc = pd.read_csv(lulc_path)
    lulc = lulc[lulc["district"] == lp]
    if len(lulc) == 0:
        raise ValueError(f"No LULC rows found for district='{lp}' in {lulc_path} -- "
                          f"available: {sorted(pd.read_csv(lulc_path)['district'].unique())}")
    wide = lulc.set_index("class")["pct"]
    return {
        "pct_built_up": float(wide.get("built", 0.0)),
        "pct_tree_cover": float(wide.get("trees", 0.0)),
        "pct_bare_sparse_vegetation": float(wide.get("bare", 0.0)),
    }


def load_hourly(district):
    surf = load_surface(district)
    pres = load_pressure(district)
    df = surf.merge(pres, on="datetime", how="inner")
    print(f"Merged surface+pressure: {len(df):,} rows "
          f"({len(surf) - len(df):,} surface rows dropped, no pressure match)")
    return df


def build_features(df, district):
    df = df.copy()
    df = df.rename(columns={
        "convective_available_potential_energy": "cape",
        "convective_inhibition": "cin",
        "total_column_water_vapour": "tcwv",
        "surface_pressure": "sp",
        "temperature_2m": "t2m",
        "dewpoint_temperature_2m": "d2m",
        "boundary_layer_height": "blh",
        "volumetric_soil_water_layer_1": "soil_moisture",
        "v_component_of_wind_10m": "v10",
        "u_component_of_wind_10m": "u10",
        "surface_solar_radiation_downwards": "ssr",
    })
    # ERA5 artifact: cin can be NaN in highly stable atmospheres (near-zero
    # CAPE). Must fill AFTER the rename above (bug caught by advisor review:
    # the old Nagaon-only script gated this on "cin" in df.columns BEFORE
    # the rename, when the column was still named convective_inhibition --
    # the fillna silently never ran, and dropna() below deleted every calm
    # hour as a result).
    df["cin"] = df["cin"].fillna(0.0)

    td850 = _dewpoint_from_rh(df["t850"] - 273.15, df["r850"])
    td700 = _dewpoint_from_rh(df["t700"] - 273.15, df["r700"])
    t850_c, t700_c, t500_c = df["t850"] - 273.15, df["t700"] - 273.15, df["t500"] - 273.15
    df["kx_real"] = (t850_c - t500_c) + td850 - (t700_c - td700)

    df = df.sort_values("datetime").reset_index(drop=True)
    tp = df["tp_mm"]
    df["tp_lag1h"] = tp.shift(1)
    df["tp_sum3h"] = tp.shift(1).rolling(3, min_periods=3).sum()
    df["tp_sum6h"] = tp.shift(1).rolling(6, min_periods=6).sum()
    df["tp_sum12h"] = tp.shift(1).rolling(12, min_periods=12).sum()
    df["tp_sum24h"] = tp.shift(1).rolling(24, min_periods=24).sum()

    atm_cols = ["cape", "cin", "tcwv", "sp", "t2m", "d2m", "blh", "soil_moisture",
                "v10", "u10", "ssr", "kx_real", "q850", "w500", "r850"]
    for col in atm_cols:
        df[f"{col}_lag1"] = df[col].shift(1)

    df["pressure_tend_2h"] = df["sp_lag1"] - df["sp"].shift(2)
    df["cape_tcwv_interaction"] = df["cape_lag1"] * df["tcwv_lag1"]
    df["wind_speed_10m"] = np.sqrt(df["u10"].shift(1)**2 + df["v10"].shift(1)**2)

    es_t2m = _sat_vapor_pressure_kpa(df["t2m_lag1"])
    es_d2m = _sat_vapor_pressure_kpa(df["d2m_lag1"])
    df["vpd_lag1"] = (es_t2m - es_d2m).clip(lower=0)
    rh_pct = (100 * es_d2m / es_t2m).clip(0, 100)

    t = df["t2m_lag1"]
    df["wetbulb_lag1"] = (
        t * np.arctan(0.151977 * np.sqrt(rh_pct + 8.313659))
        + np.arctan(t + rh_pct) - np.arctan(rh_pct - 1.676331)
        + 0.00391838 * rh_pct ** 1.5 * np.arctan(0.023101 * rh_pct)
        - 4.686035
    )

    tcwv_s = df["tcwv"].shift(1)
    kx_s = df["kx_real"].shift(1)
    cape_s = df["cape"].shift(1)
    df["tcwv_trend_6h"] = tcwv_s.diff(5)
    df["tcwv_max_6h"] = tcwv_s.rolling(6, min_periods=3).max()
    df["kx_max_6h"] = kx_s.rolling(6, min_periods=3).max()
    df["cape_trend_6h"] = cape_s.diff(5)
    df["cape_x_kx_lag1"] = df["cape_lag1"] * df["kx_real_lag1"]

    from scipy.signal import lfilter
    def compute_api(series, k):
        return lfilter([1], [1, -k], series.fillna(0))
    df["api_7d_lag1h"] = pd.Series(compute_api(df["tp_mm"], 0.9933), index=df.index).shift(1)
    df["api_30d_lag1h"] = pd.Series(compute_api(df["tp_mm"], 0.9979), index=df.index).shift(1)

    df["hour"] = df["datetime"].dt.hour
    df["doy"] = df["datetime"].dt.dayofyear
    df["month"] = df["datetime"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)

    for k, v in load_static_landcover(district).items():
        df[k] = v
    elevation_m, slope_deg = load_terrain(district)
    df["elevation_m"] = elevation_m
    df["slope_deg"] = slope_deg

    df["y_occurrence"] = (df["tp_mm"] > RAIN_THR_MM).astype(int)
    n_before = len(df)
    df = df.dropna(subset=FEATURE_COLS + ["tp_mm"]).reset_index(drop=True)
    print(f"dropna: {n_before:,} -> {len(df):,} rows ({n_before - len(df):,} dropped, "
          f"{(n_before - len(df)) / n_before:.1%})")
    return df


# --- Heavy-branch calibration legs, ported from Seychelles'
# approach_b_nwp/101_production_ensemble.py fit_csg_gamma/csg_quantiles_gamma
# and fit_gpd_tail/gpd_quantiles (read in full this session). These ONLY
# reshape the heavy-event q10/q90 INTERVAL -- they do not touch q50_final,
# per 111_adaptive_ensemble_26yr.py's actual architecture. Regional GPD
# (the 3rd leg in Seychelles' mean-of-3) is deferred: it needs a
# multi-site regional tp grid (Seychelles pooled ~25 sites via
# icechunk_regional_tp.parquet), which Assam does not have yet -- not
# built silently as a 2-of-3 substitute without saying so.
MIN_SCALE = 0.05
MIN_EXCEEDANCES = 15


def fit_csg_gamma(point_pred, actual):
    mask = actual > 0
    y = actual[mask].clip(min=1e-3)
    pred = np.clip(point_pred[mask], 1e-3, None)

    def neg_log_lik(params):
        shape, lam = params
        if shape <= 0 or lam <= 0:
            return 1e12
        scale = lam * pred + MIN_SCALE
        ll = stats.gamma.logpdf(y, a=shape, scale=scale)
        if not np.all(np.isfinite(ll)):
            return 1e12
        return -np.sum(ll)

    res = optimize.minimize(neg_log_lik, x0=[1.5, 1.0], method="Nelder-Mead", options={"maxiter": 2000})
    shape, lam = res.x
    return float(max(shape, 1e-3)), float(max(lam, 1e-3))


def csg_quantiles_gamma(alpha_low, alpha_high, p_rain, shape, lam, point_pred):
    zero_mass = 1 - p_rain
    pred = np.clip(point_pred, 1e-3, None)
    scale = lam * pred + MIN_SCALE

    def _q(alpha):
        out = np.zeros_like(p_rain)
        needs = alpha > zero_mass
        if np.any(needs):
            adj = np.clip((alpha - zero_mass[needs]) / np.clip(p_rain[needs], 1e-6, None), 1e-6, 1 - 1e-6)
            out[needs] = stats.gamma.ppf(adj, a=shape, scale=scale[needs])
        return out
    return _q(alpha_low), _q(alpha_high)


def fit_gpd_tail(actual_heavy, threshold):
    excess = actual_heavy[actual_heavy > threshold] - threshold
    if len(excess) < MIN_EXCEEDANCES:
        return None
    shape, _, scale = stats.genpareto.fit(excess, floc=0)
    return float(shape), float(scale)


def gpd_quantiles(alpha_low, alpha_high, p_rain, gpd_shape, gpd_scale, threshold):
    zero_mass = 1 - p_rain
    def _q(alpha):
        out = np.zeros_like(p_rain)
        needs = alpha > zero_mass
        if np.any(needs):
            adj = np.clip((alpha - zero_mass[needs]) / np.clip(p_rain[needs], 1e-6, None), 1e-6, 1 - 1e-6)
            out[needs] = threshold + stats.genpareto.ppf(adj, c=gpd_shape, scale=gpd_scale)
        return out
    return _q(alpha_low), _q(alpha_high)


class EnsembleGate:
    def __init__(self, lgbm, xgb_clf):
        self.lgbm = lgbm
        self.xgb_clf = xgb_clf
    def predict_proba(self, X):
        p_lgbm = self.lgbm.predict_proba(X)[:, 1]
        p_xgb = self.xgb_clf.predict_proba(X)[:, 1]
        p_ens = (p_lgbm + p_xgb) / 2.0
        return np.column_stack([1 - p_ens, p_ens])


def fit_quantile_models(train_rain):
    q_models = {}
    for q in [0.1, 0.5, 0.9]:
        reg = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=q,
                                n_estimators=150, max_depth=5, learning_rate=0.05,
                                random_state=42, n_jobs=-1, tree_method="hist", device="cuda")
        reg.fit(train_rain[FEATURE_COLS], train_rain["tp_mm"])
        q_models[q] = reg
    return q_models


def fit_models(df_train, heavy_thr_mm):
    occ_clf = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  min_child_samples=10, verbose=-1, random_state=42)
    occ_clf.fit(df_train[FEATURE_COLS], df_train["y_occurrence"])

    train_rain = df_train[df_train["y_occurrence"] == 1]
    q_models = fit_quantile_models(train_rain)

    train_rain_gate = train_rain.copy()
    train_rain_gate["y_gate"] = (train_rain_gate["tp_mm"] > heavy_thr_mm).astype(int)

    lgbm_gate = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                    verbose=-1, class_weight="balanced", random_state=42)
    lgbm_gate.fit(train_rain_gate[FEATURE_COLS], train_rain_gate["y_gate"])

    n_pos = train_rain_gate["y_gate"].sum()
    xgb_gate = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                  scale_pos_weight=(len(train_rain_gate) - n_pos) / max(1, n_pos),
                                  random_state=42, n_jobs=-1, tree_method="hist", device="cuda")
    xgb_gate.fit(train_rain_gate[FEATURE_COLS], train_rain_gate["y_gate"])

    gate_clf = EnsembleGate(lgbm_gate, xgb_gate)

    train_heavy = train_rain[train_rain["tp_mm"] > heavy_thr_mm]
    ext_reg = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=4,
                                learning_rate=0.05, random_state=42, n_jobs=-1,
                                tree_method="hist", device="cuda")
    if len(train_heavy) > 5:
        ext_reg.fit(train_heavy[FEATURE_COLS], train_heavy["tp_mm"])

    return occ_clf, q_models, gate_clf, ext_reg


def run_split(df_train, df_test, heavy_thr_mm):
    occ_clf, q_models, gate_clf, ext_reg = fit_models(df_train, heavy_thr_mm)

    X_test = df_test[FEATURE_COLS]
    out = df_test[["datetime", "tp_mm"]].copy()

    is_rain_prob = occ_clf.predict_proba(X_test)[:, 1]
    is_rain = is_rain_prob > 0.5
    out["p_rain"] = is_rain_prob
    out["q50_mm"] = np.where(is_rain, q_models[0.5].predict(X_test), 0.0)
    out["q10_mm"] = np.where(is_rain, q_models[0.1].predict(X_test), 0.0)
    out["q90_mm"] = np.where(is_rain, q_models[0.9].predict(X_test), 0.0)

    gate_p = gate_clf.predict_proba(X_test)[:, 1]
    gated = gate_p > GATE_PROBABILITY_CUTOFF
    out["gated"] = gated

    ext_pred = np.zeros(len(X_test))
    if len(df_train[df_train["tp_mm"] > heavy_thr_mm]) > 5:
        ext_pred = ext_reg.predict(X_test)

    # Soft gate blend, not a hard step function. Diagnosed 2026-09-16
    # (TASK.md "CORRECTION" section, Chroma manual-0056): the extreme
    # regressor is trained only on genuinely-heavy rows, so it always
    # outputs a heavy-event-sized value; at longer forecast horizons the
    # gate classifier's false-positive rate rises (marginal gate_p just
    # above GATE_PROBABILITY_CUTOFF), routing more actually-light-rain
    # hours through it -- with a hard np.where() those got the full
    # heavy-sized value regardless, producing the observed bias growing
    # to +100-150% by h=6. Blending by gate_p damps a marginal call
    # instead of fully substituting it, while a confident heavy call
    # (gate_p near 1) still gets close to ext_pred. Standard mixture-of-
    # experts soft gating (Jacobs et al. 1991, "Adaptive Mixtures of
    # Local Experts", Neural Computation), not an untested improvisation.
    gate_weight = np.clip(
        (gate_p - GATE_PROBABILITY_CUTOFF) / (1.0 - GATE_PROBABILITY_CUTOFF), 0.0, 1.0
    )
    blended = gate_weight * ext_pred + (1.0 - gate_weight) * out["q50_mm"].values

    out["q50_final"] = np.where(gated, blended, out["q50_mm"])
    out["q50_final"] = np.where(is_rain, np.clip(out["q50_final"], 0, None), 0.0)
    out["q90_final"] = np.maximum(out["q90_mm"], np.where(gated, blended, out["q90_mm"]))

    # --- Heavy-branch re-blend: CSG-Gamma + Local GPD, mean-of-2 ---
    # Reshapes ONLY the q10/q90 interval on gated (heavy) test rows.
    # q50_final is untouched -- the point forecast doesn't change here,
    # by construction of Seychelles' own architecture.
    train_rain = df_train[df_train["tp_mm"] > RAIN_THR_MM]
    train_point_pred = np.clip(q_models[0.5].predict(train_rain[FEATURE_COLS]), 0, None)
    train_actual = train_rain["tp_mm"].values
    heavy_mask_tr = train_actual > heavy_thr_mm
    light_mask_tr = ~heavy_mask_tr

    shape_l, lam_l = fit_csg_gamma(train_point_pred[light_mask_tr], train_actual[light_mask_tr])
    if heavy_mask_tr.sum() >= 10:
        shape_h, lam_h = fit_csg_gamma(train_point_pred[heavy_mask_tr], train_actual[heavy_mask_tr])
    else:
        shape_h, lam_h = shape_l, lam_l
    local_gpd_fit = fit_gpd_tail(train_actual, heavy_thr_mm)

    p_rain_arr = np.clip(is_rain_prob, 1e-4, 1 - 1e-4)
    point_pred_arr = out["q50_mm"].values
    q10_ens, q90_ens = out["q10_mm"].values.copy(), out["q90_mm"].values.copy()

    if gated.sum() > 0:
        q10_g, q90_g = csg_quantiles_gamma(0.10, 0.90, p_rain_arr[gated], shape_h, lam_h, point_pred_arr[gated])
        if local_gpd_fit:
            q10_lgpd, q90_lgpd = gpd_quantiles(0.10, 0.90, p_rain_arr[gated], local_gpd_fit[0], local_gpd_fit[1], heavy_thr_mm)
        else:
            q10_lgpd, q90_lgpd = q10_g, q90_g
        q10_ens[gated] = (q10_g + q10_lgpd) / 2.0
        q90_ens[gated] = (q90_g + q90_lgpd) / 2.0

    out["q10_ens"] = q10_ens
    out["q90_ens"] = np.maximum(q90_ens, out["q50_final"].values)
    out["local_gpd_shape"] = local_gpd_fit[0] if local_gpd_fit else np.nan
    out["local_gpd_scale"] = local_gpd_fit[1] if local_gpd_fit else np.nan

    return out, (occ_clf, q_models, gate_clf, ext_reg)


def run_district(district):
    print("=" * 70)
    print(f" BASE RAINFALL MODEL -- {district} (t+1h) ")
    print("=" * 70)

    models_dir = os.path.join(ROOT, "models", f"base_rainfall_{district.lower()}")
    os.makedirs(models_dir, exist_ok=True)
    out_preds = os.path.join(ROOT, "data", "rainfall_nowcast", f"base_rainfall_{district.lower()}_predictions.csv")

    raw = load_hourly(district)
    df = build_features(raw, district)
    print(f"Total usable rows: {len(df):,}")
    print(f"Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")

    train = df[df["datetime"] < TEST_SPLIT_DATE].reset_index(drop=True)
    test = df[df["datetime"] >= TEST_SPLIT_DATE].reset_index(drop=True)
    print(f"Train: {len(train):,} rows ({train['datetime'].min()} -> {train['datetime'].max()})")
    print(f"Test:  {len(test):,} rows ({test['datetime'].min()} -> {test['datetime'].max()})")

    heavy_thr_mm = compute_heavy_threshold(train)
    print(f"Heavy threshold (train-period p{HEAVY_PERCENTILE} of rainy-hour tp_mm): "
          f"{heavy_thr_mm:.2f} mm/hr")

    out, models = run_split(train, test, heavy_thr_mm)
    occ_clf, q_models, gate_clf, ext_reg = models
    out.to_csv(out_preds, index=False)

    actual = out["tp_mm"].values
    pred = out["q50_final"].values
    corr_all = np.corrcoef(actual, pred)[0, 1]
    rmse_all = np.sqrt(mean_squared_error(actual, pred))

    rainy = out[out["tp_mm"] > RAIN_THR_MM]
    corr_rainy = np.corrcoef(rainy["tp_mm"], rainy["q50_final"])[0, 1] if len(rainy) > 2 else float("nan")
    mean_bias_rainy = ((rainy["q50_final"].mean() - rainy["tp_mm"].mean()) / rainy["tp_mm"].mean()
                        if len(rainy) else float("nan"))

    heavy = out[out["tp_mm"] > heavy_thr_mm]
    heavy_train = train[train["tp_mm"] > heavy_thr_mm]
    print(f"\n--- {district} Base Model Performance (test >= 2024, t+1h) ---")
    print(f"Overall correlation (all hours): {corr_all:.3f}  RMSE: {rmse_all:.3f} mm")
    print(f"Rainy-hour correlation:          {corr_rainy:.3f}  (mean bias: {mean_bias_rainy:+.1%})")
    print(f"Heavy hours in TRAIN (2000-2023, >{heavy_thr_mm:.2f}mm): {len(heavy_train):,}  "
          f"(this is the population the GPD legs will actually fit on)")
    if not heavy.empty:
        mae_heavy = mean_absolute_error(heavy["tp_mm"], heavy["q50_final"])
        cov_base = np.mean((heavy["tp_mm"] >= heavy["q10_mm"]) & (heavy["tp_mm"] <= heavy["q90_final"]))
        width_base = np.mean(heavy["q90_final"] - heavy["q10_mm"])
        cov_ens = np.mean((heavy["tp_mm"] >= heavy["q10_ens"]) & (heavy["tp_mm"] <= heavy["q90_ens"]))
        width_ens = np.mean(heavy["q90_ens"] - heavy["q10_ens"])
        print(f"Heavy hours in TEST (>{heavy_thr_mm:.2f}mm): n={len(heavy)}  MAE(q50)={mae_heavy:.2f}mm")
        print(f"  Baseline interval  (gate+ext_reg):     80% coverage={cov_base:.1%}  width={width_base:.2f}mm")
        print(f"  CSG-Gamma+LocalGPD interval (new leg): 80% coverage={cov_ens:.1%}  width={width_ens:.2f}mm")
    else:
        print(f"No heavy (>{heavy_thr_mm:.2f}mm) hours in test period.")
        cov_base = width_base = cov_ens = width_ens = float("nan")

    joblib.dump(occ_clf, os.path.join(models_dir, "occurrence_classifier.joblib"))
    joblib.dump(q_models, os.path.join(models_dir, "quantile_models.joblib"))
    joblib.dump(gate_clf, os.path.join(models_dir, "gate_classifier.joblib"))
    joblib.dump(ext_reg, os.path.join(models_dir, "extreme_regressor.joblib"))
    with open(os.path.join(models_dir, "feature_cols.json"), "w") as f:
        json.dump({"feature_cols": FEATURE_COLS, "gate_probability_cutoff": GATE_PROBABILITY_CUTOFF,
                    "heavy_thr_mm": heavy_thr_mm, "heavy_percentile": HEAVY_PERCENTILE,
                    "rain_thr_mm": RAIN_THR_MM}, f, indent=2)

    print(f"Models saved to: {models_dir}")
    print(f"Predictions saved to: {out_preds}")

    return {
        "district": district, "corr_all": corr_all, "rmse_all": rmse_all,
        "corr_rainy": corr_rainy, "bias_rainy": mean_bias_rainy,
        "heavy_thr_mm": heavy_thr_mm,
        "n_heavy_train": len(heavy_train), "n_heavy_test": len(heavy),
        "cov_base": cov_base, "width_base": width_base,
        "cov_ens": cov_ens, "width_ens": width_ens,
    }


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target and target not in DISTRICTS:
        print(f"Unknown district '{target}'. Choices: {list(DISTRICTS)}")
        return
    districts = [target] if target else list(DISTRICTS)

    summary = []
    for d in districts:
        summary.append(run_district(d))
        print()

    if len(summary) > 1:
        print("=" * 70)
        print(" SUMMARY -- all districts ")
        print("=" * 70)
        print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
