"""
0-6 hour precipitation nowcasting pilot (v0) -- our own version of the
Seychelles project's ML approach, but literature-anchored to a specific
method rather than an untraceable 4-way ensemble: XGBoost regression on
lagged ERA5 reanalysis features, following the SCR-XGBoost design
(segmented classification + regression for 4-6h leadtime correction,
J. Meteorological Research/Springer) and the general ERA5+gradient-
boosting rainfall literature.

Pilot scope: ONE district (Nagaon/Kampur -- our best-verified reach so
far) and ONE lead-time set (t+1 .. t+6 hours), to see how the approach
performs before deciding whether to extend to all 4 districts or add the
classification stage / other components.

Target: total_precipitation (ERA5 hourly accumulated precip, metres) at
t+1..t+6. Features: current-hour values + lags (t, t-1, t-2, t-3) of the
physically relevant moisture/instability/dynamics variables.
"""
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

DISTRICT = "Nagaon"
DATA_DIR = f"data/era5_longrecord/{DISTRICT}"
LEAD_HOURS = [1, 2, 3, 4, 5, 6]
LAGS = [0, 1, 2, 3]  # 0 = current hour

FEATURE_VARS = [
    "total_precipitation",
    "convective_available_potential_energy",
    "convective_inhibition",
    "total_column_water_vapour",
    "boundary_layer_height",
    "dewpoint_temperature_2m",
    "temperature_2m",
    "surface_pressure",
    "mean_sea_level_pressure",
    "total_cloud_cover",
    "low_cloud_cover",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
]


def load_district(data_dir):
    files = sorted(glob.glob(f"{data_dir}/*.csv"))
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    # ERA5 hourly should have no gaps at hourly freq; verify, don't assume
    full_range = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    missing = full_range.difference(df["timestamp"])
    print(f"Loaded {len(df)} rows, {df['timestamp'].min()} -> {df['timestamp'].max()}, "
          f"{len(missing)} missing hours out of {len(full_range)} expected")
    return df


def build_features(df):
    feat = pd.DataFrame(index=df.index)
    for var in FEATURE_VARS:
        for lag in LAGS:
            feat[f"{var}_lag{lag}"] = df[var].shift(lag)
    feat["hour_of_day"] = df["timestamp"].dt.hour
    feat["day_of_year"] = df["timestamp"].dt.dayofyear
    feat["timestamp"] = df["timestamp"]

    targets = pd.DataFrame(index=df.index)
    for h in LEAD_HOURS:
        targets[f"precip_t+{h}"] = df["total_precipitation"].shift(-h)

    combined = pd.concat([feat, targets], axis=1).dropna()
    return combined


def main():
    df = load_district(DATA_DIR)
    data = build_features(df)

    # time-based split: train on everything before 2024, test on 2024 onward
    # (out-of-sample, and covers real events we haven't trained on)
    train = data[data["timestamp"] < "2024-01-01"]
    test = data[data["timestamp"] >= "2024-01-01"]
    print(f"Train: {len(train)} rows ({train['timestamp'].min()} -> {train['timestamp'].max()})")
    print(f"Test:  {len(test)} rows ({test['timestamp'].min()} -> {test['timestamp'].max()})")

    feature_cols = [c for c in data.columns if c not in
                    ["timestamp"] + [f"precip_t+{h}" for h in LEAD_HOURS]]

    results = []
    for h in LEAD_HOURS:
        target_col = f"precip_t+{h}"
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
        model.fit(train[feature_cols], train[target_col])
        pred = model.predict(test[feature_cols])
        pred = np.clip(pred, 0, None)  # precipitation can't be negative

        actual = test[target_col].values
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        corr = np.corrcoef(actual, pred)[0, 1]
        # skill vs. naive persistence (predict = current-hour precip)
        persistence_pred = test["total_precipitation_lag0"].values
        persistence_rmse = np.sqrt(mean_squared_error(actual, persistence_pred))

        results.append({
            "lead_hour": h, "rmse_m": round(rmse, 6), "mae_m": round(mae, 6),
            "corr": round(corr, 3), "persistence_rmse_m": round(persistence_rmse, 6),
            "skill_vs_persistence_pct": round(100 * (1 - rmse / persistence_rmse), 1),
        })
        print(f"t+{h}h: RMSE={rmse:.6f}m MAE={mae:.6f}m corr={corr:.3f} "
              f"(persistence RMSE={persistence_rmse:.6f}m, "
              f"skill={100*(1-rmse/persistence_rmse):.1f}%)")

    res_df = pd.DataFrame(results)
    res_df.to_csv("data/rainfall_nowcast/rainfall_nowcast_pilot_results.csv", index=False)
    print("\nSaved: data/rainfall_nowcast/rainfall_nowcast_pilot_results.csv")
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()
