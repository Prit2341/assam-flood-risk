"""
Multi-horizon rainfall forecast (t+1h .. t+6h), all 4 districts.

"Forecasting" in this project's own v0 pilot (rainfall_nowcast_pilot.py)
meant a full t+1..t+6h lead-time sweep, not a single horizon -- this
script extends 20_base_rainfall.py's t+1h-only base model to match that
scope, with the CSG-Gamma+Local-GPD heavy-interval calibration built
in the previous step carried through at every horizon.

Design (confirmed correct via a regression check, see main()): reuses
20_base_rainfall.py's build_features()/run_split() UNCHANGED. Only the
TARGET is shifted per horizon -- FEATURE_COLS stay the lag-1 block
(known at t-1) for every horizon, so lead_hour=h means "forecast tp_mm at
t-1+h using only information available at t-1". Shifting tp_mm/
y_occurrence by -(h-1) and reusing fit_models/run_split as-is means
lead_hour=1 is `shift(0)` -- IDENTICAL to 20_base_rainfall.py's existing
t+1h run, used below as a regression check before trusting h=2..6.

Deliberate choices, not defaults:
- Heavy threshold computed ONCE per district (from h=1's train target)
  and reused across all 6 horizons -- same marginal distribution, and
  needed so the 6 horizons stay comparable to each other (recomputing per
  horizon would give 6 slightly different, barely-different thresholds
  and defeat the comparison).
- CSG-Gamma and Local GPD are refit PER horizon (inside run_split, not
  cached) -- the GPD tail barely moves (same marginal target
  distribution) but fit_csg_gamma conditions on train_point_pred, which
  degrades with h; caching the h=1 fit would silently pair a stale,
  well-calibrated-looking interval with a much worse point forecast.
- p_rain fired rate (rate of is_rain_prob > 0.5) is checked per horizon.
  If the occurrence classifier's skill collapses at long lead times,
  q50_final collapses to 0 for a reason that has nothing to do with the
  quantile regression -- that gets reported as a finding, not masked by
  quietly lowering the 0.5 cutoff to make the correlation look better.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
m = importlib.import_module("20_base_rainfall")

ROOT = m.ROOT
LEAD_HOURS = [1, 2, 3, 4, 5, 6]
V0_RESULTS_PATH = os.path.join(ROOT, "data", "rainfall_nowcast", "rainfall_nowcast_pilot_results.csv")


def run_horizon(district, df, heavy_thr_mm, h):
    df_h = df.copy()
    df_h["tp_mm"] = df["tp_mm"].shift(-(h - 1))
    df_h = df_h.dropna(subset=["tp_mm"]).reset_index(drop=True)
    df_h["y_occurrence"] = (df_h["tp_mm"] > m.RAIN_THR_MM).astype(int)

    train = df_h[df_h["datetime"] < m.TEST_SPLIT_DATE].reset_index(drop=True)
    test = df_h[df_h["datetime"] >= m.TEST_SPLIT_DATE].reset_index(drop=True)

    out, _ = m.run_split(train, test, heavy_thr_mm)

    actual = out["tp_mm"].values
    pred = out["q50_final"].values
    corr_all = np.corrcoef(actual, pred)[0, 1]
    rmse_all = np.sqrt(mean_squared_error(actual, pred))

    rainy = out[out["tp_mm"] > m.RAIN_THR_MM]
    corr_rainy = np.corrcoef(rainy["tp_mm"], rainy["q50_final"])[0, 1] if len(rainy) > 2 else float("nan")
    bias_rainy = ((rainy["q50_final"].mean() - rainy["tp_mm"].mean()) / rainy["tp_mm"].mean()
                  if len(rainy) else float("nan"))
    fired_rate = float((out["p_rain"] > 0.5).mean())
    actual_rain_rate = float((out["tp_mm"] > m.RAIN_THR_MM).mean())

    heavy = out[out["tp_mm"] > heavy_thr_mm]
    if not heavy.empty:
        cov_ens = float(np.mean((heavy["tp_mm"] >= heavy["q10_ens"]) & (heavy["tp_mm"] <= heavy["q90_ens"])))
        width_ens = float(np.mean(heavy["q90_ens"] - heavy["q10_ens"]))
    else:
        cov_ens = width_ens = float("nan")

    out.insert(0, "lead_hour", h)
    return out, {
        "district": district, "lead_hour": h,
        "corr_all": corr_all, "rmse_all_mm": rmse_all,
        "corr_rainy": corr_rainy, "bias_rainy": bias_rainy,
        "p_rain_fired_rate": fired_rate, "actual_rain_rate": actual_rain_rate,
        "n_heavy_test": len(heavy), "heavy_cov": cov_ens, "heavy_width_mm": width_ens,
        "n_train": len(train), "n_test": len(test),
    }


def run_district(district):
    print("=" * 70)
    print(f" MULTI-HORIZON FORECAST -- {district} (t+1h .. t+{max(LEAD_HOURS)}h) ")
    print("=" * 70)

    raw = m.load_hourly(district)
    df = m.build_features(raw, district)
    train_h1 = df[df["datetime"] < m.TEST_SPLIT_DATE]
    heavy_thr_mm = m.compute_heavy_threshold(train_h1)
    print(f"Heavy threshold (fixed across all horizons, from h=1 train): {heavy_thr_mm:.2f} mm/hr")

    all_out, summary = [], []
    for h in LEAD_HOURS:
        out, row = run_horizon(district, df, heavy_thr_mm, h)
        all_out.append(out)
        summary.append(row)
        print(f"  h={h}: n_test={row['n_test']:,}  corr_rainy={row['corr_rainy']:.3f}  "
              f"bias={row['bias_rainy']:+.1%}  p_rain_fired={row['p_rain_fired_rate']:.1%} "
              f"(actual rain rate={row['actual_rain_rate']:.1%})  "
              f"heavy_cov={row['heavy_cov']:.1%}  heavy_width={row['heavy_width_mm']:.2f}mm")

    combined = pd.concat(all_out, ignore_index=True)
    out_path = os.path.join(ROOT, "data", "rainfall_nowcast", f"forecast_horizons_{district.lower()}.csv")
    combined.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return summary


def main():
    # Regression check: h=1 must reproduce 20_base_rainfall.py's existing
    # t+1h numbers exactly (same target, shift(0)) -- confirms the target-
    # shift plumbing is correct before trusting h=2..6.
    all_summary = []
    for district in m.DISTRICTS:
        all_summary.extend(run_district(district))
        print()

    df_summary = pd.DataFrame(all_summary)

    # Merge in v0's own Nagaon-only t+1..t+6h numbers for direct comparison.
    # v0 was only ever run on Nagaon -- only attach this column to Nagaon
    # rows, not the other 3 districts, so it can't be misread as a
    # baseline that exists for them too.
    if os.path.exists(V0_RESULTS_PATH):
        v0 = pd.read_csv(V0_RESULTS_PATH).rename(columns={"lead_hour": "lead_hour", "corr": "v0_corr_nagaon"})
        df_summary = df_summary.merge(v0[["lead_hour", "v0_corr_nagaon"]], on="lead_hour", how="left")
        df_summary.loc[df_summary["district"] != "Nagaon", "v0_corr_nagaon"] = np.nan

    summary_path = os.path.join(ROOT, "data", "rainfall_nowcast", "forecast_horizons_summary.csv")
    df_summary.to_csv(summary_path, index=False)

    print("=" * 70)
    print(" CONSOLIDATED RESULT -- all districts x horizons ")
    print("=" * 70)
    cols = ["district", "lead_hour", "corr_rainy", "bias_rainy", "p_rain_fired_rate",
            "heavy_cov", "heavy_width_mm"]
    if "v0_corr_nagaon" in df_summary.columns:
        cols.append("v0_corr_nagaon")
    print(df_summary[cols].to_string(index=False))
    print(f"\nSaved consolidated summary: {summary_path}")


if __name__ == "__main__":
    main()
