"""
Does GFS forecast rain carry skill against ERA5 rain (the model's target) at
leads 1-6h, and does it add information beyond the existing Nagaon model?

Usage (project root):
    python scripts/gfs_vs_era5_check.py data/gfs_test/nagaon_gfs_{start}_{end}.parquet

Conventions (checked against 20_base_rainfall.py / 30_forecast_horizons.py):
  * ERA5 target = total_precipitation (m) * 1000 -> mm/h, `timestamp` column.
  * Model row `datetime`=t with lag-1 features; for lead h the row's tp_mm is
    the actual at t+(h-1). So issue time = t-1h and valid time V = issue + h.
  * GFS precipitation_surface is a rate (kg m-2 s-1) -> *3600 = mm/h. Whether
    its hourly value is stamped at the start or end of the averaging hour vs
    ERA5 is NOT assumed: correlation is computed at valid-time offsets
    -1,0,+1h and the best offset is reported, not hard-coded.
Two alignments for the GFS init used at each issue time:
  * ideal     : latest init <= issue time (no delivery lag)
  * lag5h     : latest init <= issue time - 5h (assumed operational delay;
                the 5h figure is an ASSUMPTION, not verified in this project)
Part B (value beyond the model) uses data/rainfall_nowcast/
forecast_horizons_nagaon.csv (test period rows only), fits on the earlier
half of the GFS window and scores on the later half.
"""
import glob
import sys

import numpy as np
import pandas as pd

ERA5_DIR = "data/era5_longrecord/Nagaon"
MODEL_CSV = "data/rainfall_nowcast/forecast_horizons_nagaon.csv"


def load_era5(t0, t1):
    files = sorted(glob.glob(f"{ERA5_DIR}/Nagaon_*.csv"))
    d = pd.concat([pd.read_csv(f, usecols=["timestamp", "total_precipitation"]) for f in files])
    d["datetime"] = pd.to_datetime(d["timestamp"])
    d["era5_mm"] = d["total_precipitation"].clip(lower=0) * 1000.0
    d = d.set_index("datetime")["era5_mm"].sort_index()
    return d[(d.index >= t0) & (d.index <= t1)]


def gfs_at(gfs, issue, h, lag):
    """GFS rain (mm/h) valid at issue+h, from the latest init <= issue-lag."""
    init = (issue - pd.Timedelta(hours=lag)).dt.floor("6h")
    lead = ((issue + pd.Timedelta(hours=h)) - init) / pd.Timedelta(hours=1)
    key = pd.MultiIndex.from_arrays([init, lead.astype(int)])
    return gfs.reindex(key).values, lead.values


def corr(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return np.corrcoef(a[m], b[m])[0, 1], int(m.sum())


def main():
    path = sys.argv[1]
    g = pd.read_parquet(path)
    g["gfs_mm"] = g["precipitation_surface"] * 3600.0
    gfs = g.set_index(["init_time", "lead_h"])["gfs_mm"]
    t0, t1 = g["init_time"].min(), g["init_time"].max() + pd.Timedelta(hours=17)
    era5 = load_era5(t0, t1)
    print(f"GFS inits {g['init_time'].min()} -> {g['init_time'].max()}, "
          f"ERA5 hours {len(era5):,}")

    # ---- Part A: direct skill vs ERA5, ideal alignment, hourly valid times
    print("\nPART A - corr(GFS forecast rain, ERA5 rain), ideal alignment (no delivery lag)")
    print("valid-time offset applied to ERA5 index: -1h / 0h / +1h")
    issue_all = pd.Series(era5.index)
    issue_all = issue_all[(issue_all >= t0 + pd.Timedelta(hours=1))]
    rows = []
    for h in range(1, 7):
        row = {"lead": h}
        for off in (-1, 0, 1):
            issue = issue_all.reset_index(drop=True)
            f, _ = gfs_at(gfs, issue, h, 0)
            v = era5.reindex(issue + pd.Timedelta(hours=h + off)).values
            row[f"off{off:+d}"], row["n"] = corr(f, v)
        rows.append(row)
    a = pd.DataFrame(rows)
    print(a.round(3).to_string(index=False))
    best = int(a[["off-1", "off+0", "off+1"]].iloc[0].astype(float).values.argmax()) - 1
    print(f"Best offset at lead 1: {best:+d}h (used for Part B)")

    # ---- Part B: value beyond existing model
    m = pd.read_csv(MODEL_CSV, parse_dates=["datetime"])
    print("\nPART B - does GFS add skill beyond the model? (in-window: fit on first half, score on second)")
    out = []
    for lag in (0, 5):
        for h in range(1, 7):
            d = m[m["lead_hour"] == h].copy()
            issue = d["datetime"] - pd.Timedelta(hours=1)
            f, lead = gfs_at(gfs, issue.reset_index(drop=True), h, lag)
            d["gfs"] = f
            d["actual"] = era5.reindex(d["datetime"] + pd.Timedelta(hours=h - 1)).values
            d = d.dropna(subset=["gfs", "actual", "q50_final"])
            d = d[(d["datetime"] >= g["init_time"].min() + pd.Timedelta(hours=lag + 1))]
            if len(d) < 500:
                continue
            cut = d["datetime"].quantile(0.5)
            tr, te = d[d["datetime"] < cut], d[d["datetime"] >= cut]
            X = lambda z: np.c_[np.ones(len(z)), z["q50_final"], z["gfs"]]
            X0 = lambda z: np.c_[np.ones(len(z)), z["q50_final"]]
            b1 = np.linalg.lstsq(X(tr), tr["actual"], rcond=None)[0]
            b0 = np.linalg.lstsq(X0(tr), tr["actual"], rcond=None)[0]
            out.append({"lag_h": lag, "lead": h, "n_test": len(te),
                        "corr_model": np.corrcoef(te["q50_final"], te["actual"])[0, 1],
                        "corr_gfs_only": np.corrcoef(te["gfs"], te["actual"])[0, 1],
                        "corr_model+gfs": np.corrcoef(X(te) @ b1, te["actual"])[0, 1],
                        "corr_model_refit": np.corrcoef(X0(te) @ b0, te["actual"])[0, 1]})
    print(pd.DataFrame(out).round(3).to_string(index=False))
    print("\nCaveats: linear 2-3 parameter combo, single district, in-window split; "
          "indicative only, NOT a retrained model result.")


if __name__ == "__main__":
    main()
