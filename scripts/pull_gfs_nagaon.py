"""
Pull GFS forecast precipitation + precipitable water for Nagaon over a date
range, all 6-hourly inits, lead 0..MAX_LEAD_H, spatial mean over a 3x3
(0.25 deg) box around the Nagaon grid center (~0.75 deg, roughly district
scale, closer to ERA5's district-mean than a single cell).

Source: dynamical.org noaa-gfs-forecast via stac.dynamical.org (see
scripts/pull_gfs_test.py and Chroma manual-0064).

Usage (project root):
    python -u scripts/pull_gfs_nagaon.py 2024-07-01 2024-07-31   # timing test
    python -u scripts/pull_gfs_nagaon.py 2024-01-01 2026-03-31   # full window
Output: data/gfs_test/nagaon_gfs_{start}_{end}.parquet
"""
import os
import sys
import time

import pystac
import icechunk
import xarray as xr

OUT_DIR = "data/gfs_test"
os.makedirs(OUT_DIR, exist_ok=True)

LAT, LON = 26.25, 92.75
HALF = 0.25                      # 3x3 box at 0.25 deg
MAX_LEAD_H = 17                  # 6 + up to 5h delivery lag + up to 6h init spacing
VARS = ["precipitation_surface", "precipitable_water_atmosphere"]


def open_gfs_forecast():
    catalog = pystac.Catalog.from_file("https://stac.dynamical.org/catalog.json")
    asset = catalog.get_child("noaa-gfs-forecast").assets["icechunk-https"]
    repo = icechunk.Repository.open(icechunk.http_storage(asset.href))
    return xr.open_zarr(repo.readonly_session("main").store, chunks=None)


def main():
    start, end = sys.argv[1], sys.argv[2]
    out = os.path.join(OUT_DIR, f"nagaon_gfs_{start}_{end}.parquet")
    ds = open_gfs_forecast()
    sub = (
        ds[VARS]
        # GFS latitude is DESCENDING (90 -> -90): slice must go north -> south
        .sel(latitude=slice(LAT + HALF, LAT - HALF), longitude=slice(LON - HALF, LON + HALF))
        .sel(init_time=slice(start, f"{end}T23:59"), lead_time=slice("0h", f"{MAX_LEAD_H}h"))
    )
    print(f"Selection: {dict(sub.sizes)}", flush=True)
    assert sub.sizes["latitude"] == 3 and sub.sizes["longitude"] == 3, "expected a 3x3 box"
    t0 = time.time()
    sub = sub.astype("float32").compute()
    print(f"Computed in {time.time()-t0:.1f}s", flush=True)
    # spatial mean over the 3x3 box
    m = sub.mean(dim=["latitude", "longitude"])
    df = m.to_dataframe().reset_index()
    df = df[["init_time", "lead_time"] + VARS]
    df["lead_h"] = (df["lead_time"] / __import__("pandas").Timedelta(hours=1)).astype(int)
    df = df.drop(columns="lead_time")
    df.to_parquet(out, index=False)
    print(f"Saved {out}: {len(df):,} rows, {os.path.getsize(out)/1e6:.2f} MB, "
          f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
