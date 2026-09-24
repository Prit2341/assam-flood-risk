"""
GFS feasibility test pull (verification only, not production) -- Nagaon.

Confirms that NOAA GFS *forecast* data (a real NWP product with lead time,
not reanalysis) can be read from dynamical.org's Icechunk archive via the
direct STAC route (stac.dynamical.org), which is NOT the data.dynamical.org
convenience layer that sunsets 2026-09-30.

Source (read from dynamical.org's own catalog page, 2026-09-19):
  collection noaa-gfs-forecast, asset "icechunk-https", init_time 2021-05-01
  -> present every 6h, lead_time 0-384h (hourly through 120h), 0.25 deg.

Scope: one point (Nagaon grid center), a few days around the 2022-06-20
Kopili flood, lead_time 0..6h. Goal is structure/variable/units discovery,
not a training dataset.

Run from project root:
    python scripts/pull_gfs_test.py
"""
import os
import time

import pandas as pd
import pystac
import icechunk
import xarray as xr

OUT_DIR = "data/gfs_test"
os.makedirs(OUT_DIR, exist_ok=True)

LAT, LON = 26.25, 92.75            # Nagaon grid center (same as pressure pull)
INIT_START, INIT_END = "2022-06-18", "2022-06-21"
MAX_LEAD_H = 6


def open_gfs_forecast():
    catalog = pystac.Catalog.from_file("https://stac.dynamical.org/catalog.json")
    collection = catalog.get_child("noaa-gfs-forecast")
    asset = collection.assets["icechunk-https"]
    repo = icechunk.Repository.open(icechunk.http_storage(asset.href))
    session = repo.readonly_session("main")
    return xr.open_zarr(session.store, chunks=None)


def main():
    t0 = time.time()
    ds = open_gfs_forecast()
    print(f"Opened in {time.time()-t0:.1f}s")
    print("Dims:", dict(ds.sizes))
    print("Coords:", list(ds.coords))
    print(f"init_time: {ds.init_time.values[0]} -> {ds.init_time.values[-1]}")
    print(f"lead_time: {ds.lead_time.values[0]} -> {ds.lead_time.values[-1]}")
    print(f"lat range: {float(ds.latitude.min())}..{float(ds.latitude.max())}  "
          f"lon range: {float(ds.longitude.min())}..{float(ds.longitude.max())}")
    print("Data variables:")
    for v in ds.data_vars:
        a = ds[v].attrs
        print(f"  {v:<45} {a.get('units','?'):<14} {a.get('long_name', a.get('description',''))}")

    # Precip + a few convective/moisture candidates, whichever exist
    want = [v for v in ds.data_vars if any(k in v for k in
            ("precipitation_surface", "cape", "precipitable_water",
             "relative_humidity_2m", "temperature_2m", "total_cloud_cover"))]
    print("\nSelected for test pull:", want)

    sub = (
        ds[want]
        .sel(latitude=LAT, longitude=LON, method="nearest")
        .sel(init_time=slice(INIT_START, INIT_END),
             lead_time=slice("0h", f"{MAX_LEAD_H}h"))
    )
    t1 = time.time()
    df = sub.compute().to_dataframe().reset_index()
    print(f"\nPulled {len(df)} rows in {time.time()-t1:.1f}s")
    out = os.path.join(OUT_DIR, "nagaon_gfs_test_20220618_21.csv")
    df.to_csv(out, index=False)
    print("Saved:", out)
    print(df.head(12).to_string())
    print("\nNull % per column:")
    print((100 * df.isna().mean()).round(1).to_string())


if __name__ == "__main__":
    main()
