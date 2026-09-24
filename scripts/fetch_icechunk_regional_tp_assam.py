"""
Fetch hourly ERA5 total precipitation on the same 5x5 (0.25 deg) grid used by
fetch_icechunk_pressure_assam.py, from the public AWS Icechunk ERA5 store's
`single/temporal` group. This is the raw data for the Regional GPD leg
(Regional Frequency Analysis / index-flood, Hosking & Wallis) of the Seychelles
4-member assembler -- adapted from Seychelles' 3D_approach/97_fetch_regional_tp.py.

Output: data/icechunk_regional_tp/{district}_regional_tp.parquet
        columns: datetime, latitude, longitude, tp_mm   (25 points/hour)

CAVEAT (kept, not hidden): neighbouring 0.25 deg ERA5 cells (~28 km) are highly
spatially correlated, so 25 "sites" are far fewer than 25 independent samples;
RFA assumes intersite independence (Hosking & Wallis 1997). Same limitation as
the Seychelles original.

Run one district per process (memory is not returned between loop iterations):
    python -u scripts/fetch_icechunk_regional_tp_assam.py Nagaon
"""
import os
import sys
import time

import icechunk
import pandas as pd
import xarray as xr

OUT_DIR = "data/icechunk_regional_tp"
os.makedirs(OUT_DIR, exist_ok=True)

DISTRICTS = {
    "Sivasagar": (27.00, 94.75),
    "Marigaon": (26.25, 92.25),
    "Nagaon": (26.25, 92.75),
    "Karimganj": (24.50, 92.50),
}
HALF = 0.5
START = "2000-01-01"
END = "2026-03-31T23:00:00"      # real store max (confirmed 2026-09-16)


def main():
    district = sys.argv[1]
    lat, lon = DISTRICTS[district]
    out = os.path.join(OUT_DIR, f"{district.lower()}_regional_tp.parquet")
    if os.path.exists(out):
        print("exists, skipping:", out)
        return
    storage = icechunk.s3_storage(bucket="earthmover-icechunk-era5", prefix="icechunkV2",
                                  region="us-east-1", anonymous=True)
    session = icechunk.Repository.open(storage).readonly_session("main")
    ds = xr.open_zarr(session.store, group="single/temporal", consolidated=False)
    print("tp attrs:", dict(ds["tp"].attrs))
    print("valid_time:", ds.valid_time.values[0], "->", ds.valid_time.values[-1])
    print("lat order:", float(ds.latitude[0]), "->", float(ds.latitude[-1]), flush=True)

    t0 = time.time()
    sub = ds[["tp"]].sel(latitude=slice(lat + HALF, lat - HALF),          # ERA5 lat descending
                         longitude=slice(lon - HALF, lon + HALF),
                         valid_time=slice(START, END))
    assert sub.sizes["latitude"] == 5 and sub.sizes["longitude"] == 5, dict(sub.sizes)
    print("selection:", dict(sub.sizes), flush=True)
    comp = sub.astype("float32").compute()
    print(f"computed in {(time.time()-t0)/60:.1f} min", flush=True)
    df = comp.to_dataframe().reset_index().rename(columns={"valid_time": "datetime"})
    df["tp_mm"] = df["tp"].clip(lower=0) * 1000.0        # metres -> mm
    df = df[["datetime", "latitude", "longitude", "tp_mm"]]
    df.to_parquet(out, index=False)
    print(f"saved {out}: {len(df):,} rows, {os.path.getsize(out)/1e6:.1f} MB")
    c = df[(df.latitude == lat) & (df.longitude == lon)]
    print("centre point: n=%d nulls=%d mean=%.4f mm/h max=%.2f" %
          (len(c), c.tp_mm.isna().sum(), c.tp_mm.mean(), c.tp_mm.max()))


if __name__ == "__main__":
    main()
