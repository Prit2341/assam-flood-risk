"""
Fetch 3D pressure-level ERA5 for a 5x5 regional grid around each of the 4
scope districts, from the same public AWS Icechunk ERA5 store Seychelles
uses (s3://earthmover-icechunk-era5/, group="pressure/temporal", anonymous
read, no CDS registration/queue needed).

Reuses Seychelles' exact FETCH_PLAN (same variables/levels) for direct
comparability -- see D:\\BISAG-N\\Seychelles\\3D_approach\\
01_fetch_icechunk_3d_regional.py, which this is adapted from.

CONFIRMED LIVE (2026-09-16, not assumed): store variables
['pv','q','r','t','u','v','w','z'], 13 pressure levels (1000..50 hPa),
global lat/lon, valid_time range 1940-01-01 to 2026-03-31.

REAL GAP, stated plainly: the store's real max date is 2026-03-31 --
this does NOT cover the Sivasagar 2026-07/08 Dikhow flood event. That
event's PINN leg has no pressure-level input from this source; a
different source (CDS reanalysis-era5-pressure-levels, or a later
extension of this store) would be needed to cover it, same open problem
as Seychelles' own 122_extend_icechunk_pressure_2026.py had to solve.

GRID GEOMETRY (open decision, not yet resolved to basin-level -- this
script uses PER-DISTRICT centroids as the default, matching the scope
unit used everywhere else in this project so far. Re-evaluate against
a per-basin geometry, per TASK.md's still-open flag, before treating
this as final):
  Sivasagar centroid  26.9702, 94.6593 -> grid center (27.00, 94.75)
  Marigaon  centroid  26.2807, 92.2785 -> grid center (26.25, 92.25)
  Nagaon    centroid  26.3623, 92.7536 -> grid center (26.25, 92.75)
  Karimganj centroid  24.5795, 92.3791 -> grid center (24.50, 92.50)
  Each grid: 5x5 points on the native 0.25-degree ERA5 grid, +/-0.5 deg
  around the center (matches Seychelles' 5x5 regional-grid convention).

Run from project root:
    python scripts/fetch_icechunk_pressure_assam.py
"""
import os
import sys
import time

import pandas as pd
import xarray as xr
import icechunk

OUT_DIR = "data/icechunk_pressure"
os.makedirs(OUT_DIR, exist_ok=True)

# District centroids, rounded to the nearest 0.25-degree ERA5 grid point
DISTRICTS = {
    "Sivasagar": (27.00, 94.75),
    "Marigaon":  (26.25, 92.25),
    "Nagaon":    (26.25, 92.75),
    "Karimganj": (24.50, 92.50),
}
GRID_HALF_WIDTH = 0.5  # +/- 0.5 deg -> 5 points at 0.25 deg spacing

START = "2000-01-01"
END = "2026-03-31T23:00:00"  # real store max, confirmed live 2026-09-16

# Same FETCH_PLAN as Seychelles' 01_fetch_icechunk_3d_regional.py
FETCH_PLAN = {
    200: ["u", "v", "z"],
    500: ["w", "z", "t", "pv"],
    700: ["t", "r"],
    850: ["u", "v", "t", "q", "r", "z"],
}


def open_pressure_store():
    storage = icechunk.s3_storage(
        bucket="earthmover-icechunk-era5",
        prefix="icechunkV2",
        region="us-east-1",
        anonymous=True,
    )
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    return xr.open_zarr(session.store, group="pressure/temporal", consolidated=False)


def fetch_district_grid(ds, district, center_lat, center_lon):
    """
    One single-shot compute() call for the district's full time range --
    matches Seychelles' PROVEN, actually-measured pattern exactly
    (3D_approach/01_fetch_icechunk_3d_regional.py). Their real log
    (3D_approach/logs/01_icechunk_3d_20260630_165033.log) shows this
    takes ~15 min total for one district's full 26-year 5x5/4-level/
    7-variable pull. Resumability is at district granularity only
    (main() skips a district whose output parquet already exists).

    RAM optimization, 2026-09-16 (applied for districts fetched AFTER
    Sivasagar -- Sivasagar's already-in-flight compute() call used the
    unoptimized version and wasn't restarted, to avoid losing the sunk
    ~5min lazy-selection cost):
    1. Select only the EXACT (variable, level) pairs FETCH_PLAN actually
       needs, not all 7 variables x all 4 levels. The earlier version did
       `ds[all_vars].sel(pressure_level=all_levels)` which pulls ~32
       variable-level chunks when only ~15 are used -- this both roughly
       halves the bytes downloaded/decompressed AND halves peak RAM held
       during assembly, by merging separate per-(var,level) DataArrays
       (each .sel()'d down to its needed level, dropping the extra 3
       unused levels for that variable) into one flat Dataset before
       compute(), instead of computing the full 4-level cube per variable
       first and discarding 3/4 of it afterward.
    2. Downcast to float32 (ERA5's native precision -- xarray/zarr may
       otherwise carry float64 through the pipeline, 2x the RAM for no
       accuracy gain) and use xarray's built-in .to_dataframe() (a single
       vectorized C-level unstack) instead of the manual
       itertools.product() + per-column .reshape() loop, which builds a
       transient list of ~5.7M raw Python tuples just for the index.
    """
    lat_north = center_lat + GRID_HALF_WIDTH
    lat_south = center_lat - GRID_HALF_WIDTH
    lon_west = center_lon - GRID_HALF_WIDTH
    lon_east = center_lon + GRID_HALF_WIDTH

    print(f"  Region: lat [{lat_south}, {lat_north}] lon [{lon_west}, {lon_east}]")
    print(f"  Time: {START} -> {END}", flush=True)

    t0 = time.time()
    spatial_time_sel = dict(
        latitude=slice(lat_north, lat_south),   # ERA5 lat is descending
        longitude=slice(lon_west, lon_east),
        valid_time=slice(START, END),
    )

    # Build one DataArray per exact (variable, level) pair actually needed --
    # no over-fetch of unused levels.
    var_arrays = {}
    for lev, varlist in FETCH_PLAN.items():
        for var in varlist:
            col_name = f"{var}{lev}"
            da = (
                ds[var]
                .sel(pressure_level=lev, method="nearest")
                .sel(**spatial_time_sel)
                .astype("float32")
            )
            var_arrays[col_name] = da.drop_vars("pressure_level", errors="ignore")

    ds_region = xr.Dataset(var_arrays)
    # Drop static lat/lon-only coordinate fields (sdor/lsm/slor/z_sfc --
    # standard deviation of orography, land-sea mask, sub-gridscale
    # orography slope, surface geopotential) that ride along as
    # non-dimension coords on the source store and otherwise leak into
    # every row via .to_dataframe() below, even though they were never
    # requested in FETCH_PLAN. Real, correct values -- just unrequested
    # and wastefully repeated per timestamp; found in Marigaon's first
    # optimized run, 2026-09-16 (5,752,200-row file had 19 data columns
    # instead of the intended 15).
    extra_coords = [c for c in ("sdor", "lsm", "slor", "z_sfc") if c in ds_region.coords]
    if extra_coords:
        ds_region = ds_region.drop_vars(extra_coords)
    print(f"  Lazy selection built in {time.time()-t0:.1f}s, computing (this downloads the data)...", flush=True)

    t1 = time.time()
    ds_computed = ds_region.compute()
    print(f"  Computed in {time.time()-t1:.1f}s  shape={dict(ds_computed.sizes)}", flush=True)

    t2 = time.time()
    idx = ds_computed.to_dataframe().reset_index()
    idx = idx.rename(columns={"valid_time": "datetime"})
    print(f"  Assembled DataFrame in {time.time()-t2:.1f}s  shape={idx.shape}")
    print(f"  Total fetch+assemble: {(time.time()-t0)/60:.1f} min", flush=True)
    return idx


def main():
    # Run ONE district per process invocation (pass district name as argv[1]),
    # or all not-yet-done districts if no arg given. One-process-per-district
    # is preferred: CPython doesn't return freed memory to the OS between
    # loop iterations, so a single long-running process accumulates RSS
    # across districts even though only one district's data is logically
    # "live" at a time. A fresh process per district resets that for real.
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target and target not in DISTRICTS:
        print(f"Unknown district '{target}'. Choices: {list(DISTRICTS)}")
        return

    print("=" * 70)
    print("Icechunk ERA5 pressure-level fetch -- 5x5 grid per district")
    print(f"Store: s3://earthmover-icechunk-era5/ group=pressure/temporal")
    print(f"Time: {START} -> {END}")
    print("=" * 70, flush=True)

    ds = open_pressure_store()
    print(f"Store variables: {sorted(ds.data_vars)}")
    print(f"Store levels: {ds.pressure_level.values.tolist()}")
    store_max = pd.Timestamp(ds.valid_time.max().values)
    print(f"Store max valid_time: {store_max}")
    if store_max < pd.Timestamp("2026-07-01"):
        print("WARNING: store does not reach the 2026-07/08 Sivasagar event -- "
              "expected gap, stated per this script's docstring, not a bug.")
    sys.stdout.flush()

    districts = {target: DISTRICTS[target]} if target else DISTRICTS

    for district, (lat, lon) in districts.items():
        out_path = os.path.join(OUT_DIR, f"{district.lower()}_pressure_5x5.parquet")
        print(f"\n--- {district}: center ({lat}, {lon}) ---", flush=True)
        if os.path.exists(out_path):
            print(f"  already exists at {out_path}, skipping")
            continue

        df = fetch_district_grid(ds, district, lat, lon)

        df.to_parquet(out_path, index=False)
        sz = os.path.getsize(out_path) / 1e6
        print(f"  Saved: {out_path}  ({df.shape[0]:,} rows, {sz:.1f} MB)")

        centre = df[(df["latitude"] == lat) & (df["longitude"] == lon)]
        print(f"  Centre-point null rates ({len(centre):,} rows):")
        for col in centre.columns:
            if col not in ("datetime", "latitude", "longitude"):
                pct = 100 * centre[col].isna().mean()
                print(f"    {col:<8} {pct:.1f}%")
        sys.stdout.flush()

    print("\nDone.")


if __name__ == "__main__":
    main()
