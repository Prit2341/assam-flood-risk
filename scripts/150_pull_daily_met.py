"""
Daily gridded meteorology for the dynamic susceptibility model, 2016-01-01 .. latest, all 4 districts,
on each product's native 0.1 deg grid (cell centres inside the district bbox + 0.1 deg margin).

  rain_mm  GPM IMERG V07 Final/Late half-hourly precipitation (mm/h) summed to UTC days x 0.5 h
           (Huffman et al., IMERG V07 ATBD, NASA GSFC 2023). Satellite-gauge product, preferred over
           reanalysis rain for observed accumulations.
  swvl1    ERA5-Land daily mean volumetric soil water layer 1, 0-7 cm, m3/m3
           (Munoz-Sabater et al. 2021, ESSD 13:4349; GEE ECMWF/ERA5_LAND/DAILY_AGGR).
Pulled with Earth Engine getRegion in yearly chunks.

Out: data/dynamic_susceptibility/met/<District>_daily_met.parquet (date, lat, lon, rain_mm, swvl1)
"""
import os
import sys

import ee
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dyn_common as dc

OUT = os.path.join(dc.DYN, "met")
ee.Initialize(project="assamflood-508209")


def points(d):
    minx, miny, maxx, maxy = dc.district_polygon(d).bounds
    lats = np.arange(np.floor((miny - 0.1) * 10) / 10 + 0.05, maxy + 0.1, 0.1)
    lons = np.arange(np.floor((minx - 0.1) * 10) / 10 + 0.05, maxx + 0.1, 0.1)
    return [(round(la, 3), round(lo, 3)) for la in lats for lo in lons]


def imerg_daily(start, end):
    col = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").select("precipitation")
    n = ee.Date(end).difference(ee.Date(start), "day")
    days = ee.List.sequence(0, n.subtract(1))

    def one(k):
        d0 = ee.Date(start).advance(k, "day")
        return (col.filterDate(d0, d0.advance(1, "day")).sum().multiply(0.5)
                .rename("rain_mm").set("system:time_start", d0.millis()))
    return ee.ImageCollection(days.map(one))


def region_df(col, pts, scale):
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([lo, la]), {"la": la, "lo": lo}) for la, lo in pts])
    rows = col.getRegion(fc.geometry(), scale).getInfo()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df["date"] = pd.to_datetime(df["time"], unit="ms").dt.normalize()
    return df


def snap(df, pts):
    P = np.array(pts)
    i = np.argmin((df["latitude"].values[:, None] - P[:, 0]) ** 2 + (df["longitude"].values[:, None] - P[:, 1]) ** 2, axis=1)
    df["lat"], df["lon"] = P[i, 0], P[i, 1]
    return df


def run(d):
    os.makedirs(OUT, exist_ok=True)
    pts = points(d)
    print(f"{d}: {len(pts)} points", flush=True)
    frames = []
    last = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=3)
    for y in range(2016, last.year + 1):
        for m0 in (1, 7):
            s = pd.Timestamp(y, m0, 1)
            e = min(s + pd.DateOffset(months=6), last)
            if s >= e:
                continue
            r = snap(region_df(imerg_daily(str(s.date()), str(e.date())), pts, 11000), pts)
            sm = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(str(s.date()), str(e.date())) \
                .select("volumetric_soil_water_layer_1")
            m = snap(region_df(sm, pts, 11000), pts)
            a = r.groupby(["date", "lat", "lon"])["rain_mm"].mean()
            b = m.groupby(["date", "lat", "lon"])["volumetric_soil_water_layer_1"].mean().rename("swvl1")
            f = pd.concat([a, b], axis=1).reset_index()
            frames.append(f)
            print(f"  {s.date()}..{e.date()}: {len(f)} rows, rain mean {f['rain_mm'].mean():.2f}", flush=True)
    out = pd.concat(frames).sort_values(["date", "lat", "lon"])
    out.to_parquet(os.path.join(OUT, f"{d}_daily_met.parquet"), index=False)
    print(f"{d}: saved {len(out)} rows {out['date'].min().date()}..{out['date'].max().date()}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "all"
    for d in (dc.DISTRICTS if a == "all" else [a]):
        run(d)
