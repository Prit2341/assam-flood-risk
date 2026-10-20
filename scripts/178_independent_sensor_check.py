"""
Independent-physics check of H5 over-prediction (is C-band SAR missing water under crops?).
Step `coverage`: for the 10 largest H5 over-prediction scenes (diag/extent_error_scenes.csv), list
  - ALOS-2 PALSAR-2 ScanSAR (L-band, JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR) images within +-7 days and district coverage;
  - Sentinel-2 SR (COPERNICUS/S2_SR_HARMONIZED) images within +-3 days and the cloud-free share of the district
    (S2_CLOUD_PROBABILITY < 40 and SCL not cloud/shadow/cirrus/saturated).

Step `compare`: on each scene's GFM-observed cropland pixels, three groups:
  OVER = H5 p >= 0.3 and GFM dry (the over-prediction), DRY = H5 p < 0.02 and GFM dry (control), WET = GFM wet.
  PALSAR-2: HH gamma0 change (dB, gamma0 = 10 log10(DN^2) - 83, GEE catalog) from a dry-season baseline
  (Jan-Mar median of the same year) to the image nearest the scene with >= 90% district cover. Flooded crops can show a
  rise (double bounce) or a drop (open water), so both are reported; +-3 dB is a descriptive cut, medians are cut-free.
  Sentinel-2 (scenes with >= 40% clear district): MNDWI (Xu 2006, B3/B11) on clear pixels, share > 0 and median.
  If OVER looks like WET rather than DRY, C-band SAR (GFM) likely missed water; if OVER looks like DRY it did not.

Step `control` (paddy-water confound): the same OVER/DRY/WET points on a PALSAR-2 image (>= 90% cover) within +-1 day of
  a NON-flood GFM scene (observed flooded < 1%, obs_frac >= 0.2, > 10 days from the flood scene) in the same year's Jun-Sep (else the same months of any
  other year). Same OVER HH drop on the non-flood date -> seasonal paddy water, consistently labelled dry, so the
  over-prediction is real; drop only on the flood date -> flood water GFM did not flag (label noise).

Run:  python -u scripts/178_independent_sensor_check.py coverage | compare | control
Out:  data/dynamic_susceptibility/diag/independent_sensor_{coverage,compare}.csv
"""
import os
import sys

import ee
import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAG = os.path.join(ROOT, "data", "dynamic_susceptibility", "diag")
ee.Initialize(project="assamflood-508209")
NAME = {"Marigaon": "Morigaon"}


def districts():
    g = gpd.read_file(os.path.join(ROOT, "data", "boundaries", "scope4_districts.geojson")).to_crs(4326)
    return {NAME.get(n, n): ee.Geometry(geom.__geo_interface__) for n, geom in zip(g.dtname, g.geometry)}


def worst(n=10):
    S = pd.read_csv(os.path.join(DIAG, "extent_error_scenes.csv"), parse_dates=["date"])
    act = S[(S.obs_pct >= 2) | (S.pred_pct >= 2)]
    return act.assign(excess=act.pred_pct - act.obs_pct).nlargest(n, "excess")


def frac(img, geom, scale=200):
    """Share of the district polygon where img (0/1) is 1, with unmasked pixels counted as 0."""
    v = img.unmask(0).rename("v").reduceRegion(ee.Reducer.mean(), geom, scale, maxPixels=1e9).get("v")
    return ee.Number(v).getInfo()


def run_coverage():
    G = districts()
    rows = []
    for _, r in worst().iterrows():
        d, t = r["district"], pd.Timestamp(r["date"])
        geom = G[d]
        ps = ee.ImageCollection("JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR").filterBounds(geom) \
            .filterDate((t - pd.Timedelta(days=7)).strftime("%Y-%m-%d"), (t + pd.Timedelta(days=8)).strftime("%Y-%m-%d"))
        for im in ps.toList(50).getInfo():
            img = ee.Image(im["id"])
            day = pd.Timestamp(im["properties"]["system:time_start"], unit="ms")
            rows.append({"district": d, "scene_date": t.date(), "obs_pct": r.obs_pct, "h5_pct": r.pred_pct, "sensor": "PALSAR-2 L-band",
                         "image_date": day.date(), "lag_days": (day.normalize() - t).days,
                         "district_cover": round(frac(img.select("HH").mask().gt(0), geom), 3), "id": im["id"]})
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geom) \
            .filterDate((t - pd.Timedelta(days=3)).strftime("%Y-%m-%d"), (t + pd.Timedelta(days=4)).strftime("%Y-%m-%d"))
        cp = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        for im in s2.toList(50).getInfo():
            img = ee.Image(im["id"])
            prob = ee.Image(cp.filter(ee.Filter.eq("system:index", im["properties"]["system:index"])).first()).select("probability")
            scl = img.select("SCL")
            clear = prob.lt(40).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(1))
            day = pd.Timestamp(im["properties"]["system:time_start"], unit="ms")
            rows.append({"district": d, "scene_date": t.date(), "obs_pct": r.obs_pct, "h5_pct": r.pred_pct, "sensor": "Sentinel-2",
                         "image_date": day.date(), "lag_days": (day.normalize() - t).days,
                         "district_cover": round(frac(clear, geom), 3), "id": im["id"]})
        print(f"{d} {t.date()} done", flush=True)
    C = pd.DataFrame(rows)
    C.to_csv(os.path.join(DIAG, "independent_sensor_coverage.csv"), index=False)
    best = C.groupby(["district", "scene_date", "sensor"]).district_cover.max().unstack()
    print(C.drop(columns="id").to_string(index=False), "\n\nbest district cover per scene:\n", best)


def gamma0(img, band):
    return img.select(band).toFloat().pow(2).log10().multiply(10).subtract(83)


def group_points(d, date, n=1500, seed=0):
    sys.path.insert(0, os.path.join(ROOT, "dashboard-app", "backend"))
    import susceptibility as sus
    S, usable, tr = sus._static(d)
    lab = np.load(sus.DYN / "labels" / d / f"{date}.npz")["label"]
    p = sus._h5_map(d, date)
    crop = (S["lulc"] == 4) & (lab != 255) & ~np.isnan(p)
    groups = {"OVER": crop & (p >= 0.3) & (lab == 0), "DRY": crop & (p < 0.02) & (lab == 0), "WET": crop & (lab == 1)}
    rng = np.random.default_rng(seed)
    feats = []
    for g, m in groups.items():
        rr, cc = np.nonzero(m)
        k = rng.choice(len(rr), min(n, len(rr)), replace=False) if len(rr) else []
        for i in k:
            x, y = tr * (cc[i] + 0.5, rr[i] + 0.5)
            feats.append(ee.Feature(ee.Geometry.Point([x, y]), {"g": g, "p": float(p[rr[i], cc[i]])}))
    return ee.FeatureCollection(feats), {g: int(m.sum()) for g, m in groups.items()}


def run_compare():
    C = pd.read_csv(os.path.join(DIAG, "independent_sensor_coverage.csv"), parse_dates=["scene_date"])
    rows = []
    for (d, t), c in C.groupby(["district", "scene_date"]):
        date = t.strftime("%Y-%m-%d")
        fc, sizes = group_points(d, date)
        out = []
        ps = c[(c.sensor == "PALSAR-2 L-band") & (c.district_cover >= 0.9)].assign(a=lambda x: x.lag_days.abs()).sort_values("a")
        if len(ps):
            im = ee.Image(ps.iloc[0]["id"])
            base = ee.ImageCollection("JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR").filterDate(f"{t.year}-01-01", f"{t.year}-04-01")                 .filterBounds(fc.geometry()).map(lambda i: gamma0(i, "HH").rename("b")).median()
            dhh = gamma0(im, "HH").subtract(base).rename("dHH")
            smp = dhh.addBands(gamma0(im, "HH").rename("HH")).sampleRegions(fc, scale=25, geometries=False).getInfo()["features"]
            df = pd.DataFrame([f["properties"] for f in smp]).dropna()
            for g, x in df.groupby("g"):
                out.append({"sensor": "PALSAR-2", "image": ps.iloc[0]["image_date"], "lag": ps.iloc[0]["lag_days"], "group": g, "n": len(x),
                            "median_dHH_dB": x.dHH.median(), "share_drop3dB": (x.dHH <= -3).mean(),
                            "share_rise3dB": (x.dHH >= 3).mean(), "median_HH_dB": x.HH.median()})
        s2 = c[(c.sensor == "Sentinel-2") & (c.district_cover >= 0.4)].sort_values("district_cover", ascending=False)
        if len(s2):
            im = ee.Image(s2.iloc[0]["id"])
            prob = ee.Image(ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filter(
                ee.Filter.eq("system:index", im.get("system:index"))).first()).select("probability")
            scl = im.select("SCL")
            clear = prob.lt(40).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(1))
            mndwi = im.normalizedDifference(["B3", "B11"]).rename("mndwi").updateMask(clear)
            smp = mndwi.sampleRegions(fc, scale=20, geometries=False).getInfo()["features"]
            df = pd.DataFrame([f["properties"] for f in smp]).dropna()
            for g, x in df.groupby("g"):
                out.append({"sensor": "Sentinel-2", "image": s2.iloc[0]["image_date"], "lag": s2.iloc[0]["lag_days"], "group": g, "n": len(x),
                            "median_mndwi": x.mndwi.median(), "share_mndwi_gt0": (x.mndwi > 0).mean()})
        for o in out:
            rows.append({"district": d, "scene_date": date, "pixels_" + o["group"]: sizes[o["group"]], **o})
        print(f"{d} {date} done {sizes}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(DIAG, "independent_sensor_compare.csv"), index=False)
    cols = ["district", "scene_date", "sensor", "image", "lag", "group", "n", "median_dHH_dB", "share_drop3dB",
            "share_rise3dB", "median_HH_dB", "median_mndwi", "share_mndwi_gt0"]
    print(R[[c for c in cols if c in R]].round(3).to_string(index=False))


def nonflood_palsar(d, t, geom):
    idx = pd.read_csv(os.path.join(ROOT, "data", "dynamic_susceptibility", "labels", d, "labels_index.csv"), parse_dates=["date"])
    idx = idx[(idx.obs_frac >= 0.2) & (idx.flood_frac_obs < 0.01) & idx.date.dt.month.between(6, 9) & ((idx.date - t).abs() > pd.Timedelta(days=10))]
    idx = idx.assign(same=(idx.date.dt.year == t.year).astype(int), gap=(idx.date - t).abs()).sort_values(["same", "gap"], ascending=[False, True])
    col = ee.ImageCollection("JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR").filterBounds(geom)
    for nd in idx.date:
        for im in col.filterDate((nd - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), (nd + pd.Timedelta(days=2)).strftime("%Y-%m-%d")).toList(10).getInfo():
            if frac(ee.Image(im["id"]).select("HH").mask().gt(0), geom) >= 0.9:
                return nd, im
    return None, None


def run_control():
    G = districts()
    C = pd.read_csv(os.path.join(DIAG, "independent_sensor_coverage.csv"), parse_dates=["scene_date"])
    R0 = pd.read_csv(os.path.join(DIAG, "independent_sensor_compare.csv"))
    rows = []
    for (d, t), c in C.groupby(["district", "scene_date"]):
        date = t.strftime("%Y-%m-%d")
        fc, sizes = group_points(d, date)
        nd, im = nonflood_palsar(d, t, G[d])
        if im is None:
            print(f"{d} {date}: no non-flood PALSAR-2 date found", flush=True)
            continue
        day = pd.Timestamp(im["properties"]["system:time_start"], unit="ms").date()
        img = ee.Image(im["id"])
        base = ee.ImageCollection("JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR").filterDate(f"{t.year}-01-01", f"{t.year}-04-01")             .filterBounds(fc.geometry()).map(lambda i: gamma0(i, "HH").rename("b")).median()
        smp = gamma0(img, "HH").subtract(base).rename("dHH").sampleRegions(fc, scale=25, geometries=False).getInfo()["features"]
        df = pd.DataFrame([f["properties"] for f in smp]).dropna()
        for g, x in df.groupby("g"):
            f0 = R0[(R0.district == d) & (R0.scene_date == date) & (R0.sensor == "PALSAR-2") & (R0.group == g)]
            rows.append({"district": d, "flood_scene": date, "group": g,
                         "flood_date_image": f0.image.iloc[0] if len(f0) else None,
                         "flood_median_dHH": f0.median_dHH_dB.iloc[0] if len(f0) else np.nan,
                         "flood_share_drop3dB": f0.share_drop3dB.iloc[0] if len(f0) else np.nan,
                         "nonflood_gfm_date": nd.date(), "nonflood_image": day, "n": len(x),
                         "nonflood_median_dHH": x.dHH.median(), "nonflood_share_drop3dB": (x.dHH <= -3).mean()})
        print(f"{d} {date}: control GFM {nd.date()} / PALSAR {day}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(DIAG, "independent_sensor_control.csv"), index=False)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    {"coverage": run_coverage, "compare": run_compare, "control": run_control}[sys.argv[1]]()
