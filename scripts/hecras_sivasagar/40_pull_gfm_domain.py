"""
Held-out GFM scenes clipped to the MODEL DOMAIN (pre-registered in Chroma manual-0238, 2026-11-05).
For each pass (all STAC items of one date within 15 min, i.e. the adjacent 25 s slices of one overpass) read
ensemble_flood_extent + exclusion_mask COGs over the domain bbox in the native Equi7 grid (EPSG:27703), mosaic (first valid
wins), set excluded pixels to 255, write one uint8 raster per pass on ONE common 20 m grid (the domain bbox in EPSG:27703).
  python -u scripts/hecras_sivasagar/40_pull_gfm_domain.py
Out: digital_twin/sivasagar/data/gfm_domain/gfm_<YYYYmmddTHHMM>.tif (1 flood, 0 dry, 255 no data/excluded) + passes.csv
"""
import os
from pathlib import Path
import numpy as np, pandas as pd, requests, rasterio, geopandas as gpd
from rasterio.windows import from_bounds
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[2]; D = ROOT / "digital_twin" / "sivasagar" / "data"; OUT = D / "gfm_domain"; OUT.mkdir(exist_ok=True)
bd = gpd.read_file(D / "boundary_utm46n.gpkg")
bb4326 = [float(x) for x in bd.to_crs(4326).total_bounds]
x0, y0, x1, y1 = bd.to_crs(27703).total_bounds
x0, y1 = np.floor(x0 / 20) * 20, np.ceil(y1 / 20) * 20; W = int(np.ceil((x1 - x0) / 20)); H = int(np.ceil((y1 - y0) / 20))
T = from_origin(x0, y1, 20, 20)
WINDOWS = [("2026-07-06T00:00:00Z", "2026-07-22T23:59:59Z"), ("2026-08-09T00:00:00Z", "2026-08-22T23:59:59Z"), ("2026-07-05T23:00:00Z", "2026-07-05T23:59:59Z")]
USED = {"20260717T2337", "20260722T1139", "20260820T1148"}
items = []
for a, b in WINDOWS:
    r = requests.post("https://stac.eodc.eu/api/v1/search", json={"collections": ["GFM"], "bbox": bb4326, "datetime": f"{a}/{b}", "limit": 200}, timeout=120)
    r.raise_for_status(); items += r.json()["features"]
df = pd.DataFrame([dict(id=f["id"], t=pd.Timestamp(f["properties"]["datetime"]), fl=f["assets"]["ensemble_flood_extent"]["href"],
                        ex=f["assets"]["exclusion_mask"]["href"]) for f in items if "ensemble_flood_extent" in f["assets"]]).drop_duplicates("id").sort_values("t")
df["pass"] = (df.t.diff() > pd.Timedelta(minutes=15)).cumsum()
rows = []
for p, g in df.groupby("pass"):
    key = g.t.iloc[0].strftime("%Y%m%dT%H%M")
    if key[:13] in {u[:13] for u in USED} or any(abs((g.t.iloc[0] - pd.Timestamp(u[:8] + " " + u[9:11] + ":" + u[11:13], tz="UTC")).total_seconds()) < 900 for u in USED):
        print("skip used", key); continue
    mos = np.full((H, W), 255, np.uint8)
    for _, it in g.iterrows():
        try:
            with rasterio.open(it.fl) as s:
                win = from_bounds(x0, y1 - H * 20, x0 + W * 20, y1, transform=s.transform)
                f = s.read(1, window=win, boundless=True, fill_value=255, out_shape=(H, W))
            with rasterio.open(it.ex) as s:
                e = s.read(1, window=from_bounds(x0, y1 - H * 20, x0 + W * 20, y1, transform=s.transform), boundless=True, fill_value=1, out_shape=(H, W))
        except Exception as ex:
            print("  read fail", it.id, ex); continue
        f = np.where((e != 0) & (e != 255), 255, f)          # excluded -> no data
        put = (mos == 255) & (f != 255)
        mos[put] = f[put]
    nvalid = int((mos != 255).sum()); nfl = int((mos == 1).sum())
    fn = OUT / f"gfm_{key}.tif"
    with rasterio.open(fn, "w", driver="GTiff", height=H, width=W, count=1, dtype="uint8", crs="EPSG:27703", transform=T, nodata=255, compress="deflate") as o:
        o.write(mos, 1)
    rows.append(dict(pass_time=str(g.t.iloc[0]), key=key, n_items=len(g), valid_km2=round(nvalid * 4e-4, 1), flood_km2=round(nfl * 4e-4, 1), file=fn.name))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / "passes.csv", index=False)
