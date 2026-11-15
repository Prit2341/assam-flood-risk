"""
Run INSIDE a GRASS session (grass84.bat -c EPSG:32646 <location> --exec python scripts/193_grass_check.py).
Sri Lanka method cross-check of the Whitebox basins (scripts/191, FABDEM): r.watershed -s threshold 500 on the same FABDEM UTM 46N
terrain, outlet = max accumulation cell inside a 500 m box around each CWC gauge, r.water.outlet, area from r.stats -a.
Out: data/basin_delineation/grass_check/{grass_basin_<i>.tif, grass_summary.json}
"""
import json
import os

import grass.script as gs

ROOT = r"D:\BISAG-N\India\Assam"
AREA = os.environ.get("AREA", "sivasagar")
D = os.path.join(ROOT, "data", "basin_delineation", "grass_check" if AREA == "sivasagar" else "grass_check_" + AREA)
DEM = os.path.join(ROOT, "data", "basin_delineation", AREA + "_fabdem", "dem_utm46.tif")
gauges = json.load(open(os.path.join(D, "gauges_utm46.json")))

gs.run_command("r.in.gdal", input=DEM, output="dem", flags="o", overwrite=True)
gs.run_command("g.region", raster="dem")
gs.run_command("r.watershed", elevation="dem", threshold=500, accumulation="accum", drainage="drain", flags="s", overwrite=True)
print("r.watershed done", flush=True)
res = []
for i, g in enumerate(gauges):
    half = 500
    gs.run_command("g.region", n=g["y"] + half, s=g["y"] - half, e=g["x"] + half, w=g["x"] - half, align="dem")
    best = (-1, None, None)
    for line in gs.read_command("r.stats", input="accum", flags="1gn").splitlines():
        x, y, v = line.split()
        v = abs(float(v))
        if v > best[0]:
            best = (v, float(x), float(y))
    gs.run_command("g.region", raster="dem")
    acc, sx, sy = best
    gs.run_command("r.water.outlet", input="drain", output=f"basin_{i}", coordinates=f"{sx},{sy}", overwrite=True)
    area = sum(float(l.split()[1]) for l in gs.read_command("r.stats", input=f"basin_{i}", flags="an").splitlines() if l.strip()) / 1e6
    gs.run_command("r.out.gdal", input=f"basin_{i}", output=os.path.join(D, f"grass_basin_{i}.tif"), format="GTiff", overwrite=True)
    res.append(dict(gauge=g["name"], snap_x=sx, snap_y=sy, snap_dist_m=round(((sx - g["x"]) ** 2 + (sy - g["y"]) ** 2) ** 0.5), accum_cells=acc, accum_km2=round(acc * 900 / 1e6, 1), basin_km2=round(area, 1)))
    print(res[-1], flush=True)
json.dump(res, open(os.path.join(D, "grass_summary.json"), "w"), indent=1)
