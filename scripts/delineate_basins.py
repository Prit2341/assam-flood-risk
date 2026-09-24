"""
Basin delineation for the 4 current-scope districts (Morigaon, Sivasagar,
Nagaon, Karimganj) using HydroSHEDS/HydroBASINS Asia tile (data/hydrobasins/).
See TASK.md "[ ] Basin delineation via HydroSHEDS/HydroBASINS" -- this
resolves that item.

HydroBASINS level 6 (~sub-basin scale, ~1,000-10,000 km2 typical) is used as
the primary delineation level (standard choice for regional flood studies in
HydroBASINS-based literature); level 4 (major basin) is also clipped for
the province of the river-system membership already recorded in TASK.md
(e.g. confirming which districts fall on Brahmaputra vs Barak systems).
"""
import os

import geopandas as gpd

DISTRICTS_PATH = "data/boundaries/scope4_districts.geojson"
HYDROBASINS_DIR = "data/hydrobasins"
OUT_DIR = "data/basin_delineation"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    districts = gpd.read_file(DISTRICTS_PATH)

    for level in [4, 6]:
        basins_path = os.path.join(HYDROBASINS_DIR, f"hybas_as_lev{level:02d}_v1c.shp")
        basins = gpd.read_file(basins_path)
        if basins.crs != districts.crs:
            basins = basins.to_crs(districts.crs)

        joined = gpd.overlay(districts[["dtname", "geometry"]], basins, how="intersection")
        out_path = os.path.join(OUT_DIR, f"scope4_districts_hybas_lev{level:02d}.geojson")
        joined.to_file(out_path, driver="GeoJSON")

        print(f"\n=== HydroBASINS level {level} ===")
        for dist in districts["dtname"]:
            sub = joined[joined["dtname"] == dist]
            basin_ids = sub["HYBAS_ID"].unique().tolist() if "HYBAS_ID" in sub.columns else []
            print(f"  {dist}: intersects {len(basin_ids)} level-{level} basin(s): {basin_ids}")
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
