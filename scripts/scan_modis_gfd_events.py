"""
Scans the MODIS Global Flood Database (Tellman et al. 2021, GEE collection
GLOBAL_FLOOD_DB/MODIS_EVENTS/V1, 2000-2018) for every event that actually
overlaps each of the 4 scope districts, to find pre-made flood-extent
coverage for atlas years (1998-2023) beyond the ones already known
(2022 GFM/NRSC, 2004 Morigaon MODIS).

For each matching event: prints event id, begin/end date, and the flooded
fraction of the district's polygon (so near-zero-overlap events, e.g. floods
elsewhere in India that only clip the AOI bbox, are filtered out).

Run:
    python scripts/scan_modis_gfd_events.py
"""
import json

import ee

ee.Initialize(project="assamflood-508209")

with open("data/boundaries/scope4_districts.geojson", encoding="utf-8") as f:
    districts_fc = json.load(f)

gfd = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1")

import datetime

for feat in districts_fc["features"]:
    name = feat["properties"].get("Dist") or feat["properties"].get("dtname")
    geom = ee.Geometry(feat["geometry"])
    area_m2 = geom.area(1).getInfo()

    coll = gfd.filterBounds(geom)
    n = coll.size().getInfo()
    print(f"\n=== {name}: {n} candidate MODIS GFD events intersecting bbox ===", flush=True)

    info_list = coll.select(["flooded"]).toList(n).getInfo()
    for item in info_list:
        eid = item["properties"]["system:index"]
        b = item["properties"]["system:time_start"]
        e = item["properties"]["system:time_end"]
        bd = datetime.datetime.utcfromtimestamp(b / 1000).date()
        ed = datetime.datetime.utcfromtimestamp(e / 1000).date()
        img = ee.Image(gfd.filter(ee.Filter.eq("system:index", eid)).first())
        flooded = img.select("flooded").selfMask()
        try:
            flood_area_val = flooded.multiply(ee.Image.pixelArea()).reduceRegion(
                reducer=ee.Reducer.sum(), geometry=geom, scale=250, maxPixels=1e10, bestEffort=True
            ).get("flooded").getInfo() or 0
        except Exception as ex:
            print(f"  event {eid}: {bd} to {ed} -- ERROR {ex}", flush=True)
            continue
        pct = 100.0 * flood_area_val / area_m2
        flag = "  <-- REAL COVERAGE" if pct > 1.0 else ""
        print(f"  event {eid}: {bd} to {ed}, flooded {pct:.2f}% of district{flag}", flush=True)
