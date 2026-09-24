"""
Extracts the current 4-district scope (Morigaon, Sivasagar, Nagaon,
Karimganj) from the state-wide data/boundaries/assam_districts.geojson
into its own file, the same way barak_valley_districts.geojson was
extracted for the earlier (superseded) 3-district scope.

This closes a real gap: every district-level pull so far (ERA5, GFM,
Dynamic World) has loaded the district polygon on the fly from the
state-wide file each time - there was no single saved boundary file for
this scope, unlike the old Barak Valley scope which had one. Downstream
work (DEM clipping, HydroBASINS delineation, SoilGrids pull) needs one.

Run:`
    python scripts/extract_scope_districts.py
"""
import json
import os

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(BASE_DIR, "data", "boundaries", "assam_districts.geojson")
OUT = os.path.join(BASE_DIR, "data", "boundaries", "scope4_districts.geojson")

SCOPE_DISTRICTS = {"morigaon", "marigaon", "sivasagar", "nagaon", "karimganj"}


def main():
    with open(SRC, encoding="utf-8") as f:
        gj = json.load(f)

    matched = [
        feat for feat in gj["features"]
        if feat["properties"].get("dtname", "").lower() in SCOPE_DISTRICTS
    ]
    names = sorted(f["properties"].get("dtname") for f in matched)
    print(f"Matched {len(matched)}/4 districts: {names}")
    if len(matched) != 4:
        raise SystemExit("Expected exactly 4 districts - aborting, check dtname spelling.")

    out_gj = {"type": "FeatureCollection", "features": matched}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out_gj, f)

    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
