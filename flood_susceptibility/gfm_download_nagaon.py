"""
Download GFM ensemble_flood_extent + exclusion_mask tiles for Nagaon district
covering the June 2024 Assam flood event.

Uses STAC at stac.eodc.eu (same approach as Sri Lanka project).
Downloads the peak-flood date (Jun 17-19 2024) plus a few surrounding dates
for robustness. Files verified with rasterio pixel read after download.
"""
import os
from datetime import datetime, timedelta
from pathlib import Path

import rasterio
import requests
from pystac_client import Client

STAC_URL    = "https://stac.eodc.eu/api/v1"
NAGAON_BBOX = [92.0, 26.0, 93.7, 27.2]   # generous bbox covering Nagaon district
OUT_DIR     = Path(__file__).parent / "gfm_nagaon"
OUT_DIR.mkdir(exist_ok=True)

# June 2024 Assam flood — download all available dates in the event window
EVENT_START = datetime(2024, 6, 10)
EVENT_END   = datetime(2024, 6, 30)

ASSETS_WANTED = ("ensemble_flood_extent", "exclusion_mask")


def download_verify(url: str, dest: Path) -> tuple[bool, str]:
    if dest.exists():
        try:
            with rasterio.open(dest) as src:
                src.read(1)
            return True, "already ok"
        except Exception:
            dest.unlink()

    for attempt in range(3):
        try:
            r = requests.get(url, timeout=120, stream=True)
            r.raise_for_status()
            dest.write_bytes(r.content)
            with rasterio.open(dest) as src:
                src.read(1)
            return True, f"{len(r.content):,} bytes"
        except Exception as e:
            if dest.exists():
                dest.unlink()
            last_err = str(e)
            print(f"    attempt {attempt+1} failed: {last_err[:80]}")
    return False, last_err


catalog = Client.open(STAC_URL)

window_start = (EVENT_START - timedelta(days=3)).strftime("%Y-%m-%dT00:00:00Z")
window_end   = (EVENT_END   + timedelta(days=3)).strftime("%Y-%m-%dT23:59:59Z")

print(f"Searching GFM STAC: {window_start} to {window_end}  bbox={NAGAON_BBOX}")
search = catalog.search(
    collections=["GFM"],
    bbox=NAGAON_BBOX,
    datetime=f"{window_start}/{window_end}",
)
items = list(search.items())
dates = sorted(set(i.datetime.strftime("%Y-%m-%d") for i in items))
print(f"Found {len(items)} items across {len(dates)} dates: {dates}\n")

ok_total = fail_total = 0
for date_str in dates:
    date_items = [i for i in items if i.datetime.strftime("%Y-%m-%d") == date_str]
    date_dir = OUT_DIR / date_str
    date_dir.mkdir(exist_ok=True)
    print(f"[{date_str}] {len(date_items)} tiles")

    for item in date_items:
        for asset_key in ASSETS_WANTED:
            asset = item.assets.get(asset_key)
            if not asset:
                continue
            fname = os.path.basename(asset.href)
            dest  = date_dir / fname
            ok, msg = download_verify(asset.href, dest)
            status = "OK" if ok else "FAIL"
            print(f"  {status:4s}  {fname}  ({msg})")
            if ok:
                ok_total += 1
            else:
                fail_total += 1

print(f"\nDone. {ok_total} files ok, {fail_total} failed.")
print(f"Output: {OUT_DIR}")
