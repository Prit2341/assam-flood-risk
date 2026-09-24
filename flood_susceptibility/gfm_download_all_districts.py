"""
Download GFM ensemble_flood_extent + exclusion_mask tiles for all 4 Assam
districts across the full GFM archive (2016 - present).

Same approach as Sri Lanka project (stac.eodc.eu STAC, sequential downloads,
rasterio pixel-read verification). Output structure:
  gfm_all/<district>/<YYYY-MM-DD>/<tile_files>
"""
import os
from datetime import datetime
from pathlib import Path

import rasterio
import requests
from pystac_client import Client

STAC_URL = "https://stac.eodc.eu/api/v1"

DISTRICTS = {
    "nagaon":    [92.3, 25.9, 93.4, 26.8],
    "karimganj": [92.1, 24.1, 92.7, 25.0],
    "morigaon":  [91.9, 25.9, 92.7, 26.6],
    "sivasagar": [94.3, 26.6, 95.0, 27.4],
}

ASSETS_WANTED = ("ensemble_flood_extent", "exclusion_mask")

OUT_DIR = Path(__file__).parent / "gfm_all"
OUT_DIR.mkdir(exist_ok=True)

LOG_FILE = OUT_DIR / "download_log.txt"


def log(msg: str):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


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
            log(f"    attempt {attempt+1} failed: {last_err[:100]}")
    return False, last_err


catalog = Client.open(STAC_URL)
log(f"Connected to STAC: {catalog.title}")
log("Archive window: 2016 to 2026 (yearly chunks)\n")

def search_chunked(catalog, bbox: list, start_year: int = 2016, end_year: int = 2026) -> list:
    """Search GFM STAC in yearly chunks to avoid server pagination drop."""
    all_items = []
    for year in range(start_year, end_year + 1):
        window = f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z"
        for attempt in range(3):
            try:
                search = catalog.search(
                    collections=["GFM"],
                    bbox=bbox,
                    datetime=window,
                )
                items = list(search.items())
                all_items.extend(items)
                break
            except Exception as e:
                if attempt == 2:
                    log(f"  WARNING: {year} chunk failed after 3 attempts: {e}")
                else:
                    import time; time.sleep(5)
    return all_items


grand_ok = grand_fail = 0

for district, bbox in DISTRICTS.items():
    log(f"\n{'='*60}")
    log(f"DISTRICT: {district.upper()}  bbox={bbox}")
    log(f"{'='*60}")

    dist_dir = OUT_DIR / district
    dist_dir.mkdir(exist_ok=True)

    items = search_chunked(catalog, bbox)
    dates = sorted(set(i.datetime.strftime("%Y-%m-%d") for i in items))
    log(f"Found {len(items)} items across {len(dates)} dates")
    log(f"Dates: {dates}\n")

    ok_dist = fail_dist = 0

    for date_str in dates:
        date_items = [i for i in items if i.datetime.strftime("%Y-%m-%d") == date_str]
        date_dir = dist_dir / date_str
        date_dir.mkdir(exist_ok=True)
        log(f"  [{date_str}] {len(date_items)} tiles")

        for item in date_items:
            for asset_key in ASSETS_WANTED:
                asset = item.assets.get(asset_key)
                if not asset:
                    continue
                fname = os.path.basename(asset.href)
                dest  = date_dir / fname
                ok, msg = download_verify(asset.href, dest)
                status = "OK  " if ok else "FAIL"
                log(f"    {status}  {fname}  ({msg})")
                if ok:
                    ok_dist += 1
                else:
                    fail_dist += 1

    log(f"\n  {district}: {ok_dist} ok, {fail_dist} failed")
    grand_ok   += ok_dist
    grand_fail += fail_dist

log(f"\n{'='*60}")
log(f"TOTAL: {grand_ok} files ok, {grand_fail} failed")
log(f"Output: {OUT_DIR}")
log(f"Finished: {datetime.now()}")
