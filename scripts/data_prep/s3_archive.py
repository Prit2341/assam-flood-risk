"""
Archive of this project's data and code to S3 bucket assam-thesis.

SAFETY (by design, not by the IAM policy, which is broad):
  - only the one bucket and prefix set below; any other target is refused
  - NEVER deletes anything, locally or on S3 (the IAM user also has an explicit Deny on delete)
  - default is a DRY RUN that only reads local sizes and lists what is already on S3; --upload is needed to write
  - a file is skipped only if S3 has the same size AND the local file was not modified after the S3 copy was written;
    so interrupted runs can be re-run, and files edited since the last upload (e.g. kb_chroma/db) are uploaded again
    (with bucket versioning on, the old version is kept)
Verification after each upload: size on S3 == local size, and the SHA-256 that S3 computed and checked at upload is read back and recorded
(S3 rejects the upload if the data does not match the checksum sent; this is NOT an independent re-download and re-hash).
Credentials: default boto3 profile (C:\\Users\\<user>\\.aws), never read or printed by this script.

Usage:
    python scripts/data_prep/s3_archive.py --list-groups
    python scripts/data_prep/s3_archive.py --group all                      # dry run of everything
    python scripts/data_prep/s3_archive.py --group all --upload             # writes to S3, one S3_INDEX.md row per finished file
    python scripts/data_prep/s3_archive.py --group core --group project --upload --workers 8
    python scripts/data_prep/s3_archive.py --group core --rebuild-index     # after a cancelled run: add index rows for files already on S3
"""
import argparse
import fnmatch
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

BUCKET = "assam-thesis"
PREFIX = ""                                 # whole bucket belongs to this project; keys = path relative to project root
REGION = "ap-south-1"
ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "S3_INDEX.md"
EXCLUDE_DIRS = {"__pycache__", "node_modules", "miniforge3"}
EXCLUDE_FILES = ["*.tmp", "*.lock", "Thumbs.db", "S3_INDEX.md"]      # S3_INDEX.md is uploaded once, last, by the script itself
MTIME_SLACK = 2.0                           # seconds of tolerance between local mtime and S3 LastModified

GROUPS = {
    # core = derived / irreplaceable (outputs, models, validation products, docs)
    "core": ["flood_susceptibility", "models", "releases", "docs", "Mid_Sem_phase_1", "dashboard-app", "kb_chroma/db",
             "data/dynamic_susceptibility", "data/gfm_*", "data/rainfall_nowcast", "data/nrsc_*", "data/glofas_*",
             "data/basin_delineation", "data/ifi_impacts", "data/hysplit_trajectories", "data/modis_gfd", "data/lulc_*",
             "data/soilgrids*", "data/worldpop", "kb_chroma/*.py", "data/water_level", "data/gfs_test", "data/logs", "data/icechunk_regional_tp"],
    # raw = re-downloadable from public sources (see CLAUDE.md), about 10 GB
    "raw": ["data/gdas1", "data/icechunk_pressure", "data/boundaries", "data/hydrobasins", "data/hydrorivers",
            "data/era5_longrecord", "data/dem", "data/dem_fabdem"],
    # project = code, notes and loose top-level files
    "project": ["scripts", "tools", "logs", ".vscode", ".claude", "*.md", "*.docx", "*.png", "*.qgz", "*.txt", "*.html", "*.json",
                ".cursorrules", ".windsurfrules", ".gitignore"],
    # git history (small); remember it is only consistent if you are not committing during the upload
    "git": [".git"],
}
GROUPS["all"] = [p for g in ("core", "raw", "project", "git") for p in GROUPS[g]]


def expand(paths):
    out, seen = [], set()
    for p in paths:
        hits = sorted(ROOT.glob(p)) if any(c in p for c in "*?[") else [ROOT / p]
        for h in hits:
            if h.exists() and h not in seen:
                seen.add(h); out.append(h)
    return out


def files_of(path):
    if path.is_file():
        if not any(fnmatch.fnmatch(path.name, pat) for pat in EXCLUDE_FILES):
            yield path
        return
    for dp, dn, fn in os.walk(path):
        dn[:] = [d for d in dn if d not in EXCLUDE_DIRS]
        for f in fn:
            if not any(fnmatch.fnmatch(f, pat) for pat in EXCLUDE_FILES):
                yield Path(dp) / f


def key_of(f):
    return PREFIX + f.relative_to(ROOT).as_posix()


def existing(s3):
    """key -> (size, LastModified epoch seconds)"""
    out = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=PREFIX):
        for o in page.get("Contents", []):
            out[o["Key"]] = (o["Size"], o["LastModified"].timestamp())
    return out


def up_to_date(have, f, st):
    h = have.get(key_of(f))
    return h is not None and h[0] == st.st_size and st.st_mtime <= h[1] + MTIME_SLACK


def indexed_paths():
    if not INDEX.exists():
        return set()
    out = set()
    for line in INDEX.read_text(encoding="utf-8").splitlines():
        if line.startswith("| ") and not line.startswith("| local path") and not line.startswith("|---"):
            out.add(line.split("|")[1].strip())
    return out


_idx_lock = threading.Lock()


def append_index(rows):
    with _idx_lock:
        new = not INDEX.exists()
        with open(INDEX, "a", encoding="utf-8") as fh:
            if new:
                fh.write("# S3 index (what was copied to S3, how to bring it back)\n\nBucket s3://%s/%s (region %s). Restore a file by downloading its key to the same relative path "
                         "under the project root. A file uploaded again after local edits gets a second row (newest date is current). "
                         "Local copies are deleted only by Prit, per folder, after verification.\n\n| local path | S3 key | bytes | SHA-256 (base64, S3-computed) | date |\n|---|---|---|---|---|\n" % (BUCKET, PREFIX, REGION))
            for r in rows:
                fh.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {time.strftime('%Y-%m-%d')} |\n")


def upload_one(s3, cfg, f, sz):
    k = key_of(f)
    s3.upload_file(str(f), BUCKET, k, Config=cfg, ExtraArgs={"ChecksumAlgorithm": "SHA256"})
    h = s3.head_object(Bucket=BUCKET, Key=k, ChecksumMode="ENABLED")
    if h["ContentLength"] != sz:
        raise RuntimeError(f"SIZE MISMATCH after upload: {k} local {sz} s3 {h['ContentLength']}")
    return (f.relative_to(ROOT).as_posix(), k, sz, h.get("ChecksumSHA256", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append")
    ap.add_argument("--list-groups", action="store_true")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--workers", type=int, default=8, help="files uploaded in parallel (helps with many small files)")
    ap.add_argument("--rebuild-index", action="store_true",
                    help="add S3_INDEX.md rows for files already on S3 (same size) that have none; reads S3 metadata only")
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--prefix", default=PREFIX)
    a = ap.parse_args()
    if a.bucket != BUCKET or a.prefix != PREFIX:
        sys.exit(f"refused: this script only writes to s3://{BUCKET}/{PREFIX}")
    if a.list_groups or not a.group:
        for g, p in GROUPS.items():
            print(f"{g}: {p}")
        return
    s3 = boto3.Session(region_name=REGION).client("s3", config=Config(retries={"max_attempts": 10, "mode": "standard"},
                                                                       max_pool_connections=max(10, a.workers * 5)))
    have = existing(s3)
    print(f"target s3://{BUCKET}/{PREFIX}  ({len(have)} objects already there)  mode: {'UPLOAD' if a.upload else 'DRY RUN'}  workers: {a.workers}")
    cfg = TransferConfig(multipart_threshold=64 * 2**20, multipart_chunksize=64 * 2**20, max_concurrency=4)
    grand = grand_todo = 0
    for g in a.group:
        paths = expand(GROUPS[g]); todo = []; n = size = skip = 0
        for p in paths:
            for f in files_of(p):
                try:
                    st = f.stat()
                except OSError:
                    continue
                n += 1; size += st.st_size
                if up_to_date(have, f, st):
                    skip += 1
                else:
                    todo.append((f, st.st_size))
        todo_gb = sum(s for _, s in todo) / 1e9
        print(f"\n[{g}] {len(paths)} paths, {n} files, {size / 1e9:.2f} GB; already on S3 and unchanged: {skip}; to upload: {len(todo)} files, {todo_gb:.2f} GB")
        for f, sz in sorted(todo, key=lambda t: -t[1])[:8]:
            print(f"   {sz / 1e9:7.3f} GB  {f.relative_to(ROOT).as_posix()}")
        grand += size; grand_todo += todo_gb
        if a.rebuild_index:
            have_rows = indexed_paths(); add = []
            for p in paths:
                for f in files_of(p):
                    rel = f.relative_to(ROOT).as_posix(); k = key_of(f)
                    if rel in have_rows or k not in have or have[k][0] != f.stat().st_size:
                        continue
                    h = s3.head_object(Bucket=BUCKET, Key=k, ChecksumMode="ENABLED")
                    add.append((rel, k, h["ContentLength"], h.get("ChecksumSHA256", "")))
            if add:
                append_index(add)
            print(f"[{g}] index rebuilt: {len(add)} rows added")
            continue
        if not a.upload or not todo:
            continue
        t0 = time.time(); done = 0; nfiles = 0
        todo.sort(key=lambda t: -t[1])          # big files first, small ones fill the tail
        ex = ThreadPoolExecutor(a.workers)
        try:
            futs = {ex.submit(upload_one, s3, cfg, f, sz): sz for f, sz in todo}
            for fu in as_completed(futs):
                row = fu.result()                # raises on any failure -> run stops, re-run resumes
                append_index([row])
                nfiles += 1; done += futs[fu]
                if nfiles % 100 == 0 or futs[fu] > 5e8:
                    print(f"   {nfiles}/{len(todo)} files, {done / 1e9:.2f}/{todo_gb:.2f} GB, {time.time() - t0:.0f}s", flush=True)
        except BaseException:
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        ex.shutdown()
        print(f"[{g}] uploaded {nfiles} files, {done / 1e9:.2f} GB in {time.time() - t0:.0f}s; sizes verified against S3")
    if a.upload and not a.rebuild_index:
        row = upload_one(s3, cfg, INDEX, INDEX.stat().st_size)      # the index itself, last (its own row is not listed in it)
        print(f"uploaded {row[0]} ({row[2]} bytes)")
    print(f"\ntotal selected {grand / 1e9:.2f} GB, still to upload {grand_todo:.2f} GB")


if __name__ == "__main__":
    main()
