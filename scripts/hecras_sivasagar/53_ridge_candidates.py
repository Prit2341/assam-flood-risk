"""Terrain-only ridge candidate screen for A2 (not a levee survey or LANDMARK).

All rules registered in docs/teacher_improvement_review_2026-11-16.md.
No satellite labels are read. No terrain/model is modified.
"""
from pathlib import Path
import json
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, Window
from scipy.ndimage import median_filter
from shapely.geometry import LineString
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "digital_twin/sivasagar/data"
OUT = D / "ridge_a2"
D8 = {1: (-1, 1), 2: (0, 1), 4: (1, 1), 8: (1, 0),
      16: (1, -1), 32: (0, -1), 64: (-1, -1), 128: (-1, 0)}


def split_segments(points, accepted):
    segments, current = [], []
    for p, ok in zip(points, accepted):
        if not ok or (current and np.linalg.norm(p-current[-1]) > 180):
            if len(current) >= 2:
                segments.append(LineString(current))
            current = []
        if ok:
            current.append(p)
    if len(current) >= 2:
        segments.append(LineString(current))
    return [s for s in segments if s.length >= 1000]


def main():
    OUT.mkdir(exist_ok=True)
    with rasterio.open(D / "dem_raw.tif") as src:
        z, tr, bounds = src.read(1), src.transform, src.bounds
    arrays = []
    for name in ["d8_accum_full.tif", "d8_pointer_full.tif"]:
        with rasterio.open(D / name) as src:
            w = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            w = Window(w.col_off, w.row_off, z.shape[1], z.shape[0])
            if not src.window_transform(w).almost_equals(tr):
                raise ValueError("D8/terrain grids differ")
            arrays.append(src.read(1, window=w))
    acc, ptr = arrays
    inside = geometry_mask(gpd.read_file(D / "boundary_utm46n.gpkg").geometry,
                           out_shape=z.shape, transform=tr, invert=True)
    mask = inside & (acc * 900e-6 >= 1000) & (z < 110) & (z > 0)
    rows, cols = np.nonzero(mask)
    nodes = set(zip(rows.tolist(), cols.tolist()))
    following, indegree = {}, dict.fromkeys(nodes, 0)
    for rc in nodes:
        dr, dc = D8.get(int(ptr[rc]), (0, 0))
        nxt = (rc[0]+dr, rc[1]+dc)
        if nxt in nodes and nxt != rc:
            following[rc] = nxt
            indegree[nxt] += 1
    seen_edges, chains = set(), []
    for start in sorted(nodes):
        if indegree[start] == 1:
            continue
        path, cur = [start], start
        while cur in following and (cur, following[cur]) not in seen_edges:
            nxt = following[cur]
            seen_edges.add((cur, nxt)); path.append(nxt); cur = nxt
            if indegree[cur] != 1:
                break
        if len(path) >= 2:
            xy = np.array([tr * (c+.5, r+.5) for r, c in path])
            line = LineString(xy).simplify(45)
            if line.length >= 2000:
                chains.append(line)

    def sample(xy):
        rr, cc = rasterio.transform.rowcol(tr, xy[..., 0], xy[..., 1])
        rr, cc = np.asarray(rr), np.asarray(cc)
        valid = (rr >= 0) & (cc >= 0) & (rr < z.shape[0]) & (cc < z.shape[1])
        vals = np.full(rr.shape, np.nan)
        vals[valid] = z[rr[valid], cc[valid]]
        vals[(vals <= 0) | (vals >= 150)] = np.nan
        return vals

    ridge_rows = []
    for chain_id, line in enumerate(chains):
        s = np.arange(180, line.length-180, 90)
        xy = np.array([line.interpolate(t).coords[0] for t in s])
        left = np.array([line.interpolate(t-150).coords[0] for t in s])
        right = np.array([line.interpolate(t+150).coords[0] for t in s])
        tangent = right-left
        normal = np.c_[-tangent[:, 1], tangent[:, 0]]
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        for side in [-1, 1]:
            offsets = np.arange(30, 271, 30)
            profile_xy = xy[:, None, :] + side*normal[:, None, :]*offsets[None, :, None]
            zz = sample(profile_xy.reshape(-1, 2)).reshape(len(s), -1)
            smooth = median_filter(zz, size=(1, 3), mode="nearest")
            # Select between 60 and 240 m, then require a local maximum.
            k = np.argmax(np.nan_to_num(smooth[:, 1:-1], nan=-9999), axis=1)+1
            ix = np.arange(len(s)); peak = smooth[ix, k]
            outer_xy = xy[:, None, :] + side*normal[:, None, :]*np.arange(330, 601, 30)[None, :, None]
            outer = sample(outer_xy.reshape(-1, 2)).reshape(len(s), -1)
            reference = np.nanmedian(outer, axis=1)
            accepted = (np.isfinite(peak) & np.isfinite(reference) &
                        (peak-reference >= 1) & (peak-zz[:, 0] >= .5) &
                        (peak >= smooth[ix, k-1]) & (peak >= smooth[ix, k+1]))
            points = profile_xy[ix, k]
            for segment in split_segments(points, accepted):
                ridge_rows.append(dict(chain=chain_id, side=side,
                    length_m=segment.length, status="terrain_candidate_not_surveyed",
                    geometry=segment))
    report = dict(major_stream_pixels=int(mask.sum()), stream_chains=len(chains),
                  stream_length_km=sum(l.length for l in chains)/1000,
                  ridge_segments=len(ridge_rows), ridge_length_km=sum(r["length_m"] for r in ridge_rows)/1000,
                  inputs="raw FABDEM, D8 accumulation and pointer, domain; no flood labels")
    if ridge_rows:
        gpd.GeoDataFrame(ridge_rows, crs=32646).to_file(OUT / "ridge_candidates.gpkg", driver="GPKG")
    if chains:
        gpd.GeoDataFrame(geometry=chains, crs=32646).to_file(OUT / "major_streams.gpkg", driver="GPKG")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(z, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
              vmin=80, vmax=115, cmap="terrain", interpolation="nearest")
    for line in chains:
        ax.plot(*line.xy, color="#1681cc", linewidth=.65)
    for row in ridge_rows:
        ax.plot(*row["geometry"].xy, color="#ff3c12", linewidth=1.2)
    if chains:
        box = gpd.GeoSeries(chains).total_bounds
        ax.set_xlim(box[0]-1200, box[2]+1200); ax.set_ylim(box[1]-1200, box[3]+1200)
    ax.set_title("A2 terrain-only ridge screen: blue streams, red candidate ridges\nNot verified embankments; raw terrain unchanged")
    ax.set_xlabel("UTM 46N easting (m)"); ax.set_ylabel("northing (m)")
    fig.tight_layout(); fig.savefig(OUT / "ridge_candidates.png", dpi=160); plt.close(fig)
    (OUT / "screen.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
